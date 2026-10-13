"""Shared training machinery: device, seeding, warmup-constant-decay schedule, resumable trainer, checkpoints."""
import contextlib
import csv
import math
import os
import random
import time

import numpy as np
import torch

from ..config import save_config
from ..data.datasets import Layout, load_meta


def pick_device(cfg):
    d = cfg["device"]
    if d != "auto":
        return torch.device(d)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast(cfg, device):
    # bf16 only where the GPU has it (A100 / L4 / H100); a T4 runs in fp32
    if device.type == "cuda" and cfg["amp"] and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def setup(cfg):
    seed_all(int(cfg["seed"]))
    meta = load_meta(cfg)
    device = pick_device(cfg)
    if device.type == "cuda":  # TF32 matmuls / convolutions for whatever autocast leaves in fp32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return device, meta, Layout(cfg, meta)


def stage_dir(cfg, name):
    d = os.path.join(cfg["paths"]["out"], name)
    os.makedirs(d, exist_ok=True)
    return d


def param_groups(module, lr_scale=1.0):
    """AdamW groups: no weight decay on biases, norms, embeddings and other 1-d / token parameters."""
    decay, no_decay = [], []
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "token" in n or "emb" in n or "query" in n else decay).append(p)
    return [{"params": decay, "lr_scale": lr_scale, "wd": True},
            {"params": no_decay, "lr_scale": lr_scale, "wd": False}]


def wcd_factor(step, total, sec):
    """Warmup-constant-decay (V-JEPA 2): linear warmup from start_lr_frac, constant, linear cooldown."""
    w = max(1, int(sec["warmup_frac"] * total))
    c0 = int((1 - sec["cooldown_frac"]) * total)
    if step < w:
        return sec["start_lr_frac"] + (1 - sec["start_lr_frac"]) * step / w
    if step < c0:
        return 1.0
    t = min(1.0, (step - c0) / max(1, total - c0))
    return 1.0 + (sec["final_lr_frac"] - 1.0) * t


def in_cooldown(step, total, sec):
    return step >= int((1 - sec["cooldown_frac"]) * total)


@torch.no_grad()
def ema_update(target, online, m):
    """Fused over all parameters: two kernels instead of two per tensor."""
    tp = list(target.parameters())
    torch._foreach_mul_(tp, m)
    torch._foreach_add_(tp, [p.detach() for p in online.parameters()], alpha=1 - m)


def save_modules(path, modules, **extra):
    tmp = path + ".tmp"
    torch.save({"modules": {k: m.state_dict() for k, m in modules.items()}, **extra}, tmp)
    os.replace(tmp, path)


def load_modules(path, modules, strict=True):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run the stage that produces it first")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k, m in modules.items():
        m.load_state_dict(ck["modules"][k], strict=strict)
    return ck


class Trainer:
    """Optimiser + schedule + per-epoch bookkeeping. Resumes from {out}/last.pt; keeps {out}/best.pt (lowest
    validation loss) and {out}/history.csv. `modules` are everything saved; `trainable` [(module, lr_scale)]."""

    def __init__(self, cfg, sec, out, modules, trainable, steps_per_epoch):
        self.cfg, self.sec, self.out, self.modules = cfg, sec, out, modules
        groups = [g for m, s in trainable for g in param_groups(m, s)]
        for g in groups:
            g["weight_decay"] = sec["weight_decay"] if g.pop("wd") else 0.0
        # peak lr, scaled by sqrt(batch / lr_ref_batch) when the batch differs from the one lr was set for
        ref = sec.get("lr_ref_batch")
        self.lr = sec["lr"] * (math.sqrt(sec["batch_size"] / ref) if ref else 1.0)
        self.opt = torch.optim.AdamW(groups, lr=self.lr, betas=(0.9, 0.999))
        self.params = [p for g in groups for p in g["params"]]
        self.total = max(1, int(sec["epochs"]) * steps_per_epoch)
        self.step_n, self.epoch, self.best, self.bad = 0, 0, math.inf, 0
        self._acc = {}
        self._t0 = self._te = time.time()
        self._epoch_steps = 0
        save_config(cfg, os.path.join(out, "config.json"))
        last = os.path.join(out, "last.pt")
        if os.path.exists(last):
            ck = load_modules(last, modules)
            self.opt.load_state_dict(ck["opt"])
            self.step_n, self.epoch, self.best, self.bad = ck["step"], ck["epoch"], ck["best"], ck["bad"]
            print(f"[{os.path.basename(out)}] resumed at epoch {self.epoch}, step {self.step_n}")

    @property
    def done(self):
        return self.epoch >= int(self.sec["epochs"]) or (self.sec.get("patience") and self.bad >= self.sec["patience"])

    def set_lr(self):
        f = wcd_factor(self.step_n, self.total, self.sec)
        for g in self.opt.param_groups:
            g["lr"] = self.lr * f * g["lr_scale"]
        return f

    def step(self, loss, **logs):
        self.set_lr()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.sec.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(self.params, self.sec["grad_clip"])
        self.opt.step()
        self.step_n += 1
        self._epoch_steps += 1
        # accumulate on the device: converting to Python floats here would sync with the GPU every step
        for k, v in {"loss": loss, **logs}.items():
            s, n = self._acc.get(k, (0.0, 0))
            self._acc[k] = (s + (v.detach().float() if torch.is_tensor(v) else float(v)), n + 1)

    def end_epoch(self, val):
        """val: dict with at least 'loss'. Writes history, checkpoints; prints a line."""
        self.epoch += 1
        row = {"epoch": self.epoch, "step": self.step_n, "lr": self.lr * wcd_factor(self.step_n, self.total, self.sec)}
        row.update({f"train_{k}": float(s) / max(n, 1) for k, (s, n) in self._acc.items()})
        row.update({f"val_{k}": float(v) for k, v in val.items()})
        now = time.time()
        row["events_per_s"] = self._epoch_steps * self.sec["batch_size"] / max(now - self._te, 1e-9)
        row["minutes"] = (now - self._t0) / 60
        self._acc, self._te, self._epoch_steps = {}, now, 0
        improved = val["loss"] < self.best - 1e-6
        if improved:
            self.best, self.bad = float(val["loss"]), 0
            save_modules(os.path.join(self.out, "best.pt"), self.modules, epoch=self.epoch, val=val)
        else:
            self.bad += 1
        save_modules(os.path.join(self.out, "last.pt"), self.modules, opt=self.opt.state_dict(), step=self.step_n,
                     epoch=self.epoch, best=self.best, bad=self.bad)
        path = os.path.join(self.out, "history.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[{os.path.basename(self.out)}] epoch {self.epoch}/{self.sec['epochs']} "
              + " ".join(f"{k}={v:.4g}" for k, v in row.items() if k not in ("epoch",)) + (" *" if improved else ""))
        return self.done


def batches(loader, device, augment=None):
    """(index, hr, lr, target) on `device` (a no-op for splits preloaded there); augment(hr, lr) on training batches."""
    for i, hr, lr, tgt in loader:
        hr, lr = hr.to(device, non_blocking=True), lr.to(device, non_blocking=True)
        if augment is not None:
            hr, lr = augment(hr, lr)
        yield i, hr, lr, tgt.to(device, non_blocking=True)


def train_augment(cfg, L):
    return L.augment if cfg["data"].get("augment", True) else None
