"""Ablation: encoder -> SR predictor -> decoder trained end to end on pixels, from the LR measurement only.

init jepa     start from the three stage-1..3 checkpoints; the pretrained encoder is fine-tuned with
              encoder_lr_scale x the learning rate
init scratch  the same architecture from random weights: the supervised baseline. Comparing the two (and the
              frozen `jepa` variant) shows what self-supervised pretraining contributes.
"""
import os

import torch

from ..data.datasets import make_loader
from ..config import tuned_report
from ..pipeline import SECTIONS, SRPipeline, checkpoint_paths, pixel_loss
from .common import Trainer, autocast, batches, load_modules, setup, stage_dir, train_augment


def section(cfg, init):
    """The e2e settings of one init: the shared e2e keys, overridden by e2e.per_init.<init> (tuned per init)."""
    sec = {k: v for k, v in cfg["e2e"].items() if k != "per_init"}
    sec.update((cfg["e2e"].get("per_init") or {}).get(init) or {})
    return sec


def run(cfg, init, on_epoch=None):
    if init not in ("jepa", "scratch"):
        raise ValueError(f"e2e init must be jepa or scratch, not {init}")
    tuned_report(cfg, f"e2e_{init}")
    device, meta, L = setup(cfg)
    sec = section(cfg, init)
    pipe = SRPipeline(cfg, L)
    if init == "jepa":
        for name, (path, key) in checkpoint_paths(cfg, "jepa").items():
            if os.path.exists(path):
                load_modules(path, {key: getattr(pipe, name)}, cfg=cfg, sections=SECTIONS[name])
                print(f"[e2e_jepa] {name} <- {path}")
            elif name == "encoder":
                raise FileNotFoundError(f"{path} missing: run `pretrain` first")
    pipe = pipe.to(device)
    loader = make_loader(cfg, "train", sec["batch_size"], True, sec["max_train"], device=device)
    aug = train_augment(cfg, L)
    vloader = make_loader(cfg, "val", sec["batch_size"], False, sec["max_val"], device=device)
    out = stage_dir(cfg, f"e2e_{init}")
    enc_scale = sec["encoder_lr_scale"] if init == "jepa" else 1.0
    tr = Trainer(cfg, sec, out, pipe.modules_dict(),
                 [(pipe.encoder, enc_scale), (pipe.predictor, 1.0), (pipe.decoder, 1.0)], len(loader), on_epoch)
    while not tr.done:
        pipe.train()
        for _, hr, lr, t in batches(loader, device, aug):
            with autocast(cfg, device):
                y, _ = pipe(lr, t)
                loss = pixel_loss(y, L.normalize(hr), L, sec["energy_weight"])
            tr.step(loss)
        tr.end_epoch(validate(cfg, L, pipe, vloader, device))
    return out


@torch.no_grad()
def validate(cfg, L, pipe, loader, device):
    pipe.eval()
    tot, n = 0.0, 0
    for _, hr, lr, t in batches(loader, device):
        with autocast(cfg, device):
            tot += float(pixel_loss(pipe(lr, t)[0], L.normalize(hr), L))
        n += 1
    return {"loss": tot / max(n, 1)}
