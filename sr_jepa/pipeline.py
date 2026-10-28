"""The SR pipeline LR -> E -> latent world-model rollout -> decoder -> HR energies, and its checkpoints.

Variants (directories under runs/<dataset>/<run_name>/):
    jepa         frozen JEPA encoder (pretrain/last.pt, EMA weights) + sr/best.pt + decoder/best.pt
    e2e_jepa     the same three networks fine-tuned end to end on pixels (e2e_jepa/best.pt)
    e2e_scratch  the same architecture trained end to end from random init (e2e_scratch/best.pt): the ablation
                 that shows what self-supervised pretraining adds
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.build import build_decoder, build_encoder, build_sr_predictor
from .train.common import load_modules, upstream_dir

VARIANTS = ("jepa", "e2e_jepa", "e2e_scratch")


def latents(encoder, x, keep=None):
    """Latent space of the whole pipeline: layer-normed encoder tokens (the V-JEPA target space)."""
    h = encoder(x, keep)
    return F.layer_norm(h.float(), (h.shape[-1],))


def encode_levels(encoder, L, hr, lr):
    return [latents(encoder, L.normalize(L.level_image(hr, lr, k))) for k in range(len(L.levels))]


class SRPipeline(nn.Module):
    def __init__(self, cfg, L):
        super().__init__()
        self.L = L
        self.encoder = build_encoder(cfg, L)
        self.predictor = build_sr_predictor(cfg, L)
        self.decoder = build_decoder(cfg, L)
        self.grid = L.grid()
        self.steps = len(L.levels) - 1

    def forward(self, lr, target, grad_steps=None):
        """lr (B, P, u, v) measured energies -> (normalised HR prediction (B, C, D, H, W), predicted latent)."""
        x_lr = self.L.normalize(self.L.level_image(None, lr, 0))
        z0 = latents(self.encoder, x_lr)
        z = self.predictor.rollout(z0, self.L.cond(target), self.grid, self.steps, grad_steps)[-1]
        return self.decoder(z, self.grid, x_lr), z

    def modules_dict(self):
        return {"encoder": self.encoder, "predictor": self.predictor, "decoder": self.decoder}


# config sections that must match a checkpoint of each network (load_modules checks them)
SECTIONS = {"encoder": ("model", "encoder"), "predictor": ("model", "encoder", "sr"),
            "decoder": ("model", "encoder", "decoder")}


def frozen_encoder(cfg, L, device):
    enc = build_encoder(cfg, L)
    load_modules(os.path.join(upstream_dir(cfg), "pretrain", "last.pt"), {"target_encoder": enc}, cfg=cfg,
                 sections=SECTIONS["encoder"])
    return enc.to(device).eval().requires_grad_(False)


def checkpoint_paths(cfg, variant):
    out = upstream_dir(cfg)
    if variant == "jepa":
        return {"encoder": (os.path.join(out, "pretrain", "last.pt"), "target_encoder"),
                "predictor": (os.path.join(out, "sr", "best.pt"), "predictor"),
                "decoder": (os.path.join(out, "decoder", "best.pt"), "decoder")}
    p = os.path.join(out, variant, "best.pt")
    return {k: (p, k) for k in ("encoder", "predictor", "decoder")}


def available(cfg, variant):
    return all(os.path.exists(p) for p, _ in checkpoint_paths(cfg, variant).values())


def load_pipeline(cfg, L, device, variant="jepa"):
    pipe = SRPipeline(cfg, L)
    for name, (path, key) in checkpoint_paths(cfg, variant).items():
        load_modules(path, {key: getattr(pipe, name)}, cfg=cfg, sections=SECTIONS[name])
    return pipe.to(device).eval().requires_grad_(False)


@torch.no_grad()
def super_resolve(pipe, lr, target, project=False):
    """-> SR energies (B, P, U, V) on the native grid (zero-suppressed), predicted latent."""
    L = pipe.L
    y, z = pipe(lr, target)
    E = L.clean(L.denormalize(y))
    if project:
        E = L.project_to_lr(E, lr)
    return E, z


def pixel_loss(y, x_hr, L, energy_weight=0.0):
    """L1 on log energy densities, plus the relative error of the event energy."""
    loss = (y.float() - x_hr).abs().mean()
    if energy_weight:
        e_pred = L.denormalize(y).sum(dim=(1, 2, 3))
        e_true = L.denormalize(x_hr).sum(dim=(1, 2, 3))
        loss = loss + energy_weight * ((e_pred - e_true).abs() / e_true.clamp_min(1e-6)).mean()
    return loss
