"""Stage 2: latent super-resolution world model (V-JEPA 2-AC analogue, paper eqs. 2-4).

The frozen encoder embeds every ladder level z_k = E(level_k), k = 0 (noisy LR measurement) ... K (truth).
The block-causal predictor P plays the AC world model with resolution as time:
    teacher forcing  L_tf = 1/K sum_k || P((c, z_j)_{j<=k})_k - z_{k+1} ||_1
    rollout          L_ro = || P^K(z_0) - z_K ||_1  (autoregressive from the measurement, gradient through the
                     last step only, as V-JEPA 2-AC)
    L = L_tf + rollout_weight * L_ro
c is the condition token (level, and E_inc for showers, like the AC action/state tokens). Only P is trained.
"""
import torch

from ..data.datasets import make_loader
from ..models.build import build_sr_predictor
from ..pipeline import encode_levels, frozen_encoder
from .common import Trainer, autocast, batches, setup, stage_dir, train_augment


def sr_losses(P, zs, cond, grid, grad_steps=1):
    K = len(zs) - 1
    preds = P(zs[:-1], cond, grid)
    tf = torch.stack([(p.float() - z).abs().mean() for p, z in zip(preds, zs[1:])]).mean()
    ro = (P.rollout(zs[0], cond, grid, K, grad_steps)[-1].float() - zs[-1]).abs().mean() if K > 1 else tf
    return tf, ro


def run(cfg):
    device, meta, L = setup(cfg)
    sec = cfg["sr"]
    enc = frozen_encoder(cfg, L, device)
    P = build_sr_predictor(cfg, L).to(device)
    print(f"SR predictor {sum(p.numel() for p in P.parameters()) / 1e6:.1f}M params, levels {L.levels}")
    loader = make_loader(cfg, "train", sec["batch_size"], True, sec["max_train"], device=device)
    aug = train_augment(cfg, L)
    vloader = make_loader(cfg, "val", sec["batch_size"], False, sec["max_val"], device=device)
    out = stage_dir(cfg, "sr")
    tr = Trainer(cfg, sec, out, {"predictor": P}, [(P, 1.0)], len(loader))
    grid = L.grid()
    while not tr.done:
        P.train()
        for _, hr, lr, t in batches(loader, device, aug):
            with torch.no_grad():
                zs = encode_levels(enc, L, hr, lr)
            with autocast(cfg, device):
                tf, ro = sr_losses(P, zs, L.cond(t), grid)
            tr.step(tf + sec["rollout_weight"] * ro, tf=tf, rollout=ro)
        tr.end_epoch(validate(cfg, L, enc, P, vloader, device))
    return out


@torch.no_grad()
def validate(cfg, L, enc, P, loader, device):
    """loss = rollout L1 from the measurement alone (the inference path); identity = || z_0 - z_K ||_1, the
    latent error of not super-resolving at all."""
    P.eval()
    acc = {"loss": 0.0, "teacher_forcing": 0.0, "identity": 0.0}
    n = 0
    for _, hr, lr, t in batches(loader, device):
        zs = encode_levels(enc, L, hr, lr)
        with autocast(cfg, device):
            tf, ro = sr_losses(P, zs, L.cond(t), L.grid())
        acc["loss"] += float(ro)
        acc["teacher_forcing"] += float(tf)
        acc["identity"] += float((zs[0] - zs[-1]).abs().mean())
        n += 1
    return {k: v / max(n, 1) for k, v in acc.items()}
