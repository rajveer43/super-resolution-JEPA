"""Stage 3: pixel decoder, latent tokens (+ LR measurement) -> HR log energy densities.

Encoder and SR predictor stay frozen. Each event is decoded either from the truth latent E(HR) or, with
probability pred_latent_prob, from the predictor's rollout latent, so the decoder learns the latent
distribution it will meet at inference. Loss: L1 on log energy densities + energy_weight * relative event
energy error. Model selection on the inference path (decoding the predicted latent).
"""
import os

import torch

from ..data.datasets import make_loader
from ..models.build import build_decoder, build_sr_predictor
from ..pipeline import frozen_encoder, latents, pixel_loss
from .common import Trainer, autocast, batches, load_modules, setup, stage_dir, train_augment


def run(cfg):
    device, meta, L = setup(cfg)
    sec = cfg["decoder"]
    enc = frozen_encoder(cfg, L, device)
    P = None
    p_pred = float(sec["pred_latent_prob"])
    sr_ck = os.path.join(cfg["paths"]["out"], "sr", "best.pt")
    if os.path.exists(sr_ck):
        P = build_sr_predictor(cfg, L)
        load_modules(sr_ck, {"predictor": P})
        P = P.to(device).eval().requires_grad_(False)
    elif p_pred > 0:
        print(f"[decoder] {sr_ck} missing: training on E(HR) latents only")
        p_pred = 0.0
    D = build_decoder(cfg, L).to(device)
    loader = make_loader(cfg, "train", sec["batch_size"], True, sec["max_train"], device=device)
    aug = train_augment(cfg, L)
    vloader = make_loader(cfg, "val", sec["batch_size"], False, sec["max_val"], device=device)
    out = stage_dir(cfg, "decoder")
    tr = Trainer(cfg, sec, out, {"decoder": D}, [(D, 1.0)], len(loader))
    grid, K = L.grid(), len(L.levels) - 1
    g = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]) + tr.step_n)
    while not tr.done:
        D.train()
        for _, hr, lr, t in batches(loader, device, aug):
            with torch.no_grad():
                x_hr = L.normalize(hr)
                x_lr = L.normalize(L.level_image(hr, lr, 0))
                z = latents(enc, x_hr)
                if p_pred > 0:  # whole-batch rollout + select: no data-dependent shapes, no GPU sync
                    use = (torch.rand(len(z), generator=g) < p_pred).to(device).view(-1, 1, 1)
                    zp = P.rollout(latents(enc, x_lr), L.cond(t), grid, K)[-1].float()
                    z = torch.where(use, zp, z)
            with autocast(cfg, device):
                loss = pixel_loss(D(z, grid, x_lr), x_hr, L, sec["energy_weight"])
            tr.step(loss)
        tr.end_epoch(validate(cfg, L, enc, P, D, vloader, device))
    return out


@torch.no_grad()
def validate(cfg, L, enc, P, D, loader, device):
    """loss: decoding the SR rollout latent (inference path; = autoencoding if no predictor);
    autoencode: decoding E(HR), the decoder's ceiling."""
    D.eval()
    grid, K = L.grid(), len(L.levels) - 1
    acc = {"loss": 0.0, "autoencode": 0.0}
    n = 0
    for _, hr, lr, t in batches(loader, device):
        x_hr, x_lr = L.normalize(hr), L.normalize(L.level_image(hr, lr, 0))
        z_hr = latents(enc, x_hr)
        with autocast(cfg, device):
            ae = pixel_loss(D(z_hr, grid, x_lr), x_hr, L)
            if P is not None:
                z = P.rollout(latents(enc, x_lr), L.cond(t), grid, K)[-1]
                sr = pixel_loss(D(z, grid, x_lr), x_hr, L)
            else:
                sr = ae
        acc["loss"] += float(sr)
        acc["autoencode"] += float(ae)
        n += 1
    return {k: v / max(n, 1) for k, v in acc.items()}
