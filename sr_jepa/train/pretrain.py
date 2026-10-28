"""Stage 1: V-JEPA 2 mask-denoising pretraining in representation space (paper eq. 1).

    min_{theta, phi, Delta}  || P_phi(Delta_y, E_theta(x)) - sg(E_bar(y)) ||_1   on the masked tokens only

x is the event with multiblock-masked tokens dropped, y the full event, E_bar the EMA of E_theta with a fixed
momentum. Targets are layer-normed. Inputs are drawn from every ladder level (noisy LR measurement,
intermediate grids, truth) with `view_weights`, so the frozen encoder later embeds measurement and truth in
one space. Progressive resolution: inputs are sum-pooled by `main_pool` until the cooldown starts.
No labels are used.
"""
import copy

import torch

from ..config import tuned_report
from ..data.datasets import make_loader
from ..masks import sample_masks
from ..models.build import build_encoder, build_mask_predictor
from ..models.vit import gather_tokens
from ..pipeline import latents
from .common import Trainer, autocast, batches, ema_update, in_cooldown, setup, stage_dir, train_augment


def jepa_loss(enc, pred, target_enc, x, grid, mask_specs, g):
    with torch.no_grad():
        h = latents(target_enc, x)
    losses = []
    for spec in mask_specs:
        ctx, tgt = (t.to(x.device) for t in sample_masks(len(x), grid, spec, g))
        p = pred(enc(x, keep=ctx), ctx, tgt, grid)
        losses.append((p.float() - gather_tokens(h, tgt)).abs().mean())
    return torch.stack(losses).mean()


@torch.no_grad()
def feature_std(enc, x):
    """Collapse monitor: per-dimension std over events of the mean-pooled raw features (-> 0 if collapsed)."""
    return enc(x).float().mean(1).std(0).mean()


def run(cfg, on_epoch=None):
    tuned_report(cfg, "pretrain")
    device, meta, L = setup(cfg)
    sec = cfg["pretrain"]
    main_pool = tuple(int(p) for p in sec["main_pool"])
    for pool in (main_pool, (1, 1, 1)):
        for n, p, q, a in zip((L.D, L.H, L.W), L.patch, pool, "DHW"):
            if n % (p * q):
                raise ValueError(f"pretrain.main_pool {main_pool} x patch {L.patch} does not divide {a}={n}")
    weights = torch.tensor(sec["view_weights"], dtype=torch.float)
    if len(weights) != len(L.levels):
        raise ValueError(f"pretrain.view_weights needs one weight per ladder level {L.levels}")

    enc = build_encoder(cfg, L).to(device)
    pred = build_mask_predictor(cfg, L).to(device)
    target = copy.deepcopy(enc).requires_grad_(False)
    print(f"encoder {sum(p.numel() for p in enc.parameters()) / 1e6:.1f}M params, "
          f"predictor {sum(p.numel() for p in pred.parameters()) / 1e6:.1f}M; token grid {L.grid()} "
          f"(main phase {L.grid(main_pool)}); levels {L.levels}")
    loader = make_loader(cfg, "train", sec["batch_size"], True, sec["max_train"], device=device)
    aug = train_augment(cfg, L)
    vloader = make_loader(cfg, "val", sec["batch_size"], False, sec["max_val"], device=device)
    out = stage_dir(cfg, "pretrain")
    tr = Trainer(cfg, sec, out, {"encoder": enc, "predictor": pred, "target_encoder": target},
                 [(enc, 1.0), (pred, 1.0)], len(loader), on_epoch)
    g = torch.Generator().manual_seed(int(cfg["seed"]) + tr.step_n)
    while not tr.done:
        enc.train()
        pred.train()
        for _, hr, lr, _ in batches(loader, device, aug):
            pool = (1, 1, 1) if in_cooldown(tr.step_n, tr.total, sec) else main_pool
            k = int(torch.multinomial(weights, 1, generator=g))
            x = L.normalize(L.level_image(hr, lr, k), pool)
            with autocast(cfg, device):
                loss = jepa_loss(enc, pred, target, x, L.grid(pool), cfg["masks"], g)
            tr.step(loss, pool=pool[1])
            ema_update(target, enc, sec["ema"])
        tr.end_epoch(validate(cfg, L, enc, pred, target, vloader, device))
    return out


@torch.no_grad()
def validate(cfg, L, enc, pred, target, loader, device):
    """Native-resolution JEPA loss on the truth (loss) and on the measurement (loss_lr); collapse monitor."""
    enc.eval()
    pred.eval()
    g = torch.Generator().manual_seed(1234)
    acc = {"loss": 0.0, "loss_lr": 0.0, "feat_std": 0.0}
    n = 0
    for _, hr, lr, _ in batches(loader, device):
        with autocast(cfg, device):
            x_hr = L.normalize(hr)
            acc["loss"] += float(jepa_loss(enc, pred, target, x_hr, L.grid(), cfg["masks"], g))
            acc["loss_lr"] += float(jepa_loss(enc, pred, target, L.normalize(L.level_image(hr, lr, 0)), L.grid(),
                                              cfg["masks"], g))
            acc["feat_std"] += float(feature_std(target, x_hr))
        n += 1
    return {k: v / max(n, 1) for k, v in acc.items()}
