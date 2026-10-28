"""Frozen-feature probe (V-JEPA evaluation protocol): an attentive probe on the frozen encoder's latents of the
truth. jet: quark/gluon classification (BCE); shower: regression of the normalised log E_inc (L1).

The evaluate stage applies this one probe to latents of the truth, the measurement and the SR prediction:
if latent SR works, the task performance from the SR latent moves from the LR value towards the HR value.
"""
import torch
import torch.nn.functional as F

from ..config import tuned_report
from ..data.datasets import make_loader
from ..models.build import build_probe
from ..pipeline import frozen_encoder, latents
from .common import Trainer, autocast, batches, setup, stage_dir, train_augment


def probe_target(L, t):
    return t.float() if L.kind == "jet" else L.cond(t)[:, 0]


def probe_loss(L, out, t):
    y = probe_target(L, t)
    if L.kind == "jet":
        return F.binary_cross_entropy_with_logits(out[:, 0].float(), y)
    return (out[:, 0].float() - y).abs().mean()


def run(cfg, on_epoch=None):
    tuned_report(cfg, "probe")
    device, meta, L = setup(cfg)
    sec = cfg["probe"]
    enc = frozen_encoder(cfg, L, device)
    probe = build_probe(cfg, 1).to(device)
    loader = make_loader(cfg, "train", sec["batch_size"], True, sec["max_train"], device=device)
    aug = train_augment(cfg, L)
    vloader = make_loader(cfg, "val", sec["batch_size"], False, sec["max_val"], device=device)
    out = stage_dir(cfg, "probe")
    tr = Trainer(cfg, sec, out, {"probe": probe}, [(probe, 1.0)], len(loader), on_epoch)
    while not tr.done:
        probe.train()
        for _, hr, _, t in batches(loader, device, aug):
            with torch.no_grad():
                z = latents(enc, L.normalize(hr))
            with autocast(cfg, device):
                loss = probe_loss(L, probe(z), t)
            tr.step(loss)
        probe.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for _, hr, _, t in batches(vloader, device):
                with autocast(cfg, device):
                    tot += float(probe_loss(L, probe(latents(enc, L.normalize(hr))), t))
                n += 1
        tr.end_epoch({"loss": tot / max(n, 1)})
    return out
