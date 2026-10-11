"""Cached events -> model tensors.

Cache (written by `prepare`, same format as superresolution-difflense):
    {split}_hr.npy (N, P, U, V) truth, {split}_lr.npy (N, P, u, v) noisy measurement, {split}_target.npy (N,)

Layout maps the detector's (P, U, V) planes onto the model's (C, D, H, W) volume:
    jet    : C = P (tracks / ECAL / HCAL as channels), D = 1         -> an image
    shower : C = 1, D = P (45 layers as the "time" axis of V-JEPA)  -> a video
and every ladder level (LR measurement, intermediate grids, native truth) onto the native grid by
uniform (energy-preserving) upsampling, so all levels share one token grid and their latents align.
Values seen by the networks are log energy densities: x = log1p(E / (scale_p * cell_area)).
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .ops import sum_pool, uniform_upsample

SPLITS = ("train", "val", "test")


def load_meta(cfg):
    path = os.path.join(cfg["paths"]["cache"], "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run the `prepare` stage first")
    with open(path) as f:
        meta = json.load(f)
    if list(meta["lr_factor"]) != list(cfg["ladder"]["lr_factor"]):
        raise ValueError(f"cache lr_factor {meta['lr_factor']} != ladder.lr_factor {cfg['ladder']['lr_factor']}")
    return meta


class EventDataset(Dataset):
    """(index, hr (P,U,V), lr (P,u,v), target) from the memmapped cache; memmaps open lazily per worker."""

    def __init__(self, cfg, split, max_n=None):
        self.cache, self.split = cfg["paths"]["cache"], split
        self.target = np.load(os.path.join(self.cache, f"{split}_target.npy"))
        self.n = len(self.target) if max_n is None else min(int(max_n), len(self.target))
        self._hr = self._lr = None

    def __len__(self):
        return self.n

    def _open(self):
        if self._hr is None:
            self._hr = np.load(os.path.join(self.cache, f"{self.split}_hr.npy"), mmap_mode="r")
            self._lr = np.load(os.path.join(self.cache, f"{self.split}_lr.npy"), mmap_mode="r")

    def __getitem__(self, i):
        self._open()
        return (i, torch.from_numpy(np.array(self._hr[i], dtype=np.float32)),
                torch.from_numpy(np.array(self._lr[i], dtype=np.float32)),
                torch.tensor(self.target[i], dtype=torch.float32))


def make_loader(cfg, split, batch_size, shuffle, max_n=None, seed=0):
    ds = EventDataset(cfg, split, max_n)
    g = torch.Generator().manual_seed(int(cfg["seed"]) + seed)
    nw = int(cfg["num_workers"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle and len(ds) >= batch_size,
                      num_workers=nw, persistent_workers=nw > 0, generator=g, pin_memory=torch.cuda.is_available())


class Layout:
    def __init__(self, cfg, meta):
        self.kind = cfg["data"]["kind"]
        self.P = len(meta["planes"])
        self.native = tuple(meta["native"])
        self.scales = torch.tensor(meta["scales"], dtype=torch.float32).clamp_min(1e-12)
        self.hit_threshold = float(meta["hit_threshold"])
        self.circular_u = bool(cfg["data"].get("circular_u", False))
        self.patch = tuple(int(p) for p in cfg["model"]["patch"])
        # ladder levels LR -> native, every one with truth
        g = tuple(meta["lr_shape"])
        self.levels = [g]
        for f in cfg["ladder"]["stages"]:
            g = (g[0] * f[0], g[1] * f[1])
            self.levels.append(g)
        if self.levels[-1] != self.native:
            raise ValueError(f"ladder {self.levels} must end at the native grid {self.native}")
        if self.kind == "jet":
            self.C, self.D = self.P, 1
        else:
            self.C, self.D = 1, self.P
        self.H, self.W = self.native
        for n, p, name in zip((self.D, self.H, self.W), self.patch, "DHW"):
            if n % p:
                raise ValueError(f"model.patch {self.patch} does not divide axis {name}={n}")
        self.cond_dim = 1 if self.kind == "shower" else 0

    # ------------------------------------------------------------ levels
    def level_image(self, hr, lr, k):
        """Ladder level k on the native grid (N, P, U, V): 0 = the noisy measurement, last = truth."""
        if k == 0:
            return uniform_upsample(lr, self._factor(self.levels[0]))
        if k == len(self.levels) - 1:
            return hr
        f = self._factor(self.levels[k])
        return uniform_upsample(sum_pool(hr, f), f)

    def _factor(self, g):
        return (self.native[0] // g[0], self.native[1] // g[1])

    # ----------------------------------------------------- normalisation
    def to_model(self, x):
        """(N, P, U, V) -> (N, C, D, H, W)."""
        return x[:, :, None] if self.kind == "jet" else x[:, None]

    def from_model(self, x):
        return x[:, :, 0] if self.kind == "jet" else x[:, 0]

    def normalize(self, E, pool=(1, 1, 1)):
        """Energies (N, P, U, V) on the native grid -> network input (N, C, D', H', W'), optionally sum-pooled
        by `pool` along (D, H, W) (progressive-resolution pretraining). Values: log energy density."""
        s = self.scales.to(E.device).reshape(1, -1, 1, 1)
        x = self.to_model(E.clamp_min(0) / s)
        pd, ph, pw = pool
        if (pd, ph, pw) != (1, 1, 1):
            N, C, D, H, W = x.shape
            x = x.reshape(N, C, D // pd, pd, H // ph, ph, W // pw, pw).mean(dim=(3, 5, 7))
        return torch.log1p(x)

    def denormalize(self, x):
        """Network output on the native grid (N, C, D, H, W) -> energies (N, P, U, V) >= 0."""
        s = self.scales.to(x.device).reshape(1, -1, 1, 1)
        return torch.expm1(self.from_model(x).float().clamp(0, 30)) * s

    def cond(self, target):
        """Per-event condition of the SR world model: log E_inc for showers (known in CaloChallenge), none for jets."""
        if self.kind != "shower":
            return None
        return ((torch.log10(target.clamp_min(1.0)) - 3.0) / 1.5 - 1.0).reshape(-1, 1)

    def clean(self, E):
        """Zero-suppress SR output below the readout threshold (per native cell)."""
        return torch.where(E > self.hit_threshold, E, torch.zeros_like(E))

    def project_to_lr(self, E, lr):
        """Rescale every LR cell's block of SR cells so it sums to the measured energy (energy closure)."""
        f = self._factor(self.levels[0])
        pooled = sum_pool(E, f)
        ratio = torch.where(pooled > 0, lr / pooled.clamp_min(1e-12), torch.zeros_like(lr))
        out = E * uniform_upsample(ratio, f) * (f[0] * f[1])
        fill = uniform_upsample(torch.where(pooled > 0, torch.zeros_like(lr), lr), f)  # measured but empty in SR
        return out + fill

    def grid(self, pool=(1, 1, 1)):
        return tuple(n // (p * q) for n, p, q in zip((self.D, self.H, self.W), self.patch, pool))
