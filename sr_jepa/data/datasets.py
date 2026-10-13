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


FP16_MAX = 6.0e4


class Preloaded:
    """A whole split in one tensor per array (GPU memory or RAM), batched by indexing: no per-event reads, no
    collation, no worker processes and, on the GPU, no host-to-device copies. Iterates like the DataLoader,
    (index, hr, lr, target) in float32, with a fresh permutation every epoch."""

    def __init__(self, cfg, split, batch_size, shuffle, max_n, device, dtype, seed=0):
        cache = cfg["paths"]["cache"]
        tgt = np.load(os.path.join(cache, f"{split}_target.npy"))
        n = len(tgt) if max_n is None else min(int(max_n), len(tgt))
        self.target = torch.as_tensor(tgt[:n], dtype=torch.float32, device=device)
        self.hr = _load(os.path.join(cache, f"{split}_hr.npy"), n, device, dtype)
        self.lr = _load(os.path.join(cache, f"{split}_lr.npy"), n, device, dtype)
        self.n, self.bs, self.shuffle, self.device = n, batch_size, shuffle, device
        self.drop_last = shuffle and n >= batch_size
        self.g = torch.Generator().manual_seed(int(cfg["seed"]) + seed)

    def __len__(self):
        return self.n // self.bs if self.drop_last else -(-self.n // self.bs)

    def __iter__(self):
        order = torch.randperm(self.n, generator=self.g) if self.shuffle else torch.arange(self.n)
        order = order.to(self.device)
        for b in range(len(self)):
            i = order[b * self.bs:(b + 1) * self.bs]
            yield i, self.hr[i].float(), self.lr[i].float(), self.target[i]


def _load(path, n, device, dtype, chunk=4096):
    """memmap -> tensor on `device`, chunk by chunk (never two full copies in memory)."""
    m = np.load(path, mmap_mode="r")
    out = torch.empty((n, *m.shape[1:]), dtype=dtype, device=device)
    for s in range(0, n, chunk):
        out[s:s + chunk] = torch.from_numpy(np.array(m[s:min(n, s + chunk)], dtype=np.float32))
    return out


def _fits_fp16(cfg, split, max_n, chunk=4096):
    for what in ("hr", "lr"):
        m = np.load(os.path.join(cfg["paths"]["cache"], f"{split}_{what}.npy"), mmap_mode="r")
        n = len(m) if max_n is None else min(int(max_n), len(m))
        for s in range(0, n, chunk):
            if float(np.max(m[s:min(n, s + chunk)])) > FP16_MAX:
                return False
    return True


def _free_ram():
    """Available RAM in bytes: MemAvailable on Linux (Colab), else half the physical RAM, else 0."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 2
    except (ValueError, OSError, AttributeError):
        return 0


def _placement(cfg, split, max_n, device):
    """-> ('gpu' | 'cpu' | 'none', dtype). data.preload: auto | gpu | cpu | none. auto puts the split on the GPU
    when it takes < 35 % of its memory, else in RAM when it takes < 50 % of the free RAM, else streams it.
    data.preload_dtype float16 halves the footprint (falls back to float32 if a value exceeds the fp16 range)."""
    d = cfg["data"]
    dtype = torch.float32
    if d.get("preload_dtype", "float32") == "float16":
        if _fits_fp16(cfg, split, max_n):
            dtype = torch.float16
        else:
            print(f"  {split}: values above the fp16 range, preloading in float32")
    mode = d.get("preload", "auto")
    if mode != "auto":
        return (mode if mode != "gpu" or device.type == "cuda" else "cpu"), dtype
    size = 0
    for what in ("hr", "lr"):
        m = np.load(os.path.join(cfg["paths"]["cache"], f"{split}_{what}.npy"), mmap_mode="r")
        size += (len(m) if max_n is None else min(int(max_n), len(m))) * m[0].size * (2 if dtype == torch.float16
                                                                                       else 4)
    if device.type == "cuda" and size < 0.35 * torch.cuda.get_device_properties(device).total_memory:
        return "gpu", dtype
    if size < 0.5 * _free_ram():
        return "cpu", dtype
    return "none", dtype


def make_loader(cfg, split, batch_size, shuffle, max_n=None, seed=0, device=None):
    device = torch.device(device or "cpu")
    where, dtype = _placement(cfg, split, max_n, device)
    if where != "none":
        loader = Preloaded(cfg, split, batch_size, shuffle, max_n, device if where == "gpu" else torch.device("cpu"),
                           dtype, seed)
        print(f"  {split}: {loader.n} events preloaded to {where} ({str(dtype).split('.')[-1]})")
        return loader
    ds = EventDataset(cfg, split, max_n)
    g = torch.Generator().manual_seed(int(cfg["seed"]) + seed)
    nw = int(cfg["num_workers"])
    print(f"  {split}: {len(ds)} events streamed from the memmap cache ({nw} workers)")
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

    # ------------------------------------------------------ augmentation
    def augment(self, hr, lr):
        """Symmetry transforms applied identically to truth and measurement, so the pair stays exact (every
        transform maps whole LR cells onto LR cells, so it commutes with the sum-pooling).
        jet    : per-event reflections eta -> -eta and phi -> -phi (the padded 128 grid shifts the jet axis by
                 one pixel, well inside the jet)
        shower : per-event angular reflection and a per-batch rotation about the beam axis by whole LR cells"""
        B, dev = len(hr), hr.device
        if self.kind == "jet":
            for dim in (-2, -1):
                m = (torch.rand(B, device=dev) < 0.5).view(B, 1, 1, 1)
                hr, lr = torch.where(m, hr.flip(dim), hr), torch.where(m, lr.flip(dim), lr)
            return hr, lr
        m = (torch.rand(B, device=dev) < 0.5).view(B, 1, 1, 1)
        hr, lr = torch.where(m, hr.flip(-2), hr), torch.where(m, lr.flip(-2), lr)
        k = int(torch.randint(0, lr.shape[-2], (1,)))
        return hr.roll(k * self._factor(self.levels[0])[0], -2), lr.roll(k, -2)

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
