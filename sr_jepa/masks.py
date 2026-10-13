"""Multiblock masking on a (gd, gh, gw) token grid (V-JEPA / I-JEPA).

One mask generator places n_blocks boxes per event: each covers `spatial_scale` of the H x W token plane with
aspect ratio in `aspect` (log-uniform), and `depth_scale` of the D axis (1.0 = tubes through all layers, as
V-JEPA masks through time). As in the I-JEPA / V-JEPA collators, block sizes are drawn once per batch and
positions per event. The union of the boxes is the prediction target; the rest is the context. Counts are
equalised over the batch (random subsets), so the batch stays a dense tensor.

Vectorised over the batch: a handful of tensor ops per batch, no Python loop over events.
"""
import math

import torch


def _sizes(grid, spec, g):
    gd, gh, gw = grid
    lo, hi = spec["spatial_scale"]
    alo, ahi = spec["aspect"]
    dlo, dhi = spec.get("depth_scale", [1.0, 1.0])
    out = []
    for _ in range(int(spec["n_blocks"])):
        u = torch.rand(3, generator=g).tolist()
        area = (lo + (hi - lo) * u[0]) * gh * gw
        ar = math.exp(math.log(alo) + (math.log(ahi) - math.log(alo)) * u[1])
        h = max(1, min(gh, round(math.sqrt(area * ar))))
        w = max(1, min(gw, round(math.sqrt(area / ar))))
        d = max(1, min(gd, round((dlo + (dhi - dlo) * u[2]) * gd)))
        out.append((d, h, w))
    return torch.tensor(out)                                     # (n_blocks, 3)


def _place(B, grid, sizes, coords, g):
    """Random positions for every event and block -> masked (B, N) bool."""
    G = torch.tensor(grid)
    start = (torch.rand(B, len(sizes), 3, generator=g) * (G - sizes + 1)).long()   # (B, nb, 3)
    c = coords[None, None]                                       # (1, 1, N, 3)
    inside = ((c >= start[:, :, None]) & (c < (start + sizes)[:, :, None])).all(-1)  # (B, nb, N)
    return inside.any(1)


def sample_masks(B, grid, spec, g, min_ctx_frac=0.1, tries=20):
    """-> ctx_idx (B, Nc), tgt_idx (B, Nt) long tensors (CPU)."""
    N = math.prod(grid)
    min_ctx = max(1, int(min_ctx_frac * N))
    coords = torch.stack(torch.meshgrid(*(torch.arange(n) for n in grid), indexing="ij"), -1).reshape(N, 3)
    sizes = _sizes(grid, spec, g)
    mask = _place(B, grid, sizes, coords, g)
    for _ in range(tries):  # re-place the events whose context is too small
        bad = (~mask).sum(1) < min_ctx
        if not bad.any():
            break
        mask[bad] = _place(int(bad.sum()), grid, sizes, coords, g)
    bad = (~mask).sum(1) < min_ctx
    if bad.any():  # blocks too large for this grid: random 50 % split for those events
        r = torch.rand(int(bad.sum()), N, generator=g)
        mask[bad] = r < r.median(1, keepdim=True).values
    nt = int(mask.sum(1).min())
    nc = int((~mask).sum(1).min())
    # random subset of each set: masked tokens get keys in [0, 1), the others [2, 3); sort and cut
    r = torch.rand(B, N, generator=g)
    tgt = torch.sort(torch.where(mask, r, r + 2)).indices[:, :nt].sort(1).values
    ctx = torch.sort(torch.where(mask, r + 2, r)).indices[:, :nc].sort(1).values
    return ctx, tgt
