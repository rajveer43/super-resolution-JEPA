"""Multiblock masking on a (gd, gh, gw) token grid (V-JEPA / I-JEPA).

One mask generator places n_blocks boxes per sample: each covers `spatial_scale` of the H x W token plane
with aspect ratio in `aspect` (log-uniform), and `depth_scale` of the D axis (1.0 = tubes through all
layers, as V-JEPA masks through time). The union of the boxes is the prediction target; the rest is the
context. Counts are equalised over the batch (random subsets), as in the I-JEPA collator, so the batch
stays a dense tensor.
"""
import math

import torch


def _sample_one(grid, spec, g):
    gd, gh, gw = grid
    m = torch.zeros(grid, dtype=torch.bool)
    lo, hi = spec["spatial_scale"]
    alo, ahi = spec["aspect"]
    dlo, dhi = spec.get("depth_scale", [1.0, 1.0])
    for _ in range(int(spec["n_blocks"])):
        u = torch.rand(3, generator=g).tolist()
        area = (lo + (hi - lo) * u[0]) * gh * gw
        ar = math.exp(math.log(alo) + (math.log(ahi) - math.log(alo)) * u[1])
        h = max(1, min(gh, round(math.sqrt(area * ar))))
        w = max(1, min(gw, round(math.sqrt(area / ar))))
        d = max(1, min(gd, round((dlo + (dhi - dlo) * u[2]) * gd)))
        d0 = int(torch.randint(0, gd - d + 1, (1,), generator=g))
        h0 = int(torch.randint(0, gh - h + 1, (1,), generator=g))
        w0 = int(torch.randint(0, gw - w + 1, (1,), generator=g))
        m[d0:d0 + d, h0:h0 + h, w0:w0 + w] = True
    return m.flatten()


def sample_masks(B, grid, spec, g, min_ctx_frac=0.1, tries=20):
    """-> ctx_idx (B, Nc), tgt_idx (B, Nt) long tensors (CPU)."""
    N = math.prod(grid)
    min_ctx = max(1, int(min_ctx_frac * N))
    masks = []
    for _ in range(B):
        for _ in range(tries):
            m = _sample_one(grid, spec, g)
            if (~m).sum() >= min_ctx and m.any():
                break
        else:  # fall back to a random 50 % split
            m = torch.zeros(N, dtype=torch.bool)
            m[torch.randperm(N, generator=g)[: N // 2]] = True
        masks.append(m)
    nt = min(int(m.sum()) for m in masks)
    nc = min(int((~m).sum()) for m in masks)
    ctx, tgt = [], []
    for m in masks:
        t, c = torch.nonzero(m).flatten(), torch.nonzero(~m).flatten()
        tgt.append(t[torch.randperm(len(t), generator=g)[:nt]].sort().values)
        ctx.append(c[torch.randperm(len(c), generator=g)[:nc]].sort().values)
    return torch.stack(ctx), torch.stack(tgt)
