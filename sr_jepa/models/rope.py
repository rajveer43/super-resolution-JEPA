"""3D rotary position embedding (V-JEPA 2 section 2.1).

The head dimension is split into three equal segments for the (D, H, W) token axes; each segment gets the
1D RoPE rotation of its axis, leftover dimensions are not rotated. Positions are token indices, so the same
model runs on any token grid (progressive resolution).

A periodic axis (the CaloChallenge angle) uses frequencies 2*pi*k / n, k = 1..m: rotations then depend only on
the position modulo the n tokens of that axis, so the first and last angular bins are neighbours.
"""
import math

import torch


class Rope3D:
    def __init__(self, head_dim, base=10000.0, periodic=(False, False, False)):
        self.m = (head_dim // 3) // 2          # rotated pairs per axis
        if self.m < 1:
            raise ValueError(f"head_dim {head_dim} too small for 3D RoPE")
        self.base = base
        self.periodic = tuple(periodic)

    def freqs(self, axis, n, device):
        k = torch.arange(self.m, device=device, dtype=torch.float32)
        if self.periodic[axis]:
            return 2 * math.pi * (k + 1) / n
        return self.base ** (-k / self.m)

    def __call__(self, pos, grid):
        """pos (B, N, 3) or (N, 3) token indices, grid (gd, gh, gw) -> (cos, sin), each (B|1, 1, N, 3, m)."""
        if pos.dim() == 2:
            pos = pos[None]
        ang = torch.stack([pos[..., a:a + 1].float() * self.freqs(a, grid[a], pos.device) for a in range(3)], -2)
        return ang.cos()[:, None], ang.sin()[:, None]


def apply_rope(x, cs):
    """x (B, H, N, hd), cs from Rope3D -> x with the first 3*2m dims rotated (rotate-half per axis segment)."""
    cos, sin = (c.to(x.dtype) for c in cs)
    m = cos.shape[-1]
    B, H, N, hd = x.shape
    r = x[..., : 6 * m].reshape(B, H, N, 3, 2 * m)
    x1, x2 = r[..., :m], r[..., m:]
    r = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).reshape(B, H, N, 6 * m)
    return torch.cat([r, x[..., 6 * m:]], -1)


def grid_positions(grid, device=None):
    """(N, 3) token indices of a (gd, gh, gw) grid in row-major order (matches Conv3d patch flattening)."""
    gd, gh, gw = grid
    d, h, w = torch.meshgrid(torch.arange(gd, device=device), torch.arange(gh, device=device),
                             torch.arange(gw, device=device), indexing="ij")
    return torch.stack([d, h, w], -1).reshape(-1, 3)
