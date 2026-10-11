"""Vision transformers of the SR-JEPA pipeline, all with 3D-RoPE over the (D, H, W) token grid.

Encoder            E(x): tubelet embedding + ViT; called with `keep` indices it only sees the context tokens.
MaskPredictor      stage 1 (V-JEPA mask denoising): context tokens + learnable mask tokens at the target
                   positions -> predicted target-encoder features of the masked tokens.
LatentSRPredictor  stage 2 (V-JEPA 2-AC analogue): the ladder LR -> ... -> HR plays the role of time. Each
                   level contributes a condition token (level, optionally E_inc, like the AC action/state
                   tokens) and its latent tokens; block-causal attention lets level k see levels <= k, and its
                   outputs predict the latent of level k+1.
Decoder            stage 3: latent tokens (+ the LR measurement) -> normalised HR energies (unpatchify).
AttentiveProbe     frozen-feature evaluation: cross-attention pooling with one learnable query.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import Rope3D, apply_rope, grid_positions


def gather_tokens(x, idx):
    """x (B, N, ...), idx (B, M) -> (B, M, ...)."""
    shape = idx.shape + x.shape[2:]
    return torch.gather(x, 1, idx.reshape(*idx.shape, *([1] * (x.dim() - 2))).expand(shape))


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv3d):
        nn.init.trunc_normal_(m.weight.view(m.weight.shape[0], -1), std=0.02)
        nn.init.zeros_(m.bias)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cs=None, mask=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        if cs is not None:
            q, k = apply_rope(q, cs), apply_rope(k, cs)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, x, cs=None, mask=None):
        x = x + self.attn(self.n1(x), cs, mask)
        return x + self.mlp(self.n2(x))


class _Trunk(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio, periodic, rope_base):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.rope = Rope3D(dim // heads, rope_base, periodic)

    def run(self, x, pos, grid, mask=None):
        cs = self.rope(pos, grid)
        for b in self.blocks:
            x = b(x, cs, mask)
        return self.norm(x)


class Encoder(_Trunk):
    def __init__(self, in_ch, patch, dim, depth, heads, mlp_ratio=4.0, periodic=(False,) * 3, rope_base=1e4):
        super().__init__(dim, depth, heads, mlp_ratio, periodic, rope_base)
        self.dim = dim
        self.patch_embed = nn.Conv3d(in_ch, dim, patch, patch)
        self.apply(_init)

    def forward(self, x, keep=None):
        """x (B, C, D, H, W) -> tokens (B, N, dim); keep (B, M): encode only those tokens."""
        t = self.patch_embed(x)
        grid = tuple(t.shape[2:])
        t = t.flatten(2).transpose(1, 2)
        pos = grid_positions(grid, x.device)[None].expand(len(x), -1, -1)
        if keep is not None:
            t, pos = gather_tokens(t, keep), gather_tokens(pos, keep)
        return self.run(t, pos, grid)


class MaskPredictor(_Trunk):
    def __init__(self, enc_dim, dim, depth, heads, mlp_ratio=4.0, periodic=(False,) * 3, rope_base=1e4):
        super().__init__(dim, depth, heads, mlp_ratio, periodic, rope_base)
        self.embed = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.proj = nn.Linear(dim, enc_dim)
        self.apply(_init)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, ctx, ctx_idx, tgt_idx, grid):
        B, Nt = tgt_idx.shape
        x = torch.cat([self.embed(ctx), self.mask_token.expand(B, Nt, -1).to(ctx.dtype)], 1)
        allpos = grid_positions(grid, ctx.device)
        pos = torch.cat([allpos[ctx_idx], allpos[tgt_idx]], 1)
        return self.proj(self.run(x, pos, grid)[:, -Nt:])


class LatentSRPredictor(_Trunk):
    def __init__(self, enc_dim, dim, depth, heads, mlp_ratio=4.0, n_levels=3, cond_dim=0, residual=True,
                 periodic=(False,) * 3, rope_base=1e4):
        super().__init__(dim, depth, heads, mlp_ratio, periodic, rope_base)
        self.residual = residual
        self.embed = nn.Linear(enc_dim, dim)
        self.level_emb = nn.Parameter(torch.zeros(n_levels, dim))
        self.cond_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, dim), nn.GELU(), nn.Linear(dim, dim)) if cond_dim else None
        self.proj = nn.Linear(dim, enc_dim)
        self.apply(_init)
        nn.init.trunc_normal_(self.level_emb, std=0.02)
        nn.init.trunc_normal_(self.cond_token, std=0.02)
        if residual:  # start as the identity map z_{k+1} = z_k
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, zs, cond, grid):
        """zs: latents of levels 0..k-1, each (B, N, E) -> predictions of the latents of levels 1..k."""
        B, N, _ = zs[0].shape
        k = len(zs)
        c = self.cond_token.expand(B, 1, -1)
        if self.cond_mlp is not None:
            c = c + self.cond_mlp(cond)[:, None]
        toks = []
        for j, z in enumerate(zs):
            toks += [c + self.level_emb[j], self.embed(z) + self.level_emb[j]]
        x = torch.cat(toks, 1)
        # condition tokens sit at position 0 (no rotation); latent tokens at their grid position
        block_pos = torch.cat([torch.zeros(1, 3, dtype=torch.long, device=x.device),
                               grid_positions(grid, x.device)])
        blk = torch.arange(k, device=x.device).repeat_interleave(N + 1)
        mask = blk[:, None] >= blk[None, :]          # block-causal: level j attends to levels <= j
        out = self.proj(self.run(x, block_pos.repeat(k, 1), grid, mask).reshape(B, k, N + 1, -1)[:, :, 1:])
        return [out[:, j] + zs[j] if self.residual else out[:, j] for j in range(k)]

    def rollout(self, z0, cond, grid, steps, grad_steps=None):
        """Autoregressive prediction from the measurement latent alone: [z_1_hat, ..., z_steps_hat].
        grad_steps: only the last `grad_steps` steps are differentiated (V-JEPA 2-AC: one)."""
        seq = [z0]
        for s in range(steps):
            track = grad_steps is None or s >= steps - grad_steps
            with torch.set_grad_enabled(track and torch.is_grad_enabled()):
                seq.append(self(seq, cond, grid)[-1])
        return seq[1:]


class Decoder(_Trunk):
    def __init__(self, enc_dim, dim, depth, heads, patch, out_ch, mlp_ratio=4.0, lr_skip=True,
                 periodic=(False,) * 3, rope_base=1e4):
        super().__init__(dim, depth, heads, mlp_ratio, periodic, rope_base)
        self.patch, self.out_ch = tuple(patch), out_ch
        self.embed = nn.Linear(enc_dim, dim)
        self.lr_embed = nn.Conv3d(out_ch, dim, patch, patch) if lr_skip else None
        self.head = nn.Linear(dim, out_ch * math.prod(patch))
        self.apply(_init)

    def forward(self, z, grid, lr=None):
        """z (B, N, E), lr (B, C, D, H, W) normalised measurement on the native grid -> (B, C, D, H, W)."""
        t = self.embed(z)
        if self.lr_embed is not None:
            t = t + self.lr_embed(lr).flatten(2).transpose(1, 2)
        y = self.head(self.run(t, grid_positions(grid, z.device), grid))
        B = len(y)
        (gd, gh, gw), (pd, ph, pw) = grid, self.patch
        y = y.reshape(B, gd, gh, gw, self.out_ch, pd, ph, pw).permute(0, 4, 1, 5, 2, 6, 3, 7)
        return y.reshape(B, self.out_ch, gd * pd, gh * ph, gw * pw)


class AttentiveProbe(nn.Module):
    def __init__(self, enc_dim, n_out, heads=4, depth=1, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList(Block(enc_dim, heads, mlp_ratio) for _ in range(depth))
        self.query = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.cross = nn.MultiheadAttention(enc_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(enc_dim)
        self.head = nn.Sequential(nn.Linear(enc_dim, enc_dim), nn.GELU(), nn.Linear(enc_dim, n_out))
        self.apply(_init)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, z):
        for b in self.blocks:
            z = b(z)
        o, _ = self.cross(self.query.expand(len(z), -1, -1), z, z, need_weights=False)
        return self.head(self.norm(o[:, 0]))
