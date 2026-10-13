"""Invariants of the SR-JEPA building blocks (CPU, seconds)."""
import math

import pytest
import torch

from sr_jepa.config import load_config
from sr_jepa.data.datasets import Layout
from sr_jepa.data.ops import sum_pool
from sr_jepa.masks import sample_masks
from sr_jepa.models.rope import Rope3D, apply_rope
from sr_jepa.models.vit import Decoder, Encoder, LatentSRPredictor
from sr_jepa.train.common import wcd_factor


def _layout(name):
    cfg = load_config(f"configs/{name}.yaml")
    d = cfg["data"]
    P = d["planes"] if isinstance(d["planes"], int) else len(d["planes"])
    U, V = d["hr_shape"]
    f = cfg["ladder"]["lr_factor"]
    meta = {"planes": list(range(P)), "native": [U, V], "scales": [0.5] * P, "hit_threshold": d["hit_threshold"],
            "lr_shape": [U // f[0], V // f[1]], "lr_factor": f}
    return cfg, Layout(cfg, meta)


@pytest.mark.parametrize("name", ["smoke_qg", "smoke_calo", "qg", "calo"])
def test_layout_roundtrip_and_levels(name):
    cfg, L = _layout(name)
    E = torch.rand(2, L.P, *L.native) * (torch.rand(2, L.P, *L.native) > 0.6)
    x = L.normalize(E)
    assert x.shape == (2, L.C, L.D, L.H, L.W)
    assert torch.allclose(L.denormalize(x), E, rtol=1e-4, atol=1e-6)
    lr = sum_pool(E, L._factor(L.levels[0]))
    for k in range(len(L.levels)):  # every level conserves the event energy
        assert torch.allclose(L.level_image(E, lr, k).sum(), E.sum(), rtol=1e-4)


@pytest.mark.parametrize("name", ["qg", "calo"])
def test_augment_commutes_with_the_measurement(name):
    """Augmenting (HR, LR) must give the same pair as measuring the augmented HR, or training pairs break."""
    torch.manual_seed(0)
    _, L = _layout(name)
    f = L._factor(L.levels[0])
    hr = torch.rand(8, L.P, *L.native)
    for _ in range(5):
        a_hr, a_lr = L.augment(hr, sum_pool(hr, f))
        assert torch.allclose(sum_pool(a_hr, f), a_lr, rtol=1e-5, atol=1e-5)
        assert torch.allclose(a_hr.sum((1, 2, 3)), hr.sum((1, 2, 3)), rtol=1e-5)


def test_preloaded_loader_visits_every_event_once(tmp_path):
    from numpy.lib.format import open_memmap
    from sr_jepa.data.datasets import make_loader
    import numpy as np
    n = 37
    for what, shape in (("hr", (n, 3, 8, 8)), ("lr", (n, 3, 2, 2))):
        m = open_memmap(tmp_path / f"train_{what}.npy", "w+", np.float32, shape)
        m[:] = np.arange(n, dtype=np.float32).reshape(n, 1, 1, 1)
        m.flush()
    np.save(tmp_path / "train_target.npy", np.arange(n, dtype=np.float64))
    cfg = load_config("configs/smoke_qg.yaml", [f"paths.cache={tmp_path}", "data.preload=cpu",
                                                "data.preload_dtype=float16"])
    for shuffle, expect in ((True, 36), (False, 37)):  # shuffled batches drop the last partial batch
        seen = []
        for i, hr, lr, t in make_loader(cfg, "train", 4, shuffle):
            assert hr.dtype == torch.float32 and torch.equal(hr[:, 0, 0, 0], t) and torch.equal(i.float(), t)
            seen += t.tolist()
        assert len(seen) == expect and len(set(seen)) == expect


def test_project_to_lr_closes_energy():
    _, L = _layout("smoke_qg")
    E = torch.rand(2, L.P, *L.native)
    lr = sum_pool(torch.rand(2, L.P, *L.native), L._factor(L.levels[0]))
    lr[0, 0, 0, 0] = 0.0
    out = L.project_to_lr(E, lr)
    assert torch.allclose(sum_pool(out, L._factor(L.levels[0])), lr, rtol=1e-4, atol=1e-6)


def test_rope_is_relative_and_periodic():
    torch.manual_seed(0)
    rope = Rope3D(48, periodic=(False, True, False))
    q, k = torch.randn(1, 1, 1, 48), torch.randn(1, 1, 1, 48)

    def score(pq, pk, grid=(4, 8, 3)):
        a = apply_rope(q, rope(torch.tensor([pq], dtype=torch.float), grid))
        b = apply_rope(k, rope(torch.tensor([pk], dtype=torch.float), grid))
        return float((a * b).sum())

    assert math.isclose(score([0, 1, 0], [1, 3, 2]), score([2, 2, 1], [3, 4, 3]), rel_tol=1e-4)  # shift-invariant
    assert math.isclose(score([0, 0, 0], [0, 1, 0]), score([0, 0, 0], [0, 9, 0]), rel_tol=1e-4)  # period 8


@pytest.mark.parametrize("grid", [(1, 8, 8), (9, 8, 3), (3, 4, 3)])
def test_masks_disjoint_and_dense(grid):
    g = torch.Generator().manual_seed(0)
    for spec in load_config("configs/calo.yaml")["masks"]:
        ctx, tgt = sample_masks(5, grid, spec, g)
        assert ctx.shape[0] == tgt.shape[0] == 5 and ctx.shape[1] > 0 and tgt.shape[1] > 0
        for c, t in zip(ctx, tgt):
            assert not set(c.tolist()) & set(t.tolist())
            assert max(c.max(), t.max()) < math.prod(grid)


def test_sr_predictor_is_block_causal():
    torch.manual_seed(0)
    grid = (1, 4, 4)
    P = LatentSRPredictor(32, 32, 2, 2, n_levels=3, cond_dim=1, residual=False).eval()
    torch.nn.init.normal_(P.proj.weight, std=0.1)
    zs = [torch.randn(2, 16, 32) for _ in range(2)]
    c = torch.randn(2, 1)
    a = P(zs, c, grid)
    b = P([zs[0], torch.randn(2, 16, 32)], c, grid)
    assert torch.allclose(a[0], b[0], atol=1e-5)          # level 0's prediction ignores level 1
    assert not torch.allclose(a[1], b[1], atol=1e-3)


def test_decoder_unpatchify_matches_encoder_token_order():
    """Patch-average with the encoder's Conv3d, then unpatchify with the decoder: a patch-constant volume
    comes back unchanged only if both use the same token order."""
    patch, grid = (2, 4, 3), (3, 2, 2)
    enc = Encoder(1, patch, 24, 1, 2)
    dec = Decoder(24, 24, 0, 2, patch, 1, lr_skip=False)
    vals = torch.randn(1, 1, *grid)
    x = vals.repeat_interleave(2, 2).repeat_interleave(4, 3).repeat_interleave(3, 4)
    with torch.no_grad():
        enc.patch_embed.weight.fill_(1.0 / math.prod(patch))
        enc.patch_embed.bias.zero_()
        tokens = enc.patch_embed(x).flatten(2).transpose(1, 2)       # (1, N, 24), token value in every channel
        dec.embed = torch.nn.Identity()
        dec.run = lambda t, pos, g: t
        dec.head.weight.zero_()
        dec.head.weight[:, 0] = 1.0
        dec.head.bias.zero_()
        y = dec(tokens, grid)
    assert torch.allclose(y, x, atol=1e-5)


def test_wcd_schedule():
    sec = {"warmup_frac": 0.1, "cooldown_frac": 0.2, "start_lr_frac": 0.1, "final_lr_frac": 0.0}
    assert wcd_factor(0, 100, sec) == pytest.approx(0.1)
    assert wcd_factor(50, 100, sec) == 1.0
    assert wcd_factor(100, 100, sec) == pytest.approx(0.0)
