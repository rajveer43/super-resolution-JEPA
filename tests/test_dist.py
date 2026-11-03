"""Data-parallel training (sr_jepa/dist.py): sharding and gradient averaging, on CPU with the gloo backend."""
import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sr_jepa import dist as sdist
from sr_jepa.config import load_config
from sr_jepa.data.datasets import make_loader, shard
from sr_jepa.train.common import Trainer


def test_shard_gives_every_rank_the_same_count_and_no_overlap():
    seen = []
    for rank in range(4):
        rows, n = shard(37, rank, 4)
        idx = list(range(37))[rows]
        assert len(idx) == n == 9
        seen += idx
    assert len(set(seen)) == len(seen) == 36            # the remainder (1 event) is dropped


def test_sharded_loaders_partition_the_split(tmp_path):
    from numpy.lib.format import open_memmap
    n = 41
    for what, shape in (("hr", (n, 3, 8, 8)), ("lr", (n, 3, 2, 2))):
        m = open_memmap(tmp_path / f"train_{what}.npy", "w+", np.float32, shape)
        m[:] = np.arange(n, dtype=np.float32).reshape(n, 1, 1, 1)
        m.flush()
    np.save(tmp_path / "train_target.npy", np.arange(n, dtype=np.float64))
    cfg = load_config("configs/smoke_qg.yaml", [f"paths.cache={tmp_path}", "data.preload=cpu"])
    seen = []
    for rank in range(2):
        loader = make_loader(cfg, "train", 8, False, rank=rank, world=2)       # global 8 -> 4 per rank
        assert len(loader) == 5 and loader.bs == 4
        for _, hr, _, t in loader:
            assert torch.equal(hr[:, 0, 0, 0], t)
            seen += t.tolist()
    assert len(seen) == len(set(seen)) == 40


def test_global_batch_must_divide_by_the_number_of_gpus():
    with pytest.raises(ValueError, match="divisible"):
        sdist.local_batch(10, 4)


def _worker(rank, world, init_file, out_dir, q):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        cfg = load_config("configs/smoke_qg.yaml")
        sec = dict(cfg["sr"], lr_ref_batch=None, batch_size=4, epochs=1, grad_clip=None, weight_decay=0.0,
                   start_lr_frac=1.0, warmup_frac=0.0, cooldown_frac=0.0, final_lr_frac=1.0)
        torch.manual_seed(100 + rank)                          # different initial weights on every rank
        net = torch.nn.Linear(3, 1)
        tr = Trainer(cfg, sec, out_dir, {"net": net}, [(net, 1.0)], steps_per_epoch=1)
        x = torch.arange(12, dtype=torch.float).view(4, 3) + 10 * rank       # different data on every rank
        loss = net(x).pow(2).mean()
        tr.step(loss)
        tr.end_epoch({"loss": float(rank)})                    # mean over the ranks: 0.5
        q.put((rank, [p.detach().clone() for p in net.parameters()], tr.best))
    finally:
        dist.destroy_process_group()


def test_trainer_averages_gradients_and_metrics_over_ranks(tmp_path):
    world = 2
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, str(tmp_path / "store"), str(tmp_path / f"out{r}"), q))
             for r in range(world)]
    for r in range(world):
        os.makedirs(tmp_path / f"out{r}")
    for p in procs:
        p.start()
    res = {}
    for _ in range(world):
        rank, params, best = q.get(timeout=120)
        res[rank] = (params, best)
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0
    for a, b in zip(res[0][0], res[1][0]):
        assert torch.equal(a, b)                                # replicas stay identical
    assert res[0][1] == res[1][1] == pytest.approx(0.5)         # validation loss averaged over the ranks

    # reference: one process, the two shards as a single batch (mean loss over equal-sized shards)
    cfg = load_config("configs/smoke_qg.yaml")
    sec = dict(cfg["sr"], lr_ref_batch=None, batch_size=8, epochs=1, grad_clip=None, weight_decay=0.0,
               start_lr_frac=1.0, warmup_frac=0.0, cooldown_frac=0.0, final_lr_frac=1.0)
    torch.manual_seed(100)                                      # rank 0's initial weights are the ones broadcast
    net = torch.nn.Linear(3, 1)
    os.makedirs(tmp_path / "ref")
    tr = Trainer(cfg, sec, str(tmp_path / "ref"), {"net": net}, [(net, 1.0)], steps_per_epoch=1)
    x = torch.cat([torch.arange(12, dtype=torch.float).view(4, 3) + 10 * r for r in range(world)])
    tr.step(net(x).pow(2).mean())
    for a, b in zip(res[0][0], net.parameters()):
        assert torch.allclose(a, b.detach(), atol=1e-6)
