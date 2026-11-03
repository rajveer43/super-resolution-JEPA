"""Data-parallel training over several GPUs of one node (launched by `torchrun`, see slurm/common.sh).

Every rank holds a replica of the networks and a disjoint shard of the data; the Trainer averages the gradients
after each backward pass (Trainer.step) and the epoch metrics after each epoch (Trainer.end_epoch). Without
torchrun (WORLD_SIZE unset) nothing here does anything, so single-GPU and CPU runs are unchanged.

batch_size in the config is always the GLOBAL batch: each rank trains on batch_size / world events per step, so the
optimisation (and the learning-rate scaling by sqrt(batch / lr_ref_batch)) does not depend on the number of GPUs.
"""
import os

import torch
import torch.distributed as dist


def world():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def active():
    return world() > 1


def is_main():
    return rank() == 0


def init():
    """Join the process group when started by torchrun; each rank then works on its own GPU."""
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1 or dist.is_initialized():
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("nccl")
    else:
        dist.init_process_group("gloo")


def shutdown():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def local_batch(batch_size, w=None):
    """Events per rank of a global batch."""
    w = world() if w is None else w
    if batch_size % w:
        raise ValueError(f"global batch size {batch_size} must be divisible by the number of GPUs ({w})")
    return batch_size // w


def _device():
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


def all_reduce_mean(tensors):
    """Average a list of same-device tensors over the ranks, in place, with one collective."""
    if not active() or not tensors:
        return
    flat = torch.cat([t.reshape(-1) for t in tensors])
    dist.all_reduce(flat)           # SUM: AVG is not available on every backend
    flat /= world()
    o = 0
    for t in tensors:
        t.copy_(flat[o:o + t.numel()].view_as(t))
        o += t.numel()


def mean_dict(d):
    """{name: float} averaged over the ranks (identical on every rank afterwards)."""
    if not active() or not d:
        return d
    keys = list(d)                  # same order on every rank: the stages build these dicts identically
    t =torch.tensor([float(d[k]) for k in keys], dtype=torch.float64, device=_device())
    all_reduce_mean([t])
    return dict(zip(keys, t.tolist()))


def broadcast_state(modules):
    """Make every rank start from rank 0's weights (parameters and buffers)."""
    if not active():
        return
    for m in modules:
        for t in list(m.parameters()) + list(m.buffers()):
            dist.broadcast(t.data, 0)
