"""Single-node data parallelism for the fixed-size Masked VAE batch."""
from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import rng_state
from .losses import DYNAMIC_CATEGORIES


class TrainingGroup:
    def __init__(self, device):
        self.device = device
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    @property
    def primary(self):
        return self.rank == 0

    def wrap(self, model):
        device_ids = [self.device.index] if self.device.type == "cuda" else None
        return DistributedDataParallel(model, device_ids=device_ids,
                                       broadcast_buffers=False, gradient_as_bucket_view=True)

    def split_batch(self, effective):
        return effective[self.rank::self.world_size]

    def accumulation_context(self, model, *, synchronize):
        return nullcontext() if synchronize else model.no_sync()

    def backward(self, loss):
        # Each loss already uses the global denominators. DDP averages gradients.
        (loss * self.world_size).backward()

    def _sum(self, values, dtype):
        tensor = torch.tensor(values, dtype=dtype, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.tolist()

    def sum_counts(self, counts):
        fields = [counts[key][category] for key in ("changed", "unchanged")
                  for category in DYNAMIC_CATEGORIES]
        batch_size, static, *totals = self._sum([counts["batch_size"], counts["static"], *fields],
                                               torch.int64)
        size = len(DYNAMIC_CATEGORIES)
        return {"batch_size": batch_size, "static": static,
                "changed": dict(zip(DYNAMIC_CATEGORIES, totals[:size])),
                "unchanged": dict(zip(DYNAMIC_CATEGORIES, totals[size:]))}

    def sum_metrics(self, aggregate):
        fields = []
        for key, value in aggregate.items():
            if isinstance(value, dict):
                fields.extend((value, inner_key) for inner_key in value)
            else:
                fields.append((aggregate, key))
        totals = self._sum([row[key] for row, key in fields], torch.float64)
        for (row, key), total in zip(fields, totals):
            row[key] = type(row[key])(total)

    def gather_rng(self):
        states = [None] * self.world_size
        dist.all_gather_object(states, rng_state(self.device))
        return states

    def broadcast_stop(self, stop):
        flag = torch.tensor(int(stop), dtype=torch.uint8, device=self.device)
        dist.broadcast(flag, src=0)
        return bool(flag.item())


@contextmanager
def cuda_training_group():
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["LOCAL_WORLD_SIZE"]) != world_size:
        raise ValueError("E01 DDP runs on one server with one worker per visible GPU")
    if (not dist.is_available() or not dist.is_nccl_available()
            or not 0 <= local_rank < torch.cuda.device_count()):
        raise RuntimeError("E01 DDP requires a visible CUDA GPU per worker and PyTorch NCCL")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        yield TrainingGroup(device)
    except BaseException:
        # Let torchrun terminate peers if any worker fails in a collective.
        raise
    else:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()
