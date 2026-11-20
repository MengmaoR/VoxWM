import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from voxel_vae.checkpoint import restore_training_rng, rng_state
from voxel_vae.distributed_training import TrainingGroup
from voxel_vae.losses import DYNAMIC_CATEGORIES
from voxel_vae.tests.test_masked_loss import CONFIG
from voxel_vae.training_curves import read_metric_rows
from voxel_vae.vocabulary import CanonicalVocabulary


class _FixtureModel(nn.Module):
    latent_channels = 1

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 2)
        self.encoder_input = nn.Linear(2, 2)
        self.moments_output = nn.Linear(2, 2)
        self.decoder_hidden = nn.Linear(2, 2)
        self.decoder_logits = nn.Linear(2, 3)

    def architecture_config(self):
        return {"name": "synthetic-ddp-fixture"}

    def forward(self, tokens):
        hidden = self.encoder_input(self.embedding(tokens)).tanh()
        mu, logvar = self.moments_output(hidden.mean(dim=(1, 2, 3))).chunk(2, dim=1)
        decoded = self.decoder_hidden(hidden) + mu[:, None, None, None]
        logits = self.decoder_logits(decoded.tanh()).movedim(-1, 1)
        return {"logits": logits, "mu": mu, "logvar": logvar}


class _FixtureDataset:
    def __init__(self, *_):
        pass

    def read_numpy(self, item):
        values = np.full((16, 16, 16), item % 3, dtype=np.int64)
        values[0, 0, 0] = (item + 1) % 3
        return values, item


class _FixtureMasks(dict):
    def statistics(self, _):
        return {}


def _fit_fixture(output, group=None, resume=None):
    # This fixture has no disk dataset, so it also runs in the local Torch-only environment.
    stub_zarr = importlib.util.find_spec("zarr") is None
    if stub_zarr:
        sys.modules["zarr"] = ModuleType("zarr")
    try:
        from voxel_vae import train
    finally:
        if stub_zarr:
            del sys.modules["zarr"]

    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = torch.device("cpu")
    vocab = CanonicalVocabulary((("air", 0), ("mcl_core:dirt", 0), ("mcl_core:stone", 0)))
    categories = ["Static"] * 32 + [key for key in DYNAMIC_CATEGORIES for _ in range(8)]
    index = SimpleNamespace(records=[SimpleNamespace(category=key, split="train") for key in categories],
                            roots=["synthetic"], manifest_hashes=["synthetic"])
    masks = _FixtureMasks()
    for item, category in enumerate(categories[32:], start=32):
        # Only one D1 row has any unchanged voxels, so one rank's local denominator is zero.
        mask = np.full((16, 16, 16), category == "D1" and item != 32, dtype=bool)
        mask[0, 0, 0] = True
        masks[item] = mask
    config = {"seed": 42, "device": str(device), "learning_rate": 1e-4,
              "max_steps": 4, "validate_every": 1, "loss": CONFIG,
              "early_stopping": {"min_steps": 0, "patience_validations": 1}}

    def fixed_validation(model, *_args, **_kwargs):
        if group is not None:
            assert group.primary
        assert isinstance(model, _FixtureModel)
        return {"loss_components": {"loss_total": 1.0}, "latent": {}}

    with (patch.object(train, "VoxelSnapshotDataset", _FixtureDataset),
          patch.object(train, "evaluate_indices", fixed_validation)):
        return train.fit_masked(
            _FixtureModel(), index, vocab, list(range(64)), [0], [1], masks, output,
            config, phase="E01-Masked", group=group, resume=resume)


def _rank_worker(rank, root):
    root = Path(root)
    dist.init_process_group("gloo", init_method=(root / "gloo").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=12))
    device = torch.device("cpu")
    try:
        group = TrainingGroup(device)
        result = _fit_fixture(root / "ddp", group)
        assert (result is not None) == (rank == 0)
        payload = torch.load(root / "ddp/checkpoint_last.pt", weights_only=False)
        assert len(payload["rng_by_rank"]) == 2
        assert not torch.equal(payload["rng_by_rank"][0]["torch"], payload["rng_by_rank"][1]["torch"])
        torch.manual_seed(999)
        restore_training_rng(payload, device, rank)
        assert torch.equal(torch.get_rng_state(), payload["rng_by_rank"][rank]["torch"])
        resumed = _fit_fixture(root / "ddp", group, resume=root / "ddp/checkpoint_best.pt")
        if group.primary:
            assert resumed["global_step"] == 2 and resumed["stopped_early"]
    finally:
        dist.destroy_process_group()


class DDPTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "PyTorch Gloo required")
    def test_two_ranks_match_single_process_and_stop_together(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _fit_fixture(root / "single")
            mp.spawn(_rank_worker, args=(folder,), nprocs=2, join=True)
            single = torch.load(root / "single/checkpoint_last.pt", weights_only=False)
            ddp = torch.load(root / "ddp/checkpoint_last.pt", weights_only=False)
            for payload in (single, ddp):
                self.assertEqual(payload["global_step"], 2)
                self.assertEqual(payload["early_stopping"]["stop_step"], 2)
                self.assertEqual(payload["early_stopping"]["best_step"], 1)
            for key, expected in single["model"].items():
                torch.testing.assert_close(ddp["model"][key], expected)
            train_rows = []
            for mode in ("single", "ddp"):
                path = root / mode / "metrics.jsonl"
                rows = read_metric_rows(path)
                train_rows.append([row for row in rows if row.get("event") is None])
            history = (root / "ddp/metrics.jsonl").read_text().splitlines()
            resumes = [json.loads(line) for line in history if line and '"event": "resume"' in line]
            self.assertEqual([(row["global_step"], row["world_size"]) for row in resumes], [(1, 2)])
            self.assertEqual([row["global_step"] for row in train_rows[1]], [1, 2])
            for reference, distributed in zip(*train_rows):
                self.assertAlmostEqual(reference["loss_total"], distributed["loss_total"], places=6)
                self.assertEqual(reference["exact_token_accuracy"], distributed["exact_token_accuracy"])

    def test_legacy_rng_is_restored_only_for_rank_zero(self):
        device = torch.device("cpu")
        torch.manual_seed(42)
        payload = {"config": {"device": "cpu"}, "rng": rng_state(device)}
        torch.manual_seed(43)
        new_rank_state = torch.get_rng_state()
        restore_training_rng(payload, device, rank=1)
        self.assertTrue(torch.equal(torch.get_rng_state(), new_rank_state))
        restore_training_rng(payload, device, rank=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), payload["rng"]["torch"]))


if __name__ == "__main__":
    unittest.main()
