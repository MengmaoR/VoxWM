"""Small CPU fixtures for the frozen inference metrics and sampling contract."""
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

stub_zarr = importlib.util.find_spec("zarr") is None
if stub_zarr:
    sys.modules["zarr"] = ModuleType("zarr")

from voxel_vae import dataset, freeze_audit
from voxel_vae.change_masks import compute_chain_masks
from voxel_vae.config import load_config, save_config
from voxel_vae.e00 import _sha256
from voxel_vae.e01 import DEFAULT_CONFIG, TRAIN_DIAGNOSTICS, VALIDATION
from voxel_vae.evaluate import evaluate_indices
from voxel_vae.freeze_d3 import D3Audit
from voxel_vae.freeze_statistics import Moments, TrainLatents, add_token_counts, embedding_coverage
from voxel_vae.latent_normalization import denormalize_latent, normalize_latent
from voxel_vae.vocabulary import CanonicalVocabulary

if stub_zarr:
    del sys.modules["zarr"]


class FixtureIndex:
    def __init__(self, records):
        self.records = records

    def indices(self, split):
        return [i for i, row in enumerate(self.records) if row.split == split]


def record(category, step=None, *, seed=42, split="test"):
    return SimpleNamespace(
        category=category, chain_id=None if step is None else category, chain_index=step,
        root_index=0, shard=0, world_seed=seed, split=split, biome_group="fixture", stratum="surface")


class PosteriorFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 32
        self.latent_channels = 16
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def architecture_config(self):
        return {"fixture": True}

    def encode(self, tokens):
        assert not self.training and not torch.is_grad_enabled()
        self.target = tokens
        self.expected = tokens[:, ::4, ::4, ::4].float()[:, None].expand(-1, 16, -1, -1, -1) * self.weight
        return self.expected, torch.full_like(self.expected, -0.5)

    def encode_mean(self, tokens):
        return self.encode(tokens)[0]

    def decode(self, mu):
        assert torch.equal(mu, self.expected)
        prediction = self.target.clone()
        prediction[:, 0, 0, 1] = (prediction[:, 0, 0, 1] + 1) % 4
        return F.one_hot(prediction, num_classes=4).movedim(-1, 1).float() * 4

    def forward(self, tokens):
        raise AssertionError("Audit must explicitly use the posterior mean")


class MaskFixture(dict):
    def changed(self, i):
        return self[i]

    def statistics(self, _):
        return {}


class FreezeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_semantics_pairs_and_spatial_accounting(self):
        vocab = CanonicalVocabulary((("air", 0), ("mcl_fire:fire", 0),
                                     ("mcl_fire:fire", 7), ("stone", 0)))
        audit = D3Audit(vocab)
        before = np.zeros((1, 16, 16, 16), dtype=int)
        before[0, 0, 0, :3] = [1, 2, 3]
        pred = before.copy()
        pred[0, 0, 0, :3] = [1, 1, 0]
        audit.observe_states(before, pred, before > 0)
        after = before[0].copy()
        after[0, 0, 1] = 1
        audit.observe_transition(before[0], after, pred[0], after)
        transitions = audit.transition_rows()
        stats = audit.summary(transitions)
        self.assertEqual(sum(stats[k] for k in (
            "exact_correct_count", "param2_only_error_count", "node_error_count")), 3)
        self.assertEqual(stats["total_changed_voxels"], 3)
        self.assertEqual(stats["node_name_accuracy"], 2 / 3)
        self.assertEqual(stats["param2_accuracy_given_node_correct"], 1 / 2)
        self.assertEqual(audit.confusion_rows()[0]["gt_param2"], 7)
        self.assertEqual(audit.confusion_rows(node_only=True)[0]["gt_node_name"], "stone")
        self.assertEqual(transitions[0]["count"], 1)
        self.assertEqual(transitions[0]["pair_exact_reconstruction_rate"], 0)
        self.assertEqual(transitions[0]["node_name_pair_exact_rate"], 1)
        self.assertEqual(audit.spatial_summary()["marginals"]["distance_to_boundary"]["errors"][0], 2)

    def test_exact_split_counts_and_seed_isolation(self):
        rows = ([record("Static", seed=1, split="train")] * 160_000
                + [record("Static", seed=2, split="validation")] * 20_000
                + [record("Static", seed=3)] * 19_995
                + [record("D4", step, seed=4) for step in reversed(range(5))])
        manifest = {"subsets": {TRAIN_DIAGNOSTICS: [{"sample_index": 0}],
                                VALIDATION: [{"sample_index": 160_000}]}}
        index = FixtureIndex(rows)
        selected, groups, result = freeze_audit.audit_split(index, manifest)
        self.assertEqual(len(selected["test"]), 20_000)
        self.assertEqual(sum(map(len, groups)), 20_000)
        self.assertEqual([rows[i].chain_index for i in groups[-1]], list(range(5)))
        self.assertEqual(result["d4_anchor_samples"], 0)
        with self.assertRaisesRegex(ValueError, "Expected split counts"):
            freeze_audit.audit_split(FixtureIndex(rows[:-1]), manifest)
        rows[-6] = record("Static", seed=1)
        with self.assertRaisesRegex(ValueError, "World seed split leakage"):
            freeze_audit.audit_split(index, manifest)
        with self.assertRaisesRegex(ValueError, "complete five-state"):
            freeze_audit.ordered_groups(index, selected["test"][:-1])

    def test_d4_anchor_index_is_not_a_committed_sample(self):
        with TemporaryDirectory() as tmp:
            shard = Path(tmp) / "shard_0000"
            shard.mkdir()
            with closing(sqlite3.connect(shard / "journal.sqlite")) as db, db:
                db.executescript("CREATE TABLE chains(data TEXT, anchor_idx INTEGER);"
                                 "CREATE TABLE samples(idx INTEGER, category TEXT, metadata TEXT);")
                db.execute("INSERT INTO chains VALUES (?,?)",
                           (json.dumps({"id": "chain", "snapshot_indices": list(range(5))}), 800))
                for i in range(5):
                    db.execute("INSERT INTO samples VALUES (?,?,?)", (i, "D4", json.dumps({
                        "snapshot_id": i, "category": "D4", "chain_id": "chain", "chain_index": i,
                        "world_seed": 13, "split": "test"})))
            with patch.object(dataset.zarr, "open_group", return_value={"tokens": range(5)}, create=True):
                rows = dataset._read_shard(0, shard)
        self.assertEqual([r.snapshot_index for r in rows], list(range(5)))

    def test_vocab_frequency_and_embedding_updates(self):
        vocab = CanonicalVocabulary((("air", 0), ("fire", 0), ("fire", 2), ("stone", 0)))
        counts = {name: np.zeros(4, dtype=np.int64) for name in ("train", "validation", "test")}
        add_token_counts(counts["train"], np.array([[0, 0], [1, 0]]))
        add_token_counts(counts["validation"], np.array([2]))
        initial = np.ones((4, 2))
        final = initial.copy()
        final[0] = [4, 5]
        frequencies, rows, summary = embedding_coverage(vocab, counts, initial, final)
        self.assertEqual([row["train_count"] for row in frequencies], [3, 1, 0, 0])
        self.assertEqual(rows[0]["embedding_l2_drift"], 5)
        self.assertAlmostEqual(rows[0]["embedding_cosine"], 9 / np.sqrt(82))
        self.assertEqual(summary["seen_in_train"], 2)
        self.assertEqual(summary["unseen_in_train_but_seen_in_held_out"], 1)
        self.assertEqual(summary["never_in_any_center_split"], 1)
        self.assertEqual([row["token_id"] for row in summary["seen_but_no_drift"]], [1])

    def test_streaming_latent_statistics_and_normalization(self):
        mu = np.random.default_rng(3).normal(size=(9, 16, 4, 4, 4))
        category = ["Static"] * 3 + ["D3"] * 6
        train = TrainLatents(16)
        train.update(mu[:4], category[:4])
        train.update(mu[4:], category[4:])
        s = train.summary()
        np.testing.assert_allclose(s["mean"], mu.mean(axis=(0, 2, 3, 4)))
        np.testing.assert_allclose(s["std"], mu.std(axis=(0, 2, 3, 4)))
        np.testing.assert_allclose(s["by_category"]["D3"]["std"], mu[3:].std(axis=(0, 2, 3, 4)))
        self.assertAlmostEqual(s["global_std"], float(mu.std()))
        combined = Moments(16)
        combined.update(np.moveaxis(mu, 1, -1).reshape(-1, 16))
        np.testing.assert_allclose(s["variance"], combined.summary()["variance"])
        stats = {**s, "latent_shape": [16, 4, 4, 4], "eps": 1e-6}
        values = torch.from_numpy(mu)
        torch.testing.assert_close(denormalize_latent(normalize_latent(values, stats), stats), values)

    def test_candidate_is_loaded_read_only_and_checks_prepared_data(self):
        vocab = CanonicalVocabulary(tuple((f"node_{i:04d}", 0) for i in range(1580)))
        config = load_config(DEFAULT_CONFIG)
        identity = {"split_160k.json": "fixture"}
        model = PosteriorFixture()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "dataset_manifest.json"
            manifest.write_text("{}")
            index = SimpleNamespace(roots=[str(root)], manifest_hashes=[_sha256(manifest)])
            save_config(root / "config.yaml", config)
            checkpoint = root / "checkpoint_best.pt"
            torch.save({
                "model": model.state_dict(), "vocabulary_K": vocab.K, "vocabulary_sha256": vocab.sha256,
                "architecture": model.architecture_config(), "latent_shape": [16, 4, 4, 4],
                "dataset_roots": index.roots, "dataset_manifest_hashes": index.manifest_hashes,
                "phase": "E01-Masked", "config": {"loss": config["loss"], "prepared_data": identity},
                "global_step": 27_000, "git_commit": "fixture", "best_metric": 0.25,
            }, checkpoint)
            original = checkpoint.read_bytes()
            prepared = ((vocab, index, {}), identity, {})
            with (patch.object(freeze_audit, "VOCABULARY_HASH", vocab.sha256),
                  patch.object(freeze_audit, "_model", return_value=model),
                  patch.object(freeze_audit, "load_prepared_e01", return_value=prepared),
                  patch.object(torch.optim.AdamW, "step", side_effect=AssertionError)):
                *_, frozen, provenance = freeze_audit.open_candidate(root, checkpoint)
                self.assertEqual(provenance["checkpoint_step"], 27_000)
                self.assertFalse(frozen.training)
                self.assertFalse(frozen.weight.requires_grad)
                self.assertEqual(checkpoint.read_bytes(), original)
                identity["split_160k.json"] = "changed"
                with self.assertRaisesRegex(ValueError, "config.prepared_data"):
                    freeze_audit.open_candidate(root, checkpoint)

    def test_streaming_matches_the_existing_e01_evaluator(self):
        vocab = CanonicalVocabulary((("air", 0), ("mcl_fire:fire", 0), ("mcl_fire:fire", 1), ("stone", 0)))
        rows = [record("Static")]
        states = [np.full((16, 16, 16), 3, dtype=np.int64)]
        masks = MaskFixture()
        for category in ("D1", "D2", "D3", "D4"):
            chain = np.zeros((5, 16, 16, 16), dtype=np.int64)
            for i in range(5):
                chain[i, 0, 0, :2] = [(i + 1) % 4, i % 4]
            offset = len(rows)
            rows.extend(record(category, i) for i in range(5))
            states.extend(chain)
            masks.update(enumerate(compute_chain_masks(chain), start=offset))
        index = FixtureIndex(rows)
        indices = index.indices("test")
        groups, _ = freeze_audit.ordered_groups(index, indices)
        config = load_config(DEFAULT_CONFIG)
        config["training"]["precision"] = "fp32"

        class ToyDataset:
            def __init__(self, _index, _vocab, selected):
                self.selected = selected

            def __len__(self):
                return len(self.selected)

            def read_numpy(self, i):
                return states[self.selected[i]], self.selected[i]

            def __getitem__(self, i):
                tokens, sample_index = self.read_numpy(i)
                return {"tokens": torch.from_numpy(tokens), "sample_index": sample_index}

        cpu = torch.device("cpu")
        model = PosteriorFixture()
        with (TemporaryDirectory() as tmp,
              patch("voxel_vae.evaluate.VoxelSnapshotDataset", ToyDataset),
              patch.object(freeze_audit, "VoxelSnapshotDataset", ToyDataset),
              patch.object(torch.optim.AdamW, "step", side_effect=AssertionError)):
            path = Path(tmp)
            reference = evaluate_indices(model, index, vocab, indices, cpu, batch_size=3,
                                         change_masks=masks, masked_loss_config=config["loss"])
            model.train()
            counts = freeze_audit.evaluate_streaming(index, vocab, groups, model, config, cpu, 3, 0, path)
            self.assertEqual(int(counts.sum()), 21 * 16 ** 3)
            self.assertFalse(model.training)
            self.assertIsNone(model.weight.grad)
            streamed = json.loads((path / "test_metrics.json").read_text())
        for key in ("cross_entropy", "exact_token_accuracy", "non_air_exact_token_accuracy"):
            self.assertAlmostEqual(streamed[key], reference[key], places=6)
        for key in ("static_preservation", "dynamic", "categories"):
            self.assertEqual(streamed[key], reference[key])
        for key, expected in reference["loss_components"].items():
            actual = streamed["loss_components"][key]
            if isinstance(expected, dict):
                np.testing.assert_allclose(list(actual.values()), list(expected.values()), atol=1e-6)
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        for key, value in reference["latent"]["adjacent_dynamic_latent_l2"].items():
            self.assertAlmostEqual(streamed["latent"]["adjacent_dynamic_latent_l2"][key], value)
        for category, expected in reference["dynamic_by_category"].items():
            for metric, value in expected.items():
                self.assertEqual(streamed["dynamic_by_category"][category][metric], value)
        for key, expected in reference["dynamic_reconstruction"].items():
            if key != "by_category":
                self.assertEqual(streamed["dynamic_reconstruction"][key], expected)


if __name__ == "__main__":
    unittest.main()
