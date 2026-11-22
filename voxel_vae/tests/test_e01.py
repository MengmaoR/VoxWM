import contextlib
import io
import tempfile
import unittest
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from voxel_vae.config import load_config, save_config
from voxel_vae.dataset import DatasetIndex, SampleRef
from voxel_vae.e00 import _write_json
from voxel_vae.e01 import DEFAULT_CONFIG, PHASE, build_subset_manifest, rebalance, run
from voxel_vae.e01_split import SPLIT_FILE
from voxel_vae.sampler import BALANCE, BalancedBatchSampler, BalancedEpochSampler
from voxel_vae.vocabulary import CanonicalVocabulary


class E01Tests(unittest.TestCase):
    def test_full_train_pool_and_fixed_held_out_chains(self):
        vocab = CanonicalVocabulary((("air", 0),))
        records = []

        def add(root, split, category, seed, chain=None, chain_index=None):
            records.append(SampleRef(
                root, 0, f"/clip_{root}/shards/shard_0000", len(records),
                category, chain, chain_index, seed, split, "Plains", "surface", None))

        for root in range(2):
            for split, static, chains in (("train", 50, 5), ("validation", 12, 3), ("test", 2, 1)):
                for _ in range(static):
                    add(root, split, "Static", len(records))
                for category in ("D1", "D2", "D3", "D4"):
                    for number in range(chains):
                        seed = len(records)
                        for frame in range(5):
                            add(root, split, category, seed, f"{split}-{category}-{number}", frame)
        index = DatasetIndex(["/clip_0", "/clip_1"], records, ["a", "b"], vocab.sha256)
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(config["training"]["precision"], "bf16")
        self.assertEqual(config["training"]["max_steps"], 20_000)
        self.assertEqual(config["training"]["validate_every"], 1000)
        self.assertEqual(config["validation"], {"static_states": 640, "chains_per_dynamic_category": 32})
        config["validation"] = {"static_states": 16, "chains_per_dynamic_category": 4}
        manifest = build_subset_manifest(index, vocab, config)
        self.assertEqual(manifest, build_subset_manifest(index, vocab, config))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            vocab.save(output / "training_vocabulary.json")
            index.save(output / "dataset_index.json")
            _write_json(output / "subset_manifest.json", manifest)
            save_config(output / "config.yaml", config)
            run_args = Namespace(output_root=output, device="cuda:0", resume=True, epochs=2,
                                 no_early_stop=True)
            with patch("voxel_vae.e01._fit_prepared_masked", return_value={}) as fit:
                with contextlib.redirect_stdout(io.StringIO()):
                    run(run_args)
                args, kwargs = fit.call_args
            counts = {"train": 305, "validation": 149, "test": 34}
            with patch.dict("voxel_vae.e01_split.SPLIT_COUNTS", counts), \
                    patch("voxel_vae.e01._fit_prepared_masked", return_value={}) as fit, \
                    contextlib.redirect_stdout(io.StringIO()):
                rebalance(Namespace(output_root=output))
                split_file = output / SPLIT_FILE
                saved_split = split_file.read_bytes()
                rebalance(Namespace(output_root=output))
                self.assertEqual(split_file.read_bytes(), saved_split)
                run(run_args)
                migrated_args, migrated_kwargs = fit.call_args
            self.assertEqual(migrated_args[6:], args[6:])
            self.assertEqual(migrated_args[4].summary()["splits"], counts)
            self.assertEqual(migrated_kwargs["previous_prepared_data"], args[2]["prepared_data"])
            self.assertIn(SPLIT_FILE, migrated_args[2]["prepared_data"])
            for old, new in zip(index.records, migrated_args[4].records):
                self.assertEqual(old.key, new.key)
                if old.split != "test":
                    self.assertEqual(old, new)
        _, _, phase_config, _, plan_index, train, diagnostics, validation = args
        self.assertEqual(train, plan_index.indices(split="train"))
        self.assertEqual(len(diagnostics), 160)
        self.assertGreater(len(train), len(diagnostics))
        self.assertEqual(kwargs["phase"], PHASE)
        self.assertTrue(kwargs["lazy_masks"])
        self.assertEqual((kwargs["resume"], kwargs["epochs"]), ("auto", 2))
        self.assertTrue(kwargs["no_early_stop"])
        self.assertEqual(phase_config["early_stopping"], config["early_stopping"])
        self.assertTrue(phase_config["prepared_data"]["dataset_index.json"])
        self.assertEqual(kwargs["metadata"]["training_pool"]["by_root"], {"/clip_0": 150, "/clip_1": 150})
        sampler = BalancedBatchSampler(plan_index, train)
        self.assertEqual(sum(map(len, sampler.pools.values())), len(train))
        self.assertEqual(sampler.counts(sampler.batch(1)), Counter(BALANCE))

        grouped = defaultdict(list)
        for item in validation:
            row = plan_index.records[item]
            self.assertEqual(row.split, "validation")
            if row.chain_id is not None:
                grouped[row.root_index, row.chain_id].append(row.chain_index)
        self.assertEqual(len(validation), 16 + 4 * 4 * 5)
        self.assertEqual(len(grouped), 4 * 4)
        self.assertTrue(all(frames == list(range(5)) for frames in grouped.values()))

    def test_each_epoch_covers_unequal_training_pools_exactly_once(self):
        sizes = {"Static": 79_751, "D1": 20_105, "D2": 19_980, "D3": 19_865, "D4": 20_025}
        records = []
        for category, size in sizes.items():
            records.extend([SimpleNamespace(category=category, split="train")] * size)
        pool_size = len(records)
        records.append(SimpleNamespace(category="Static", split="validation"))
        records.append(SimpleNamespace(category="D1", split="test"))
        index = SimpleNamespace(records=records)
        sampler = BalancedEpochSampler(index, range(len(records)), seed=42)
        self.assertEqual(sampler.steps_per_epoch, 2496)
        self.assertNotEqual(sampler.batch(0), sampler.batch(2496))
        for epoch in range(2):
            seen = set()
            for step in range(epoch * 2496, (epoch + 1) * 2496):
                batch = sampler.batch(step)
                self.assertIn(len(batch), (63, 64))
                self.assertEqual(len(set(batch)), len(batch))
                self.assertTrue(seen.isdisjoint(batch))
                for category, count in sampler.counts(batch).items():
                    base = sizes[category] // 2496
                    self.assertIn(count, (base, base + 1))
                seen.update(batch)
            self.assertEqual(seen, set(range(pool_size)))
        resumed = BalancedEpochSampler(index, range(len(records)), seed=42)
        self.assertEqual(resumed.batch(3123), sampler.batch(3123))


if __name__ == "__main__":
    unittest.main()
