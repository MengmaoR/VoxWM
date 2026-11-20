import contextlib
import io
import tempfile
import unittest
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

from voxel_vae.config import load_config, save_config
from voxel_vae.dataset import DatasetIndex, SampleRef
from voxel_vae.e00 import _write_json
from voxel_vae.e01 import DEFAULT_CONFIG, PHASE, build_subset_manifest, run
from voxel_vae.sampler import BALANCE, BalancedBatchSampler
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
            with patch("voxel_vae.e01._fit_prepared_masked", return_value={}) as fit:
                with contextlib.redirect_stdout(io.StringIO()):
                    run(Namespace(output_root=output, device="cuda:0", resume=False))
                args, kwargs = fit.call_args
        _, _, phase_config, _, plan_index, train, diagnostics, validation = args
        self.assertEqual(train, plan_index.indices(split="train"))
        self.assertEqual(len(diagnostics), 160)
        self.assertGreater(len(train), len(diagnostics))
        self.assertEqual(kwargs["phase"], PHASE)
        self.assertTrue(kwargs["lazy_masks"])
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


if __name__ == "__main__":
    unittest.main()
