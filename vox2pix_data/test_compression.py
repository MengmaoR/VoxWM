"""Tiny temporary fixtures for v3 byte accounting; no Native collection."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from numcodecs import Blosc
import numpy as np
import zarr

from .compression import compression_report
from .protocol import V3_FINAL_PROFILE, v3_plan, v3_version
from .storage import RAW_FIELDS, V3_WHITE_FIELDS, scene_shapes, write_json


class CompressionTests(unittest.TestCase):
    def fixture(self, root, mode="production"):
        config = dict(mode=mode, scenes=10, seed=42, views_per_scene=5, width=2, height=2,
                      lighting_profile=V3_FINAL_PROFILE,
                      scene_kinds=["surface", "cave", "underwater"], scenes_per_kind=1)
        write_json(root / "manifest.json", dict(version=v3_version(config), config=config, axes="ENU"))
        write_json(root / "node_vocabulary.json", ["air", "stone"])
        slot = v3_plan(config)[0]
        scene = root / "scenes" / "scene_0000"
        (scene / "rgb").mkdir(parents=True)
        write_json(scene / "scene.json", dict(slot, surface_biome="fixture", split="train"))
        group = zarr.open_group(str(scene / "voxels.zarr"), mode="w")
        for name, dtype in (RAW_FIELDS | V3_WHITE_FIELDS).items():
            group.create_dataset(name, data=np.zeros(scene_shapes(config)[name], dtype=dtype),
                                 compressor=Blosc(cname="zstd", clevel=3))
        for index in range(5):
            (scene / "rgb" / f"{index:03d}.png").write_bytes(b"size-only fixture")
        return config, scene

    def test_all_v3_fields_and_actual_frame_projection(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self.fixture(root)
            expected = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            (root / "compression_report.md").write_text("previous report" * 1000)
            (root / "collection_progress.json").write_text("excluded runtime state")
            with patch("zarr.core.Array.__getitem__", side_effect=AssertionError("must not decode arrays")):
                result = compression_report(root)
            self.assertEqual(result["pilot_total_bytes"], expected)
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual(result["frame_count"], 5)
            self.assertEqual(set(result["white_model"]), set(V3_WHITE_FIELDS) | {"total"})
            self.assertEqual(result["white_model"]["total"]["raw_bytes"], 5 * 2 * 2 * 5)
            self.assertEqual(result["projection_frame_scale"], 10000)
            fixed = result["metadata"]["fixed_dataset_bytes"]
            self.assertEqual(result["projected_50k_total_bytes"], (expected - fixed) * 10000 + fixed)
            self.assertNotIn("recommended_resolution", result)
            self.assertEqual(compression_report(root)["pilot_total_bytes"], expected)

    def test_auxiliary_images_are_not_official_frames(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, scene = self.fixture(root, mode="smoke")
            auxiliary = scene / "lighting_comparison"
            auxiliary.mkdir()
            (auxiliary / "control.png").write_bytes(b"extra")
            result = compression_report(root)
            self.assertEqual(result["frame_count"], 5)
            self.assertEqual(result["auxiliary_lighting_pngs"]["actual_bytes"], 5)
            fixed = result["metadata"]["fixed_dataset_bytes"]
            self.assertEqual(result["projected_50k_total_bytes"],
                             (result["pilot_total_bytes"] - fixed - 5) * 10000 + fixed)

    def test_legacy_dataset_keeps_legacy_report(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "manifest.json", dict(config=dict(mode="pilot-v2")))
            with patch("vox2pix_data.qa.compression_report", return_value={"legacy": True}) as old:
                self.assertEqual(compression_report(root), {"legacy": True})
                old.assert_called_once_with(root)


if __name__ == "__main__":
    unittest.main()
