from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from four_d_vggt.compat.checkpoint import checkpoint_payload
from four_d_vggt.models.model import FourDVGGT
from infer import PREDICTION_KEYS, load_model, predict


ROOT = Path(__file__).resolve().parents[1]


def tiny_config():
    return {
        "profile": "4d-vggt-geometry-v1",
        "img_size": 56,
        "embed_dim": 64,
        "depth": 4,
        "num_heads": 4,
        "patch_embed": "conv",
        "cached_layer_indices": [0, 1, 2, 3],
        "spatial_head_options": {"num_heads": 4, "trunk_depth": 2},
        "dense_head_options": {"features": 16, "out_channels": [16, 32, 64, 64]},
    }


class GeometryReleaseTest(unittest.TestCase):
    def test_source_tree_excludes_unreleased_heads_and_training(self):
        self.assertFalse(any((ROOT / "training").rglob("*.py")))
        self.assertFalse((ROOT / "four_d_vggt/heads/temporal_head.py").is_file())
        self.assertFalse(any((ROOT / "four_d_vggt/heads/track_modules").rglob("*.py")))
        forbidden = ("dynamic_mask", "temporal_head", "track_modules", "query_points")
        sources = [ROOT / "infer.py", *sorted((ROOT / "four_d_vggt").rglob("*.py"))]
        for source in sources:
            text = source.read_text().lower()
            for token in forbidden:
                self.assertNotIn(token, text, f"{token} leaked into {source.relative_to(ROOT)}")

    def test_checkpoint_roundtrip_and_output_contract(self):
        torch.manual_seed(3)
        model = FourDVGGT(**tiny_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "geometry.pt"
            torch.save(checkpoint_payload(model), checkpoint)
            loaded, report = load_model(checkpoint, device="cpu")
            self.assertEqual(report["released_tasks"], ["camera", "depth", "point"])
            images = torch.rand(1, 2, 3, 56, 56)
            expected, _ = predict(model, images, camera_decoding="none")
            actual, _ = predict(loaded, images, camera_decoding="none")
            self.assertEqual(set(actual), PREDICTION_KEYS)
            for key in expected:
                self.assertTrue(torch.equal(torch.from_numpy(expected[key]), torch.from_numpy(actual[key])), key)


if __name__ == "__main__":
    unittest.main()
