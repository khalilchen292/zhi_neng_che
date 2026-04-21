from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from vision.config_loader import load_vision_config, validate_vision_config


class VisionConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_vision_config(str(repo_root / "config" / "vision_model.yaml"))
        version = validate_vision_config(cfg)
        self.assertTrue(version)

    def test_missing_semantic_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = tmp_path / "model.onnx"
            model_path.write_bytes(b"")
            cfg_path = tmp_path / "cfg.yaml"
            cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"version": "x", "weights_path": str(model_path)},
                        "class_name_to_semantic": {"cone": "cone"},
                    }
                ),
                encoding="utf-8",
            )
            cfg = load_vision_config(str(cfg_path))
            with self.assertRaises(ValueError):
                validate_vision_config(cfg)


if __name__ == "__main__":
    unittest.main()
