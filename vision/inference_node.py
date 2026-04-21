#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision.config_loader import load_vision_config, validate_vision_config


def parse_args() -> argparse.Namespace:
    return _build_parser(REPO_ROOT).parse_args()


def _build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vision inference node")
    parser.add_argument(
        "--config",
        default=str(repo_root / "config" / "vision_model.yaml"),
        help="Path to main configuration file",
    )
    parser.add_argument("--model-path", default=None, help="Override model weights path")
    parser.add_argument("--mapping-file", default=None, help="Override class mapping file")
    return parser


def main() -> int:
    args = parse_args()
    cfg = load_vision_config(args.config, args.model_path, args.mapping_file)
    version = validate_vision_config(cfg)
    model_path = cfg["model"]["weights_path"]
    print(f"[vision] model_version={version} model_path={model_path}")
    print("[vision] inference node initialized with centralized config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
