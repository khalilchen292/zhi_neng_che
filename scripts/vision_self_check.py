#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision.config_loader import load_vision_config, validate_vision_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vision startup self check")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "vision_model.yaml"),
        help="Path to main configuration file",
    )
    parser.add_argument("--model-path", default=None, help="Override model weights path")
    parser.add_argument("--mapping-file", default=None, help="Override class mapping file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_vision_config(args.config, args.model_path, args.mapping_file)
    version = validate_vision_config(cfg)
    print(f"[self-check] model_version={version}")
    print(f"[self-check] model_path={cfg['model']['weights_path']}")
    print("[self-check] mapping_ok=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
