from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import yaml

REQUIRED_SEMANTICS = {"cone", "qrcode", "marker", "no_go_white_cross"}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误: {path}")
    return data


def load_vision_config(config_path: str, model_path: str | None = None, mapping_file: str | None = None) -> Dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"主配置文件不存在: {config_file}")

    config = _load_yaml(config_file)
    model_cfg = config.setdefault("model", {})
    mapping_cfg = config.setdefault("class_name_to_semantic", {})

    if model_path:
        model_cfg["weights_path"] = model_path

    if mapping_file:
        override = _load_yaml(Path(mapping_file))
        mapping_cfg = override.get("class_name_to_semantic", override)
        config["class_name_to_semantic"] = mapping_cfg

    return config


def validate_vision_config(config: Dict[str, Any]) -> str:
    model = config.get("model", {})
    model_path = Path(str(model.get("weights_path", "")))
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    mapping = config.get("class_name_to_semantic", {})
    if not isinstance(mapping, dict):
        raise ValueError("class_name_to_semantic 必须是字典")

    semantics = set(mapping.values())
    missing = sorted(REQUIRED_SEMANTICS - semantics)
    if missing:
        raise ValueError(f"类别映射缺少内部语义: {missing}")

    version = str(model.get("version", "unknown"))
    return version
