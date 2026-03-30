"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).resolve()
    with file_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in YAML file {file_path}, got {type(payload)!r}")
    return payload


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config and resolve referenced sub-config files."""
    experiment_path = Path(path).resolve()
    experiment = _load_yaml(experiment_path)

    resolved = {
        "experiment_name": experiment.get("name", experiment_path.stem),
        "experiment_path": str(experiment_path),
        "data": _resolve_section(experiment, "data", base_dir=experiment_path.parent),
        "model": _resolve_section(experiment, "model", base_dir=experiment_path.parent),
        "loss": dict(experiment.get("loss", {})),
        "optimizer": dict(experiment.get("optimizer", {})),
        "trainer": _resolve_section(experiment, "trainer", base_dir=experiment_path.parent),
    }
    return resolved


def _resolve_section(
    experiment: dict[str, Any], key: str, *, base_dir: Path
) -> dict[str, Any]:
    section = dict(experiment.get(key, {}))
    config_ref = section.pop("config", None)
    if config_ref is None:
        return section

    config_path = Path(config_ref)
    if not config_path.is_absolute():
        base_candidate = (base_dir / config_path).resolve()
        cwd_candidate = config_path.resolve()
        config_path = base_candidate if base_candidate.exists() else cwd_candidate

    resolved = _load_yaml(config_path)
    resolved.update(section)
    resolved["_config_path"] = str(config_path)
    return resolved
