"""
Apply a conservative ICLabel artifact-exclusion policy to NeuroCogEEG YAML files.

This script updates experiment YAML files so that ICLabel keeps ``brain`` and
``other`` components, while excluding only known artifact labels with sufficient
ICLabel confidence.

Run from the repository root:
    python pipelines/apply_iclabel_known_artifacts_policy.py
"""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = [
    PROJECT_ROOT / "configs" / "flanker.yaml",
    PROJECT_ROOT / "configs" / "gonogo.yaml",
    PROJECT_ROOT / "configs" / "readysetgo.yaml",
    PROJECT_ROOT / "configs" / "tmt.yaml",
]

POLICY = {
    "enabled": True,
    "method": "iclabel",
    # New conservative policy:
    "exclude_mode": "known_artifacts_only",
    "keep_labels": [
        "brain",
        "other",
    ],
    "exclude_labels": [
        "eye",
        "eye blink",
        "eye movement",
        "muscle",
        "muscle artifact",
        "heart",
        "heart beat",
        "line noise",
        "channel noise",
    ],
    "min_exclusion_probability": 0.70,
    # Keep the old key for compatibility/documentation. The updated
    # preprocessing.py ignores this when exclude_mode is present.
    "exclude_non_brain": False,
}


def update_yaml(path: Path) -> bool:
    if not path.exists():
        print(f"Skipped missing config: {path}")
        return False

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if "preprocessing" not in config:
        raise KeyError(f"Missing preprocessing section in {path}")

    iclabel = dict(config["preprocessing"].get("iclabel", {}))
    iclabel.update(POLICY)
    config["preprocessing"]["iclabel"] = iclabel

    backup = path.with_suffix(path.suffix + ".pre_iclabel_known_artifacts.bak")

    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

    print(f"Updated: {path}")
    return True


def main() -> None:
    updated_count = 0

    for path in CONFIG_FILES:
        if update_yaml(path):
            updated_count += 1

    if updated_count == 0:
        raise SystemExit("No config files were updated.")

    print("FINAL RESULT: PASS")


if __name__ == "__main__":
    main()
