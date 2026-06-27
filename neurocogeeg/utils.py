"""
General helper utilities for NeuroCogEEG.

This module contains small reusable helper functions that are not specific
to any single EEG analysis method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    file_path:
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        YAML content as a Python dictionary.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.

    ValueError
        If the YAML file is empty or does not contain a dictionary.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if content is None:
        raise ValueError(f"YAML file is empty: {path}")

    if not isinstance(content, dict):
        raise ValueError(
            f"YAML file must contain a dictionary at the top level: {path}"
        )

    return content


def save_yaml(data: dict[str, Any], file_path: str | Path) -> None:
    """
    Save a dictionary as a YAML file.

    Parameters
    ----------
    data:
        Dictionary to save.

    file_path:
        Destination YAML file path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def ensure_directory(directory_path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    directory_path:
        Directory path to create.

    Returns
    -------
    Path
        Created or existing directory path.
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path