"""
Export utilities for NeuroCogEEG.

This module contains helper functions for saving analysis results in
SPSS-compatible CSV format and for saving configuration snapshots.

The module does not perform EEG analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from neurocogeeg.utils import save_yaml


def to_dataframe(
    rows: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert analysis rows into a pandas DataFrame.

    Parameters
    ----------
    rows:
        Either a list of dictionaries or an existing DataFrame.

    Returns
    -------
    pd.DataFrame
        Converted DataFrame.

    Raises
    ------
    ValueError
        If the provided list is empty.
    TypeError
        If input type is unsupported.
    """
    if isinstance(rows, pd.DataFrame):
        return rows.copy()

    if isinstance(rows, list):
        if len(rows) == 0:
            raise ValueError("No rows were provided for export.")

        return pd.DataFrame(rows)

    raise TypeError(
        "rows must be either a list of dictionaries or a pandas DataFrame."
    )


def save_spss_csv(
    rows: list[dict[str, Any]] | pd.DataFrame,
    output_path: str | Path,
    sep: str = ";",
    decimal: str = ",",
    encoding: str = "utf-8-sig",
    index: bool = False,
) -> Path:
    """
    Save results in SPSS-friendly CSV format.

    Parameters
    ----------
    rows:
        Analysis results as a list of dictionaries or DataFrame.

    output_path:
        Destination CSV path.

    sep:
        Column separator. Default is semicolon, suitable for Turkish locale.

    decimal:
        Decimal separator. Default is comma, suitable for Turkish locale.

    encoding:
        File encoding. ``utf-8-sig`` helps Excel recognize Turkish characters.

    index:
        Whether to write the DataFrame index.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = to_dataframe(rows)

    dataframe.to_csv(
        path,
        sep=sep,
        decimal=decimal,
        encoding=encoding,
        index=index,
    )

    return path


def save_excel(
    rows: list[dict[str, Any]] | pd.DataFrame,
    output_path: str | Path,
    index: bool = False,
) -> Path:
    """
    Save results as an Excel file.

    Parameters
    ----------
    rows:
        Analysis results as a list of dictionaries or DataFrame.

    output_path:
        Destination Excel path.

    index:
        Whether to write the DataFrame index.

    Returns
    -------
    Path
        Path to the saved Excel file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = to_dataframe(rows)

    dataframe.to_excel(
        path,
        index=index,
    )

    return path


def save_config_snapshot(
    config: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save the configuration used during an analysis.

    This improves reproducibility by storing the exact YAML settings used for
    a given analysis run.

    Parameters
    ----------
    config:
        Configuration dictionary.

    output_path:
        Destination YAML path.

    Returns
    -------
    Path
        Path to the saved configuration file.
    """
    path = Path(output_path)
    save_yaml(config, path)
    return path


def add_subject_metadata(
    result_row: dict[str, Any],
    subject_id: str,
    group: str,
    experiment: str,
) -> dict[str, Any]:
    """
    Add standard subject metadata to an analysis result row.

    Parameters
    ----------
    result_row:
        Analysis result dictionary.

    subject_id:
        Participant identifier.

    group:
        Group name.

    experiment:
        Experiment name.

    Returns
    -------
    dict[str, Any]
        Result row with metadata fields added at the beginning.
    """
    metadata = {
        "subject_id": subject_id,
        "group": group,
        "experiment": experiment,
    }

    return {
        **metadata,
        **result_row,
    }


def merge_result_dicts(
    *result_dicts: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge multiple result dictionaries into a single row.

    Later dictionaries override earlier dictionaries if duplicate keys exist.

    Parameters
    ----------
    *result_dicts:
        Result dictionaries to merge.

    Returns
    -------
    dict[str, Any]
        Merged result dictionary.
    """
    merged: dict[str, Any] = {}

    for result_dict in result_dicts:
        merged.update(result_dict)

    return merged