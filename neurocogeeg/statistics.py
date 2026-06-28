"""
Statistical helper utilities for NeuroCogEEG.

This module provides small reusable statistical helper functions used by
pipelines and reports.

It is not intended to replace full statistical analysis software such as SPSS,
JASP, Jamovi or R.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def nanmean(values: list[float] | np.ndarray) -> float:
    """
    Compute NaN-safe mean.

    Parameters
    ----------
    values:
        Numeric values.

    Returns
    -------
    float
        Mean value ignoring NaNs.
    """
    array = np.asarray(values, dtype=float)

    if array.size == 0:
        return float("nan")

    return float(np.nanmean(array))


def nanstd(values: list[float] | np.ndarray, ddof: int = 1) -> float:
    """
    Compute NaN-safe standard deviation.

    Parameters
    ----------
    values:
        Numeric values.

    ddof:
        Delta degrees of freedom.

    Returns
    -------
    float
        Standard deviation ignoring NaNs.
    """
    array = np.asarray(values, dtype=float)

    valid_count = np.sum(~np.isnan(array))

    if valid_count <= ddof:
        return float("nan")

    return float(np.nanstd(array, ddof=ddof))


def nansem(values: list[float] | np.ndarray) -> float:
    """
    Compute NaN-safe standard error of the mean.

    Parameters
    ----------
    values:
        Numeric values.

    Returns
    -------
    float
        Standard error of the mean ignoring NaNs.
    """
    array = np.asarray(values, dtype=float)
    valid_count = np.sum(~np.isnan(array))

    if valid_count <= 1:
        return float("nan")

    return float(nanstd(array, ddof=1) / np.sqrt(valid_count))


def summarize_numeric_values(
    values: list[float] | np.ndarray,
    prefix: str,
) -> dict[str, float | int]:
    """
    Summarize numeric values with NaN-safe descriptive statistics.

    Parameters
    ----------
    values:
        Numeric values.

    prefix:
        Prefix used for output variable names.

    Returns
    -------
    dict[str, float | int]
        Descriptive statistics.
    """
    array = np.asarray(values, dtype=float)
    valid_values = array[~np.isnan(array)]

    if valid_values.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_sd": np.nan,
            f"{prefix}_sem": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }

    return {
        f"{prefix}_n": int(valid_values.size),
        f"{prefix}_mean": float(np.mean(valid_values)),
        f"{prefix}_sd": nanstd(valid_values),
        f"{prefix}_sem": nansem(valid_values),
        f"{prefix}_median": float(np.median(valid_values)),
        f"{prefix}_min": float(np.min(valid_values)),
        f"{prefix}_max": float(np.max(valid_values)),
    }


def summarize_dataframe_by_group(
    dataframe: pd.DataFrame,
    group_column: str,
    value_columns: list[str],
) -> pd.DataFrame:
    """
    Compute descriptive statistics by group for selected columns.

    Parameters
    ----------
    dataframe:
        Input DataFrame.

    group_column:
        Name of the grouping column.

    value_columns:
        Numeric columns to summarize.

    Returns
    -------
    pd.DataFrame
        Group-level descriptive summary.
    """
    if group_column not in dataframe.columns:
        raise KeyError(f"Group column not found: {group_column}")

    missing_columns = [
        column for column in value_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(f"Value columns not found: {missing_columns}")

    rows: list[dict[str, Any]] = []

    for group_name, group_df in dataframe.groupby(group_column):
        row: dict[str, Any] = {
            group_column: group_name,
        }

        for column in value_columns:
            summary = summarize_numeric_values(
                group_df[column].to_numpy(dtype=float),
                prefix=column,
            )
            row.update(summary)

        rows.append(row)

    return pd.DataFrame(rows)


def compute_difference(
    value_a: float,
    value_b: float,
) -> float:
    """
    Compute simple difference between two values.

    Parameters
    ----------
    value_a:
        First value.

    value_b:
        Second value.

    Returns
    -------
    float
        ``value_a - value_b``.
    """
    if np.isnan(value_a) or np.isnan(value_b):
        return float("nan")

    return float(value_a - value_b)


def compute_change_score(
    early_value: float,
    late_value: float,
) -> float:
    """
    Compute late-minus-early change score.

    Parameters
    ----------
    early_value:
        Early-period value.

    late_value:
        Late-period value.

    Returns
    -------
    float
        ``late_value - early_value``.
    """
    if np.isnan(early_value) or np.isnan(late_value):
        return float("nan")

    return float(late_value - early_value)


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Divide two numbers safely.

    Parameters
    ----------
    numerator:
        Numerator.

    denominator:
        Denominator.

    Returns
    -------
    float
        Division result. Returns NaN if denominator is zero or NaN.
    """
    if denominator == 0 or np.isnan(denominator):
        return float("nan")

    return float(numerator / denominator)


def compute_percent_change(
    baseline_value: float,
    followup_value: float,
) -> float:
    """
    Compute percent change from baseline to follow-up.

    Parameters
    ----------
    baseline_value:
        Baseline value.

    followup_value:
        Follow-up value.

    Returns
    -------
    float
        Percent change.
    """
    difference = compute_difference(
        value_a=followup_value,
        value_b=baseline_value,
    )

    return safe_divide(difference, baseline_value) * 100.0