"""
Quality-control utilities for NeuroCogEEG.

This module provides helper functions for summarizing preprocessing, event,
epoch and trial-count quality information.

It does not perform EEG analysis.
"""

from __future__ import annotations

from typing import Any

import mne
import numpy as np


def summarize_event_counts(
    events: np.ndarray,
    event_id: dict[str, int],
) -> dict[str, int]:
    """
    Count events according to an event_id dictionary.

    Parameters
    ----------
    events:
        MNE events array.

    event_id:
        Dictionary mapping event names to numeric event codes.

    Returns
    -------
    dict[str, int]
        Event count summary.
    """
    summary: dict[str, int] = {}

    for event_name, event_code in event_id.items():
        summary[f"event_count_{event_name}"] = int(
            np.sum(events[:, 2] == event_code)
        )

    summary["event_count_total"] = int(events.shape[0])

    return summary


def summarize_epochs(
    epochs: mne.Epochs,
    prefix: str = "epochs",
) -> dict[str, int | float]:
    """
    Summarize epoch counts and drop percentage.

    MNE stores non-selected events as ``IGNORED`` in ``epochs.drop_log``.
    These ignored events should not be counted as dropped epochs.

    Parameters
    ----------
    epochs:
        MNE Epochs object.

    prefix:
        Prefix used for output variable names.

    Returns
    -------
    dict[str, int | float]
        Epoch quality summary.
    """
    kept_count = len(epochs)

    ignored_count = 0
    dropped_count = 0

    for drop_entry in epochs.drop_log:
        if len(drop_entry) == 0:
            continue

        if "IGNORED" in drop_entry:
            ignored_count += 1
        else:
            dropped_count += 1

    selected_total = kept_count + dropped_count

    if selected_total == 0:
        drop_percent = float("nan")
    else:
        drop_percent = float((dropped_count / selected_total) * 100.0)

    return {
        f"{prefix}_total": int(selected_total),
        f"{prefix}_kept": int(kept_count),
        f"{prefix}_dropped": int(dropped_count),
        f"{prefix}_ignored": int(ignored_count),
        f"{prefix}_drop_percent": drop_percent,
    }


def summarize_epochs_by_condition(
    epochs: mne.Epochs,
    prefix: str = "condition",
) -> dict[str, int]:
    """
    Count kept epochs for each condition.

    Parameters
    ----------
    epochs:
        MNE Epochs object.

    prefix:
        Prefix used for output variable names.

    Returns
    -------
    dict[str, int]
        Condition-wise epoch counts.
    """
    summary: dict[str, int] = {}

    for condition_name in epochs.event_id:
        summary[f"{prefix}_{condition_name}_kept"] = int(
            len(epochs[condition_name])
        )

    return summary


def check_minimum_count(
    count: int,
    minimum: int,
) -> bool:
    """
    Check whether a count satisfies a minimum threshold.

    Parameters
    ----------
    count:
        Observed count.

    minimum:
        Required minimum count.

    Returns
    -------
    bool
        ``True`` if count is greater than or equal to minimum.
    """
    return int(count) >= int(minimum)


def summarize_minimum_trial_requirement(
    observed_count: int,
    minimum_count: int,
    label: str,
) -> dict[str, int | bool]:
    """
    Summarize whether a minimum trial-count requirement is met.

    Parameters
    ----------
    observed_count:
        Observed trial count.

    minimum_count:
        Required minimum trial count.

    label:
        Label used in output variable names.

    Returns
    -------
    dict[str, int | bool]
        Minimum-count QC summary.
    """
    return {
        f"{label}_observed_count": int(observed_count),
        f"{label}_minimum_required": int(minimum_count),
        f"{label}_minimum_met": check_minimum_count(
            count=observed_count,
            minimum=minimum_count,
        ),
    }


def summarize_preprocessing_info(
    preprocessing_info: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract compact QC information from preprocessing output.

    Parameters
    ----------
    preprocessing_info:
        Dictionary returned by preprocessing functions.

    Returns
    -------
    dict[str, Any]
        Compact preprocessing QC summary.
    """
    excluded_components = preprocessing_info.get(
        "ica_excluded_components",
        [],
    )

    return {
        "qc_marker_channel": preprocessing_info.get("marker_channel"),
        "qc_ica_enabled": preprocessing_info.get("ica_enabled"),
        "qc_ica_excluded_count": len(excluded_components),
        "qc_ica_excluded_components": ",".join(
            str(component) for component in excluded_components
        ),
    }


def summarize_raw_info(
    raw: mne.io.BaseRaw,
    prefix: str = "raw",
) -> dict[str, int | float]:
    """
    Summarize basic Raw object information.

    Parameters
    ----------
    raw:
        MNE Raw object.

    prefix:
        Prefix used for output variable names.

    Returns
    -------
    dict[str, int | float]
        Raw data summary.
    """
    duration_seconds = raw.n_times / raw.info["sfreq"]

    return {
        f"{prefix}_n_channels": int(len(raw.ch_names)),
        f"{prefix}_n_samples": int(raw.n_times),
        f"{prefix}_sfreq": float(raw.info["sfreq"]),
        f"{prefix}_duration_s": float(duration_seconds),
    }


def merge_qc_summaries(
    *summaries: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge multiple QC dictionaries.

    Later dictionaries override earlier dictionaries if keys overlap.

    Parameters
    ----------
    *summaries:
        QC dictionaries.

    Returns
    -------
    dict[str, Any]
        Merged QC dictionary.
    """
    merged: dict[str, Any] = {}

    for summary in summaries:
        merged.update(summary)

    return merged


def qc_status_from_boolean(
    value: bool,
) -> str:
    """
    Convert a boolean QC result into a readable status string.

    Parameters
    ----------
    value:
        Boolean QC value.

    Returns
    -------
    str
        ``"PASS"`` or ``"FAIL"``.
    """
    return "PASS" if value else "FAIL"