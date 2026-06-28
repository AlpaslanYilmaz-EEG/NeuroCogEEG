"""
Epoch creation utilities for NeuroCogEEG.

This module contains reusable functions for extracting event markers from
EDF marker channels and creating MNE Epochs objects.

Experiment-specific event codes, epoch windows and baseline intervals must
be provided through YAML configuration files.
"""

from __future__ import annotations

from typing import Any

import mne
import numpy as np


def extract_events_from_marker_channel(
    raw: mne.io.BaseRaw,
    marker_channel: str,
    marker_config: dict[str, Any],
) -> np.ndarray:
    """
    Extract MNE-compatible events from a marker channel.

    Parameters
    ----------
    raw:
        Raw object containing EEG channels and the marker channel.

    marker_channel:
        Name of the marker channel.

    marker_config:
        Marker configuration from device YAML. Expected keys include
        ``scale_factor`` and ``zero_value``.

    Returns
    -------
    np.ndarray
        MNE events array with shape ``(n_events, 3)``.
    """
    if marker_channel not in raw.ch_names:
        raise ValueError(f"Marker channel not found in raw: {marker_channel}")

    scale_factor = marker_config.get("scale_factor", 1)
    zero_value = marker_config.get("zero_value", 0)

    marker_data = raw.copy().pick(
        picks=[marker_channel],
        verbose=False,
    ).get_data()[0]

    marker_values = np.round(marker_data * scale_factor).astype(int)

    marker_changes = np.diff(marker_values)
    event_indices = np.where(np.abs(marker_changes) > 0)[0] + 1

    events = np.column_stack(
        (
            event_indices,
            np.zeros_like(event_indices),
            marker_values[event_indices],
        )
    )

    events = events[events[:, 2] != zero_value]

    return events.astype(int)


def make_event_id(events_config: dict[str, int]) -> dict[str, int]:
    """
    Convert event configuration into an MNE event_id dictionary.

    Parameters
    ----------
    events_config:
        Event configuration from an experiment YAML file.

    Returns
    -------
    dict[str, int]
        MNE-compatible event_id dictionary.
    """
    return {
        event_name: int(event_code)
        for event_name, event_code in events_config.items()
    }


def create_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    event_id: dict[str, int],
    epoch_config: dict[str, Any],
    reject_criteria: dict[str, float] | None = None,
    picks: str | list[str] = "eeg",
) -> mne.Epochs:
    """
    Create MNE Epochs from raw data and events.

    Parameters
    ----------
    raw:
        Raw EEG object.

    events:
        MNE events array.

    event_id:
        MNE event_id dictionary.

    epoch_config:
        Epoch configuration from YAML. Expected keys are ``tmin``, ``tmax``
        and optionally ``baseline``.

    reject_criteria:
        Optional MNE rejection criteria.

    picks:
        Channels to include in epochs. Default is ``"eeg"``.

    Returns
    -------
    mne.Epochs
        Epochs object.
    """
    baseline_config = epoch_config.get("baseline", None)

    if baseline_config is None:
        baseline = None
    else:
        baseline = (
            baseline_config["start"],
            baseline_config["end"],
        )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=epoch_config["tmin"],
        tmax=epoch_config["tmax"],
        baseline=baseline,
        reject=reject_criteria,
        preload=True,
        on_missing="ignore",
        verbose=False,
    )

    epochs.pick(picks=picks)

    return epochs


def get_reject_criteria(
    quality_control_config: dict[str, Any],
) -> dict[str, float] | None:
    """
    Convert YAML rejection criteria from microvolts to volts.

    Parameters
    ----------
    quality_control_config:
        Quality-control section from experiment YAML.

    Returns
    -------
    dict[str, float] | None
        MNE-compatible rejection criteria.
    """
    reject_config = quality_control_config.get("reject_criteria", None)

    if reject_config is None:
        return None

    reject_criteria: dict[str, float] = {}

    if "eeg_uv" in reject_config:
        reject_criteria["eeg"] = reject_config["eeg_uv"] * 1e-6

    return reject_criteria


def filter_event_sequence(
    events: np.ndarray,
    first_event_code: int,
    next_event_code: int,
) -> np.ndarray:
    """
    Select events where a given event is followed by another given event.

    This is useful for selecting stimulus events that are followed by correct
    responses, as in Flanker and Go/No-Go analyses.

    Parameters
    ----------
    events:
        MNE events array.

    first_event_code:
        Event code that must occur first.

    next_event_code:
        Event code that must immediately follow the first event.

    Returns
    -------
    np.ndarray
        Filtered events array containing only the first events.
    """
    selected_events = []

    for index in range(len(events) - 1):
        current_event = events[index]
        next_event = events[index + 1]

        if (
            current_event[2] == first_event_code
            and next_event[2] == next_event_code
        ):
            selected_events.append(current_event)

    if not selected_events:
        return np.empty((0, 3), dtype=int)

    return np.asarray(selected_events, dtype=int)