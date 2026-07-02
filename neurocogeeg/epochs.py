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

def deduplicate_close_repeated_events(
    events: np.ndarray,
    sfreq: float,
    duplicate_window_ms: float,
) -> tuple[np.ndarray, int]:
    """
    Remove repeated markers with the same event code within a short time window.

    If the same event code occurs repeatedly within duplicate_window_ms,
    the first marker is kept and later repetitions are ignored.

    Different event codes are not merged, even if they occur within the same
    time window.

    Parameters
    ----------
    events:
        MNE events array with shape (n_events, 3).

    sfreq:
        Sampling frequency in Hz.

    duplicate_window_ms:
        Duplicate detection window in milliseconds.

    Returns
    -------
    tuple[np.ndarray, int]
        Deduplicated events array and number of dropped duplicate events.
    """
    if events is None or len(events) == 0:
        return np.empty((0, 3), dtype=int), 0

    if duplicate_window_ms <= 0:
        return events.astype(int), 0

    duplicate_window_samples = int(
        round((duplicate_window_ms / 1000.0) * float(sfreq))
    )

    kept_events = []
    dropped_count = 0
    last_kept_sample_by_code: dict[int, int] = {}

    for event in events:
        sample = int(event[0])
        event_code = int(event[2])

        last_sample = last_kept_sample_by_code.get(event_code)

        if last_sample is not None:
            if sample - last_sample <= duplicate_window_samples:
                dropped_count += 1
                continue

        kept_events.append(event)
        last_kept_sample_by_code[event_code] = sample

    if not kept_events:
        return np.empty((0, 3), dtype=int), dropped_count

    return np.asarray(kept_events, dtype=int), dropped_count

def deduplicate_close_repeated_events(
    events: np.ndarray,
    sfreq: float,
    duplicate_window_ms: float,
) -> tuple[np.ndarray, int]:
    """
    Remove repeated markers with the same event code within a short time window.

    If the same event code occurs repeatedly within duplicate_window_ms,
    the first marker is kept and later repetitions are ignored.

    Different event codes are not merged, even if they occur within the same
    time window.
    """
    if events is None or len(events) == 0:
        return np.empty((0, 3), dtype=int), 0

    if duplicate_window_ms <= 0:
        return events.astype(int), 0

    duplicate_window_samples = int(
        round((duplicate_window_ms / 1000.0) * float(sfreq))
    )

    kept_events = []
    dropped_count = 0
    last_kept_sample_by_code: dict[int, int] = {}

    for event in events:
        sample = int(event[0])
        event_code = int(event[2])

        last_sample = last_kept_sample_by_code.get(event_code)

        if last_sample is not None:
            if sample - last_sample <= duplicate_window_samples:
                dropped_count += 1
                continue

        kept_events.append(event)
        last_kept_sample_by_code[event_code] = sample

    if not kept_events:
        return np.empty((0, 3), dtype=int), dropped_count

    return np.asarray(kept_events, dtype=int), dropped_count

def extract_events_from_marker_channel(
    raw: mne.io.BaseRaw,
    marker_channel: str,
    marker_config: dict[str, Any],
) -> np.ndarray:
    """
    Extract MNE-compatible events from a marker channel.

    Repeated markers are deduplicated centrally. If the same marker code is
    emitted more than once within duplicate_window_ms, the first marker is kept
    and later repetitions are ignored.

    Different event codes are not merged.
    """
    if marker_channel not in raw.ch_names:
        raise ValueError(f"Marker channel not found in raw: {marker_channel}")

    scale_factor = marker_config.get("scale_factor", 1)
    zero_value = marker_config.get("zero_value", 0)
    duplicate_window_ms = marker_config.get("duplicate_window_ms", 100)

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
    events = events.astype(int)

    events, _dropped_duplicate_count = deduplicate_close_repeated_events(
        events=events,
        sfreq=float(raw.info["sfreq"]),
        duplicate_window_ms=float(duplicate_window_ms),
    )

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

#    epochs.pick(picks=picks)

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