"""
Behavioral analysis utilities for NeuroCogEEG.

This module contains reusable helper functions for extracting behavioral
measures from MNE-style event arrays.

Important
---------
Numeric event codes do not have global meaning in NeuroCogEEG.

The same numeric marker value may represent different events in different
experiments. Therefore, event codes must always be provided by the
experiment-specific YAML configuration file.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def get_event_code(
    events_config: dict[str, int],
    event_name: str,
) -> int:
    """
    Get an event code from an experiment configuration.

    Parameters
    ----------
    events_config:
        Event section from an experiment YAML file.

    event_name:
        Name of the event, for example ``stimulus`` or ``correct_response``.

    Returns
    -------
    int
        Numeric event code.

    Raises
    ------
    KeyError
        If the requested event name is not present in the configuration.
    """
    if event_name not in events_config:
        available = ", ".join(events_config.keys())
        raise KeyError(
            f"Event name not found: {event_name!r}. "
            f"Available events are: {available}"
        )

    return int(events_config[event_name])


def count_event(
    events: np.ndarray,
    event_code: int,
) -> int:
    """
    Count how many times a specific event code occurs.

    Parameters
    ----------
    events:
        MNE events array with shape ``(n_events, 3)``.

    event_code:
        Numeric event code to count.

    Returns
    -------
    int
        Number of occurrences.
    """
    return int(np.sum(events[:, 2] == event_code))


def count_events_from_config(
    events: np.ndarray,
    events_config: dict[str, int],
) -> dict[str, int]:
    """
    Count all events defined in an experiment configuration.

    Parameters
    ----------
    events:
        MNE events array.

    events_config:
        Event section from an experiment YAML file.

    Returns
    -------
    dict[str, int]
        Dictionary mapping event names to event counts.
    """
    counts: dict[str, int] = {}

    for event_name, event_code in events_config.items():
        counts[event_name] = count_event(events, int(event_code))

    return counts


def event_samples(
    events: np.ndarray,
    event_code: int,
) -> np.ndarray:
    """
    Return sample indices for a given event code.

    Parameters
    ----------
    events:
        MNE events array.

    event_code:
        Numeric event code.

    Returns
    -------
    np.ndarray
        Sample indices of matching events.
    """
    return events[events[:, 2] == event_code, 0]


def event_times_seconds(
    events: np.ndarray,
    event_code: int,
    sfreq: float,
) -> np.ndarray:
    """
    Return event times in seconds for a given event code.

    Parameters
    ----------
    events:
        MNE events array.

    event_code:
        Numeric event code.

    sfreq:
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Event times in seconds.
    """
    samples = event_samples(events, event_code)
    return samples / sfreq


def compute_transition_latencies_ms(
    events: np.ndarray,
    start_code: int,
    end_code: int,
    sfreq: float,
    min_latency_ms: float | None = None,
    max_latency_ms: float | None = None,
) -> list[float]:
    """
    Compute latencies between adjacent start and end events.

    This is useful for reaction time calculations, for example:

    stimulus -> correct_response

    The function only considers immediately adjacent event pairs.

    Parameters
    ----------
    events:
        MNE events array.

    start_code:
        Event code that starts the interval.

    end_code:
        Event code that ends the interval.

    sfreq:
        Sampling frequency in Hz.

    min_latency_ms:
        Optional minimum accepted latency in milliseconds.

    max_latency_ms:
        Optional maximum accepted latency in milliseconds.

    Returns
    -------
    list[float]
        Latencies in milliseconds.
    """
    latencies: list[float] = []

    for index in range(len(events) - 1):
        current_event = events[index]
        next_event = events[index + 1]

        if current_event[2] == start_code and next_event[2] == end_code:
            latency_ms = (
                (next_event[0] - current_event[0]) / sfreq
            ) * 1000.0

            if min_latency_ms is not None and latency_ms < min_latency_ms:
                continue

            if max_latency_ms is not None and latency_ms > max_latency_ms:
                continue

            latencies.append(float(latency_ms))

    return latencies


def summarize_latencies(
    latencies_ms: list[float],
    prefix: str,
) -> dict[str, float]:
    """
    Summarize a list of latencies.

    Parameters
    ----------
    latencies_ms:
        Latencies in milliseconds.

    prefix:
        Prefix used for output variable names.

    Returns
    -------
    dict[str, float]
        Summary statistics.
    """
    if not latencies_ms:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean_ms": np.nan,
            f"{prefix}_median_ms": np.nan,
            f"{prefix}_sd_ms": np.nan,
            f"{prefix}_min_ms": np.nan,
            f"{prefix}_max_ms": np.nan,
        }

    values = np.asarray(latencies_ms, dtype=float)

    return {
        f"{prefix}_n": int(values.size),
        f"{prefix}_mean_ms": float(np.mean(values)),
        f"{prefix}_median_ms": float(np.median(values)),
        f"{prefix}_sd_ms": float(np.std(values, ddof=1))
        if values.size > 1
        else 0.0,
        f"{prefix}_min_ms": float(np.min(values)),
        f"{prefix}_max_ms": float(np.max(values)),
    }


def compute_accuracy_percent(
    correct_count: int,
    total_count: int,
) -> float:
    """
    Compute accuracy percentage.

    Parameters
    ----------
    correct_count:
        Number of correct responses.

    total_count:
        Total number of relevant trials or stimuli.

    Returns
    -------
    float
        Accuracy percentage. Returns NaN if ``total_count`` is zero.
    """
    if total_count == 0:
        return float("nan")

    return float((correct_count / total_count) * 100.0)


def compute_task_duration_seconds(
    events: np.ndarray,
    start_code: int,
    end_codes: Iterable[int],
    sfreq: float,
) -> float:
    """
    Compute task duration from the first start event to the last end event.

    Parameters
    ----------
    events:
        MNE events array.

    start_code:
        Event code marking task start.

    end_codes:
        Event codes that may mark task progress or completion.

    sfreq:
        Sampling frequency in Hz.

    Returns
    -------
    float
        Task duration in seconds. Returns NaN if required events are missing.
    """
    start_samples = event_samples(events, start_code)

    if start_samples.size == 0:
        return float("nan")

    end_codes_set = set(int(code) for code in end_codes)
    end_mask = np.isin(events[:, 2], list(end_codes_set))
    end_samples = events[end_mask, 0]

    if end_samples.size == 0:
        return float("nan")

    start_sample = start_samples[0]
    end_sample = end_samples[-1]

    if end_sample <= start_sample:
        return float("nan")

    return float((end_sample - start_sample) / sfreq)


def compute_inter_event_intervals_seconds(
    events: np.ndarray,
    event_codes: Iterable[int],
    sfreq: float,
) -> list[float]:
    """
    Compute intervals between consecutive selected events.

    This is useful for digital TMT click-to-click interval calculations.

    Parameters
    ----------
    events:
        MNE events array.

    event_codes:
        Event codes to include.

    sfreq:
        Sampling frequency in Hz.

    Returns
    -------
    list[float]
        Inter-event intervals in seconds.
    """
    event_codes_set = set(int(code) for code in event_codes)
    selected_mask = np.isin(events[:, 2], list(event_codes_set))
    selected_samples = events[selected_mask, 0]

    if selected_samples.size < 2:
        return []

    intervals = np.diff(selected_samples) / sfreq

    return [float(value) for value in intervals]


def summarize_intervals_seconds(
    intervals_seconds: list[float],
    prefix: str,
) -> dict[str, float]:
    """
    Summarize intervals measured in seconds.

    Parameters
    ----------
    intervals_seconds:
        Interval values in seconds.

    prefix:
        Prefix used for output variable names.

    Returns
    -------
    dict[str, float]
        Summary statistics.
    """
    if not intervals_seconds:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean_s": np.nan,
            f"{prefix}_median_s": np.nan,
            f"{prefix}_sd_s": np.nan,
            f"{prefix}_min_s": np.nan,
            f"{prefix}_max_s": np.nan,
        }

    values = np.asarray(intervals_seconds, dtype=float)

    return {
        f"{prefix}_n": int(values.size),
        f"{prefix}_mean_s": float(np.mean(values)),
        f"{prefix}_median_s": float(np.median(values)),
        f"{prefix}_sd_s": float(np.std(values, ddof=1))
        if values.size > 1
        else 0.0,
        f"{prefix}_min_s": float(np.min(values)),
        f"{prefix}_max_s": float(np.max(values)),
    }


def split_events_by_time(
    events: np.ndarray,
    split_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split events into early and late parts using a sample index.

    Parameters
    ----------
    events:
        MNE events array.

    split_sample:
        Sample index used as split point.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Early events and late events.
    """
    early_events = events[events[:, 0] < split_sample]
    late_events = events[events[:, 0] >= split_sample]

    return early_events, late_events
