"""
Plotting utilities for NeuroCogEEG.

This module provides reusable plotting functions for ERP, RP and other
time-locked EEG waveforms.

The functions save figures to disk and do not perform EEG analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from neurocogeeg.constants import V_TO_UV


def get_existing_channels(
    available_channels: list[str],
    requested_channels: list[str],
) -> list[str]:
    """
    Return requested channels that exist in the current data.

    Parameters
    ----------
    available_channels:
        Channel names available in the Evoked object.

    requested_channels:
        ROI channel names requested by YAML configuration.

    Returns
    -------
    list[str]
        Existing requested channels.

    Raises
    ------
    ValueError
        If none of the requested channels are available.
    """
    existing_channels = [
        channel
        for channel in requested_channels
        if channel in available_channels
    ]

    if not existing_channels:
        raise ValueError(
            "None of the requested ROI channels were found. "
            f"Requested: {requested_channels}. "
            f"Available: {available_channels}"
        )

    return existing_channels


def extract_roi_waveform_uv(
    evoked: mne.Evoked,
    roi_channels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract ROI-averaged evoked waveform in microvolts.

    Parameters
    ----------
    evoked:
        MNE Evoked object.

    roi_channels:
        ROI channel names.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Time vector in seconds and ROI-averaged waveform in microvolts.
    """
    existing_channels = get_existing_channels(
        available_channels=evoked.ch_names,
        requested_channels=roi_channels,
    )

    evoked_roi = evoked.copy().pick(picks=existing_channels)

    waveform_uv = np.mean(evoked_roi.data, axis=0) * V_TO_UV

    return evoked_roi.times, waveform_uv


def save_roi_evoked_plot(
    evoked: mne.Evoked,
    roi_channels: list[str],
    output_path: str | Path,
    title: str,
    x_label: str = "Time (s)",
    y_label: str = "Amplitude (µV)",
    show_zero_lines: bool = True,
) -> Path:
    """
    Save an ROI-averaged Evoked waveform plot.

    Parameters
    ----------
    evoked:
        MNE Evoked object.

    roi_channels:
        ROI channel names.

    output_path:
        Destination figure path.

    title:
        Figure title.

    x_label:
        X-axis label.

    y_label:
        Y-axis label.

    show_zero_lines:
        Whether to draw zero-time and zero-amplitude reference lines.

    Returns
    -------
    Path
        Saved figure path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    times, waveform_uv = extract_roi_waveform_uv(
        evoked=evoked,
        roi_channels=roi_channels,
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(times, waveform_uv, label="ROI average")

    if show_zero_lines:
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.axhline(0.0, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


def save_group_evoked_comparison_plot(
    evoked_by_group: dict[str, mne.Evoked],
    roi_channels: list[str],
    output_path: str | Path,
    title: str,
    x_label: str = "Time (s)",
    y_label: str = "Amplitude (µV)",
    show_zero_lines: bool = True,
) -> Path:
    """
    Save ROI-averaged Evoked waveforms for multiple groups.

    Parameters
    ----------
    evoked_by_group:
        Dictionary mapping group names to Evoked objects.

    roi_channels:
        ROI channel names.

    output_path:
        Destination figure path.

    title:
        Figure title.

    x_label:
        X-axis label.

    y_label:
        Y-axis label.

    show_zero_lines:
        Whether to draw zero-time and zero-amplitude reference lines.

    Returns
    -------
    Path
        Saved figure path.
    """
    if not evoked_by_group:
        raise ValueError("evoked_by_group is empty.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    for group_name, evoked in evoked_by_group.items():
        times, waveform_uv = extract_roi_waveform_uv(
            evoked=evoked,
            roi_channels=roi_channels,
        )

        ax.plot(times, waveform_uv, label=group_name)

    if show_zero_lines:
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.axhline(0.0, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


def save_metric_bar_plot(
    values: dict[str, float],
    output_path: str | Path,
    title: str,
    y_label: str,
) -> Path:
    """
    Save a simple bar plot for scalar metrics.

    Parameters
    ----------
    values:
        Dictionary mapping labels to numeric values.

    output_path:
        Destination figure path.

    title:
        Figure title.

    y_label:
        Y-axis label.

    Returns
    -------
    Path
        Saved figure path.
    """
    if not values:
        raise ValueError("values is empty.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(values.keys())
    numeric_values = [values[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(labels, numeric_values)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path
