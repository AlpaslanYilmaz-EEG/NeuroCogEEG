"""
Response-locked potential analysis utilities for NeuroCogEEG.

This module provides generic functions for extracting response-locked
movement-related potentials, including Readiness Potential (RP) and
Post-Movement Positivity (PMP).

Important
---------
RP and PMP time windows are experiment-specific. Therefore, they must be
provided through the corresponding experiment YAML file.

This module must not contain hard-coded RP or PMP time windows.
"""

from __future__ import annotations

from typing import Literal

import mne
import numpy as np

from neurocogeeg.constants import V_TO_UV


PeakPolarity = Literal["negative", "positive"]


def get_time_window(
    window_config: dict[str, float],
) -> tuple[float, float]:
    """
    Convert a YAML window configuration into a time-window tuple.

    Parameters
    ----------
    window_config:
        Dictionary containing ``start`` and ``end`` values in seconds.

    Returns
    -------
    tuple[float, float]
        Start and end time in seconds.
    """
    return (
        float(window_config["start"]),
        float(window_config["end"]),
    )


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


def extract_roi_waveform(
    evoked: mne.Evoked,
    roi_channels: list[str],
    window_config: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract ROI-averaged waveform within a time window.

    Parameters
    ----------
    evoked:
        MNE Evoked object.

    roi_channels:
        ROI channel names.

    window_config:
        Time-window configuration containing ``start`` and ``end`` in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Time vector in seconds and ROI-averaged waveform in volts.
    """
    tmin, tmax = get_time_window(window_config)

    existing_channels = get_existing_channels(
        available_channels=evoked.ch_names,
        requested_channels=roi_channels,
    )

    evoked_roi = (
        evoked.copy()
        .pick(picks=existing_channels)
        .crop(tmin=tmin, tmax=tmax)
    )

    roi_waveform = np.mean(evoked_roi.data, axis=0)

    return evoked_roi.times, roi_waveform


def compute_mean_uv(
    evoked: mne.Evoked,
    roi_channels: list[str],
    window_config: dict[str, float],
) -> float:
    """
    Compute mean amplitude in microvolts.

    Parameters
    ----------
    evoked:
        MNE Evoked object.

    roi_channels:
        ROI channel names.

    window_config:
        Time-window configuration.

    Returns
    -------
    float
        Mean amplitude in microvolts.
    """
    _, waveform = extract_roi_waveform(
        evoked=evoked,
        roi_channels=roi_channels,
        window_config=window_config,
    )

    return float(np.mean(waveform) * V_TO_UV)


def compute_peak_uv(
    evoked: mne.Evoked,
    roi_channels: list[str],
    window_config: dict[str, float],
    polarity: PeakPolarity,
) -> float:
    """
    Compute peak amplitude in microvolts.

    Parameters
    ----------
    evoked:
        MNE Evoked object.

    roi_channels:
        ROI channel names.

    window_config:
        Time-window configuration.

    polarity:
        ``"negative"`` for negative peak or ``"positive"`` for positive peak.

    Returns
    -------
    float
        Peak amplitude in microvolts.
    """
    _, waveform = extract_roi_waveform(
        evoked=evoked,
        roi_channels=roi_channels,
        window_config=window_config,
    )

    if polarity == "negative":
        return float(np.min(waveform) * V_TO_UV)

    if polarity == "positive":
        return float(np.max(waveform) * V_TO_UV)

    raise ValueError(f"Unsupported polarity: {polarity!r}")


def compute_area_uv_s(
    evoked: mne.Evoked,
    roi_channels: list[str],
    window_config: dict[str, float],
) -> float:
    """
    Compute waveform area in microvolt-seconds.

    The area is calculated using trapezoidal integration over the
    ROI-averaged waveform.

    Parameters
    ----------
    evoked:
        MNE Evoked object.

    roi_channels:
        ROI channel names.

    window_config:
        Time-window configuration.

    Returns
    -------
    float
        Area in microvolt-seconds.
    """
    times, waveform = extract_roi_waveform(
        evoked=evoked,
        roi_channels=roi_channels,
        window_config=window_config,
    )

    waveform_uv = waveform * V_TO_UV

    return float(np.trapz(waveform_uv, times))


def compute_slope_uv_per_s(
    evoked: mne.Evoked,
    roi_channels: list[str],
    window_config: dict[str, float],
) -> float:
    """
    Compute linear slope of ROI waveform in microvolts per second.

    Parameters
    ----------
    evoked:
        MNE Evoked object.

    roi_channels:
        ROI channel names.

    window_config:
        Time-window configuration.

    Returns
    -------
    float
        Linear slope in microvolts per second.
    """
    times, waveform = extract_roi_waveform(
        evoked=evoked,
        roi_channels=roi_channels,
        window_config=window_config,
    )

    if times.size < 2:
        return float("nan")

    waveform_uv = waveform * V_TO_UV

    slope, _intercept = np.polyfit(times, waveform_uv, deg=1)

    return float(slope)


def compute_rp_metrics_from_evoked(
    evoked: mne.Evoked,
    roi_channels: list[str],
    rp_window_config: dict[str, float],
    prefix: str = "rp",
    peak_polarity: PeakPolarity = "negative",
) -> dict[str, float]:
    """
    Compute RP metrics from a response-locked Evoked object.

    Parameters
    ----------
    evoked:
        Response-locked MNE Evoked object.

    roi_channels:
        ROI channel names.

    rp_window_config:
        RP time-window configuration from YAML.

    prefix:
        Prefix used for output variable names.

    peak_polarity:
        Peak polarity for RP. Default is ``"negative"``.

    Returns
    -------
    dict[str, float]
        RP metric dictionary.
    """
    return {
        f"{prefix}_mean_uv": compute_mean_uv(
            evoked=evoked,
            roi_channels=roi_channels,
            window_config=rp_window_config,
        ),
        f"{prefix}_peak_uv": compute_peak_uv(
            evoked=evoked,
            roi_channels=roi_channels,
            window_config=rp_window_config,
            polarity=peak_polarity,
        ),
        f"{prefix}_area_uv_s": compute_area_uv_s(
            evoked=evoked,
            roi_channels=roi_channels,
            window_config=rp_window_config,
        ),
        f"{prefix}_slope_uv_per_s": compute_slope_uv_per_s(
            evoked=evoked,
            roi_channels=roi_channels,
            window_config=rp_window_config,
        ),
    }


def compute_pmp_metrics_from_evoked(
    evoked: mne.Evoked,
    roi_channels: list[str],
    pmp_window_config: dict[str, float],
    prefix: str = "pmp",
    peak_polarity: PeakPolarity = "positive",
) -> dict[str, float]:
    """
    Compute Post-Movement Positivity metrics from an Evoked object.

    Parameters
    ----------
    evoked:
        Response-locked MNE Evoked object.

    roi_channels:
        ROI channel names.

    pmp_window_config:
        PMP time-window configuration from YAML.

    prefix:
        Prefix used for output variable names.

    peak_polarity:
        Peak polarity for PMP. Default is ``"positive"``.

    Returns
    -------
    dict[str, float]
        PMP metric dictionary.
    """
    return {
        f"{prefix}_mean_uv": compute_mean_uv(
            evoked=evoked,
            roi_channels=roi_channels,
            window_config=pmp_window_config,
        ),
        f"{prefix}_peak_uv": compute_peak_uv(
            evoked=evoked,
            roi_channels=roi_channels,
            window_config=pmp_window_config,
            polarity=peak_polarity,
        ),
        f"{prefix}_area_uv_s": compute_area_uv_s(
            evoked=evoked,
            roi_channels=roi_channels,
            window_config=pmp_window_config,
        ),
    }


def compute_response_locked_metrics_from_evoked(
    evoked: mne.Evoked,
    roi_channels: list[str],
    rp_window_config: dict[str, float],
    pmp_window_config: dict[str, float],
) -> dict[str, float]:
    """
    Compute both RP and PMP metrics from a response-locked Evoked object.

    Parameters
    ----------
    evoked:
        Response-locked MNE Evoked object.

    roi_channels:
        ROI channel names.

    rp_window_config:
        RP time-window configuration from YAML.

    pmp_window_config:
        PMP time-window configuration from YAML.

    Returns
    -------
    dict[str, float]
        Combined RP and PMP metric dictionary.
    """
    rp_metrics = compute_rp_metrics_from_evoked(
        evoked=evoked,
        roi_channels=roi_channels,
        rp_window_config=rp_window_config,
    )

    pmp_metrics = compute_pmp_metrics_from_evoked(
        evoked=evoked,
        roi_channels=roi_channels,
        pmp_window_config=pmp_window_config,
    )

    return {
        **rp_metrics,
        **pmp_metrics,
    }


def compute_response_locked_metrics_from_epochs(
    epochs: mne.Epochs,
    roi_channels: list[str],
    rp_window_config: dict[str, float],
    pmp_window_config: dict[str, float],
    condition: str | None = None,
) -> dict[str, float]:
    """
    Compute RP and PMP metrics from response-locked epochs.

    Parameters
    ----------
    epochs:
        Response-locked MNE Epochs object.

    roi_channels:
        ROI channel names.

    rp_window_config:
        RP time-window configuration from YAML.

    pmp_window_config:
        PMP time-window configuration from YAML.

    condition:
        Optional condition name. If provided, metrics are computed only for
        that condition. If ``None``, all epochs are averaged.

    Returns
    -------
    dict[str, float]
        RP and PMP metric dictionary including epoch count.
    """
    if condition is not None:
        if condition not in epochs.event_id:
            return {
                "rp_trial_count": 0,
                "rp_mean_uv": np.nan,
                "rp_peak_uv": np.nan,
                "rp_area_uv_s": np.nan,
                "rp_slope_uv_per_s": np.nan,
                "pmp_mean_uv": np.nan,
                "pmp_peak_uv": np.nan,
                "pmp_area_uv_s": np.nan,
            }

        selected_epochs = epochs[condition]
    else:
        selected_epochs = epochs

    trial_count = len(selected_epochs)

    if trial_count == 0:
        return {
            "rp_trial_count": 0,
            "rp_mean_uv": np.nan,
            "rp_peak_uv": np.nan,
            "rp_area_uv_s": np.nan,
            "rp_slope_uv_per_s": np.nan,
            "pmp_mean_uv": np.nan,
            "pmp_peak_uv": np.nan,
            "pmp_area_uv_s": np.nan,
        }

    evoked = selected_epochs.average()

    metrics = compute_response_locked_metrics_from_evoked(
        evoked=evoked,
        roi_channels=roi_channels,
        rp_window_config=rp_window_config,
        pmp_window_config=pmp_window_config,
    )

    return {
        "rp_trial_count": int(trial_count),
        **metrics,
    }