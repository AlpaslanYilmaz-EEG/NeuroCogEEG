"""
ERP component analysis utilities for NeuroCogEEG.

This module provides generic functions for extracting ERP component amplitudes
from MNE Evoked objects.

Important
---------
ERP time windows and ROI definitions are experiment-specific. Therefore, they
must be provided through the corresponding experiment YAML file.

This module must not contain hard-coded component windows such as N2 = 200-350 ms.
"""

from __future__ import annotations

from typing import Any, Literal

import mne
import numpy as np

from neurocogeeg.constants import V_TO_UV


AmplitudeMethod = Literal["mean", "peak"]
PeakPolarity = Literal["negative", "positive"]


def get_time_window(
    window_config: dict[str, float],
) -> tuple[float, float]:
    """
    Convert a YAML ERP window configuration into a time-window tuple.

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
        ROI channel names requested by the YAML configuration.

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


def extract_roi_data(
    evoked: mne.Evoked,
    roi_channels: list[str],
    window_config: dict[str, float],
) -> np.ndarray:
    """
    Extract ERP data from selected ROI channels and time window.

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
    np.ndarray
        Data array in volts with shape ``(n_channels, n_times)``.
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

    return evoked_roi.data


def compute_mean_amplitude_uv(
    evoked: mne.Evoked,
    roi_channels: list[str],
    window_config: dict[str, float],
) -> float:
    """
    Compute mean ERP amplitude in microvolts.

    The mean is computed across ROI channels and time points.

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
    data = extract_roi_data(
        evoked=evoked,
        roi_channels=roi_channels,
        window_config=window_config,
    )

    return float(np.mean(data) * V_TO_UV)


def compute_peak_amplitude_uv(
    evoked: mne.Evoked,
    roi_channels: list[str],
    window_config: dict[str, float],
    polarity: PeakPolarity,
) -> float:
    """
    Compute peak ERP amplitude in microvolts.

    First, channels in the ROI are averaged. Then the positive or negative peak
    is extracted from the ROI waveform.

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
    data = extract_roi_data(
        evoked=evoked,
        roi_channels=roi_channels,
        window_config=window_config,
    )

    roi_waveform = np.mean(data, axis=0)

    if polarity == "negative":
        return float(np.min(roi_waveform) * V_TO_UV)

    if polarity == "positive":
        return float(np.max(roi_waveform) * V_TO_UV)

    raise ValueError(f"Unsupported polarity: {polarity!r}")


def compute_component_amplitude_uv(
    evoked: mne.Evoked,
    component_name: str,
    window_config: dict[str, float],
    roi_channels: list[str],
    method: AmplitudeMethod = "mean",
    polarity: PeakPolarity | None = None,
) -> dict[str, float]:
    """
    Compute ERP component amplitude and return it as a result dictionary.

    Parameters
    ----------
    evoked:
        MNE Evoked object.

    component_name:
        Component name used in output variable names, for example ``"n2"``.

    window_config:
        Component time-window configuration from YAML.

    roi_channels:
        ROI channel names from YAML.

    method:
        Amplitude extraction method. Either ``"mean"`` or ``"peak"``.

    polarity:
        Required when ``method="peak"``. Use ``"negative"`` or ``"positive"``.

    Returns
    -------
    dict[str, float]
        Dictionary containing ERP amplitude in microvolts.
    """
    component_key = component_name.lower()

    if method == "mean":
        amplitude = compute_mean_amplitude_uv(
            evoked=evoked,
            roi_channels=roi_channels,
            window_config=window_config,
        )
    elif method == "peak":
        if polarity is None:
            raise ValueError("polarity must be provided when method='peak'.")

        amplitude = compute_peak_amplitude_uv(
            evoked=evoked,
            roi_channels=roi_channels,
            window_config=window_config,
            polarity=polarity,
        )
    else:
        raise ValueError(f"Unsupported amplitude method: {method!r}")

    return {
        f"{component_key}_amplitude_uv": amplitude,
    }


def compute_difference_evoked(
    evoked_a: mne.Evoked,
    evoked_b: mne.Evoked,
    weight_a: float = 1.0,
    weight_b: float = -1.0,
) -> mne.Evoked:
    """
    Compute a difference wave between two Evoked objects.

    This is useful for ERN-like analyses, for example:

    error response - correct response

    Parameters
    ----------
    evoked_a:
        First Evoked object.

    evoked_b:
        Second Evoked object.

    weight_a:
        Weight for the first Evoked object.

    weight_b:
        Weight for the second Evoked object.

    Returns
    -------
    mne.Evoked
        Combined Evoked object.
    """
    return mne.combine_evoked(
        [evoked_a, evoked_b],
        weights=[weight_a, weight_b],
    )


def compute_component_from_epochs(
    epochs: mne.Epochs,
    condition: str,
    component_name: str,
    window_config: dict[str, float],
    roi_channels: list[str],
    method: AmplitudeMethod = "mean",
    polarity: PeakPolarity | None = None,
) -> dict[str, float]:
    """
    Compute ERP component amplitude from epochs for one condition.

    Parameters
    ----------
    epochs:
        MNE Epochs object.

    condition:
        Condition name in the Epochs object.

    component_name:
        ERP component name.

    window_config:
        Component time-window configuration.

    roi_channels:
        ROI channel names.

    method:
        Amplitude extraction method.

    polarity:
        Peak polarity if peak amplitude is requested.

    Returns
    -------
    dict[str, float]
        ERP component result dictionary.
    """
    if condition not in epochs.event_id:
        return {
            f"{component_name.lower()}_amplitude_uv": np.nan,
        }

    if len(epochs[condition]) == 0:
        return {
            f"{component_name.lower()}_amplitude_uv": np.nan,
        }

    evoked = epochs[condition].average()

    return compute_component_amplitude_uv(
        evoked=evoked,
        component_name=component_name,
        window_config=window_config,
        roi_channels=roi_channels,
        method=method,
        polarity=polarity,
    )


def compute_difference_component_from_epochs(
    epochs: mne.Epochs,
    condition_a: str,
    condition_b: str,
    component_name: str,
    window_config: dict[str, float],
    roi_channels: list[str],
    method: AmplitudeMethod = "mean",
    polarity: PeakPolarity | None = None,
    weight_a: float = 1.0,
    weight_b: float = -1.0,
) -> dict[str, float]:
    """
    Compute ERP component amplitude from a difference wave.

    Example use case:

    ERN = error_response - correct_response

    Parameters
    ----------
    epochs:
        MNE Epochs object.

    condition_a:
        First condition name.

    condition_b:
        Second condition name.

    component_name:
        ERP component name.

    window_config:
        Component time-window configuration.

    roi_channels:
        ROI channel names.

    method:
        Amplitude extraction method.

    polarity:
        Peak polarity if peak amplitude is requested.

    weight_a:
        Weight for first condition.

    weight_b:
        Weight for second condition.

    Returns
    -------
    dict[str, float]
        ERP component result dictionary.
    """
    component_key = component_name.lower()

    if condition_a not in epochs.event_id or condition_b not in epochs.event_id:
        return {
            f"{component_key}_amplitude_uv": np.nan,
        }

    if len(epochs[condition_a]) == 0 or len(epochs[condition_b]) == 0:
        return {
            f"{component_key}_amplitude_uv": np.nan,
        }

    evoked_a = epochs[condition_a].average()
    evoked_b = epochs[condition_b].average()

    difference_evoked = compute_difference_evoked(
        evoked_a=evoked_a,
        evoked_b=evoked_b,
        weight_a=weight_a,
        weight_b=weight_b,
    )

    return compute_component_amplitude_uv(
        evoked=difference_evoked,
        component_name=component_name,
        window_config=window_config,
        roi_channels=roi_channels,
        method=method,
        polarity=polarity,
    )