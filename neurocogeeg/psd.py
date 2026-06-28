"""
Power spectral density utilities for NeuroCogEEG.

This module provides reusable functions for computing absolute and relative
band power from continuous EEG data.

Frequency bands and ROI definitions are experiment-specific and must be
provided through YAML configuration files or explicitly passed to these
functions.
"""

from __future__ import annotations

from typing import Any

import mne
import numpy as np


def get_existing_channels(
    available_channels: list[str],
    requested_channels: list[str],
) -> list[str]:
    """
    Return requested channels that exist in the current data.

    Parameters
    ----------
    available_channels:
        Channel names available in the Raw object.

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


def parse_band_config(
    band_config: dict[str, float] | tuple[float, float] | list[float],
) -> tuple[float, float]:
    """
    Convert a frequency-band configuration into an ``(fmin, fmax)`` tuple.

    Parameters
    ----------
    band_config:
        Frequency band configuration. Supported forms are:

        ``{"fmin": 4.0, "fmax": 8.0}``

        or

        ``(4.0, 8.0)``

    Returns
    -------
    tuple[float, float]
        Frequency limits in Hz.
    """
    if isinstance(band_config, dict):
        return (
            float(band_config["fmin"]),
            float(band_config["fmax"]),
        )

    if isinstance(band_config, (tuple, list)) and len(band_config) == 2:
        return (
            float(band_config[0]),
            float(band_config[1]),
        )

    raise ValueError(f"Invalid band configuration: {band_config!r}")


def compute_psd_array(
    raw: mne.io.BaseRaw,
    channels: list[str],
    fmin: float,
    fmax: float,
    n_fft_seconds: float = 2.0,
    method: str = "welch",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD array for selected channels.

    Parameters
    ----------
    raw:
        MNE Raw object.

    channels:
        Channel names to include.

    fmin:
        Minimum frequency in Hz.

    fmax:
        Maximum frequency in Hz.

    n_fft_seconds:
        FFT window duration in seconds.

    method:
        PSD method. Default is ``"welch"``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        PSD array with shape ``(n_channels, n_freqs)`` and frequency vector.
    """
    existing_channels = get_existing_channels(
        available_channels=raw.ch_names,
        requested_channels=channels,
    )

    sfreq = raw.info["sfreq"]
    n_fft = int(sfreq * n_fft_seconds)

    spectrum = raw.copy().pick(picks=existing_channels).compute_psd(
        method=method,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        verbose=False,
    )

    psds, freqs = spectrum.get_data(return_freqs=True)

    return psds, freqs


def integrate_power(
    psds: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """
    Integrate spectral power within a frequency band.

    Parameters
    ----------
    psds:
        PSD array with shape ``(n_channels, n_freqs)``.

    freqs:
        Frequency vector.

    fmin:
        Minimum frequency in Hz.

    fmax:
        Maximum frequency in Hz.

    Returns
    -------
    np.ndarray
        Band power for each channel.
    """
    freq_mask = np.logical_and(freqs >= fmin, freqs <= fmax)

    if not np.any(freq_mask):
        return np.full(psds.shape[0], np.nan)

    return np.trapz(
        psds[:, freq_mask],
        freqs[freq_mask],
        axis=1,
    )


def compute_roi_band_power(
    raw: mne.io.BaseRaw,
    roi_channels: list[str],
    band_config: dict[str, float] | tuple[float, float] | list[float],
    total_band_config: dict[str, float] | tuple[float, float] | list[float],
    relative: bool = True,
    n_fft_seconds: float = 2.0,
) -> float:
    """
    Compute ROI band power.

    Parameters
    ----------
    raw:
        MNE Raw object.

    roi_channels:
        ROI channel names.

    band_config:
        Target frequency-band configuration.

    total_band_config:
        Total frequency range used for relative power.

    relative:
        If ``True``, return percentage of total power. If ``False``, return
        absolute integrated band power.

    n_fft_seconds:
        FFT window duration in seconds.

    Returns
    -------
    float
        ROI band power. Relative power is returned as percent.
    """
    band_fmin, band_fmax = parse_band_config(band_config)
    total_fmin, total_fmax = parse_band_config(total_band_config)

    psds, freqs = compute_psd_array(
        raw=raw,
        channels=roi_channels,
        fmin=total_fmin,
        fmax=total_fmax,
        n_fft_seconds=n_fft_seconds,
    )

    band_power = integrate_power(
        psds=psds,
        freqs=freqs,
        fmin=band_fmin,
        fmax=band_fmax,
    )

    if not relative:
        return float(np.nanmean(band_power))

    total_power = integrate_power(
        psds=psds,
        freqs=freqs,
        fmin=total_fmin,
        fmax=total_fmax,
    )

    relative_power = (band_power / total_power) * 100.0

    return float(np.nanmean(relative_power))


def compute_multiple_band_powers(
    raw: mne.io.BaseRaw,
    roi_channels: list[str],
    frequency_bands: dict[str, Any],
    total_band_config: dict[str, float] | tuple[float, float] | list[float],
    prefix: str,
    relative: bool = True,
    n_fft_seconds: float = 2.0,
) -> dict[str, float]:
    """
    Compute multiple frequency-band powers for one ROI.

    Parameters
    ----------
    raw:
        MNE Raw object.

    roi_channels:
        ROI channel names.

    frequency_bands:
        Dictionary of frequency-band configurations.

    total_band_config:
        Total frequency range used for relative power.

    prefix:
        Prefix used for output variable names.

    relative:
        If ``True``, compute relative power in percent.

    n_fft_seconds:
        FFT window duration in seconds.

    Returns
    -------
    dict[str, float]
        Band-power result dictionary.
    """
    suffix = "relative_percent" if relative else "absolute_power"

    results: dict[str, float] = {}

    for band_name, band_config in frequency_bands.items():
        value = compute_roi_band_power(
            raw=raw,
            roi_channels=roi_channels,
            band_config=band_config,
            total_band_config=total_band_config,
            relative=relative,
            n_fft_seconds=n_fft_seconds,
        )

        results[f"{prefix}_{band_name}_{suffix}"] = value

    return results


def compute_roi_psd_summary(
    raw: mne.io.BaseRaw,
    roi_config: dict[str, list[str]],
    frequency_bands: dict[str, Any],
    total_band_config: dict[str, float] | tuple[float, float] | list[float],
    relative: bool = True,
    n_fft_seconds: float = 2.0,
) -> dict[str, float]:
    """
    Compute band powers for multiple ROIs.

    Parameters
    ----------
    raw:
        MNE Raw object.

    roi_config:
        Dictionary mapping ROI names to channel lists.

    frequency_bands:
        Dictionary of frequency-band configurations.

    total_band_config:
        Total frequency range used for relative power.

    relative:
        If ``True``, compute relative power in percent.

    n_fft_seconds:
        FFT window duration in seconds.

    Returns
    -------
    dict[str, float]
        Combined PSD result dictionary.
    """
    results: dict[str, float] = {}

    for roi_name, roi_channels in roi_config.items():
        roi_results = compute_multiple_band_powers(
            raw=raw,
            roi_channels=roi_channels,
            frequency_bands=frequency_bands,
            total_band_config=total_band_config,
            prefix=roi_name,
            relative=relative,
            n_fft_seconds=n_fft_seconds,
        )

        results.update(roi_results)

    return results


def crop_raw_segment(
    raw: mne.io.BaseRaw,
    start_time: float,
    end_time: float,
) -> mne.io.BaseRaw:
    """
    Crop a raw segment.

    Parameters
    ----------
    raw:
        MNE Raw object.

    start_time:
        Start time in seconds.

    end_time:
        End time in seconds.

    Returns
    -------
    mne.io.BaseRaw
        Cropped raw segment.
    """
    if end_time <= start_time:
        raise ValueError(
            f"end_time must be greater than start_time. "
            f"Got start={start_time}, end={end_time}"
        )

    return raw.copy().crop(
        tmin=start_time,
        tmax=end_time,
    )


def compute_early_late_psd_summary(
    raw: mne.io.BaseRaw,
    start_time: float,
    end_time: float,
    roi_config: dict[str, list[str]],
    frequency_bands: dict[str, Any],
    total_band_config: dict[str, float] | tuple[float, float] | list[float],
    relative: bool = True,
    n_fft_seconds: float = 2.0,
) -> dict[str, float]:
    """
    Compute early and late PSD summaries for a continuous task segment.

    The task interval is split into two equal temporal halves.

    Parameters
    ----------
    raw:
        MNE Raw object.

    start_time:
        Task start time in seconds.

    end_time:
        Task end time in seconds.

    roi_config:
        Dictionary mapping ROI names to channel lists.

    frequency_bands:
        Frequency-band configurations.

    total_band_config:
        Total frequency range used for relative power.

    relative:
        If ``True``, compute relative power in percent.

    n_fft_seconds:
        FFT window duration in seconds.

    Returns
    -------
    dict[str, float]
        Early, late and change-score PSD results.
    """
    midpoint = (start_time + end_time) / 2.0

    early_raw = crop_raw_segment(
        raw=raw,
        start_time=start_time,
        end_time=midpoint,
    )

    late_raw = crop_raw_segment(
        raw=raw,
        start_time=midpoint,
        end_time=end_time,
    )

    early_results = compute_roi_psd_summary(
        raw=early_raw,
        roi_config=roi_config,
        frequency_bands=frequency_bands,
        total_band_config=total_band_config,
        relative=relative,
        n_fft_seconds=n_fft_seconds,
    )

    late_results = compute_roi_psd_summary(
        raw=late_raw,
        roi_config=roi_config,
        frequency_bands=frequency_bands,
        total_band_config=total_band_config,
        relative=relative,
        n_fft_seconds=n_fft_seconds,
    )

    results: dict[str, float] = {}

    for key, early_value in early_results.items():
        late_value = late_results[key]

        results[f"early_{key}"] = early_value
        results[f"late_{key}"] = late_value
        results[f"change_{key}"] = late_value - early_value

    return results