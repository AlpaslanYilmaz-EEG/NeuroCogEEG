"""
Functional connectivity utilities for NeuroCogEEG.

This module provides reusable functions for computing spectral connectivity
between EEG channel pairs.

Connectivity methods, frequency bands and channel pairs are experiment-specific
and must be provided through YAML configuration files.

No channel pair or frequency band should be hard-coded in this module.
"""

from __future__ import annotations

from typing import Any

import mne
import numpy as np


def parse_frequency_band(
    band_config: dict[str, float] | tuple[float, float] | list[float],
) -> tuple[float, float]:
    """
    Convert a frequency-band configuration into an ``(fmin, fmax)`` tuple.

    Parameters
    ----------
    band_config:
        Frequency-band configuration. Supported forms are:

        ``{"fmin": 4.0, "fmax": 8.0}``

        or

        ``(4.0, 8.0)``.

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

    raise ValueError(f"Invalid frequency-band configuration: {band_config!r}")


def make_pair_key(
    pair_group: str,
    source: str,
    target: str,
) -> str:
    """
    Create a readable output key for a connectivity pair.

    Parameters
    ----------
    pair_group:
        Pair group name, for example ``interhemispheric``.

    source:
        Source channel name.

    target:
        Target channel name.

    Returns
    -------
    str
        Pair key.
    """
    return f"{pair_group}_{source}_{target}"


def flatten_connectivity_pairs(
    pairs_config: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    """
    Flatten YAML pair configuration into a list of pair dictionaries.

    Parameters
    ----------
    pairs_config:
        Pair configuration from YAML.

        Example:

        ``{"interhemispheric": [{"source": "F3", "target": "F4"}]}``

    Returns
    -------
    list[dict[str, str]]
        Flattened list with keys ``group``, ``source`` and ``target``.
    """
    flattened_pairs: list[dict[str, str]] = []

    for pair_group, pair_list in pairs_config.items():
        for pair in pair_list:
            flattened_pairs.append(
                {
                    "group": pair_group,
                    "source": pair["source"],
                    "target": pair["target"],
                }
            )

    return flattened_pairs


def build_indices_from_pairs(
    ch_names: list[str],
    pairs: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]], list[dict[str, str]]]:
    """
    Convert channel-pair definitions into MNE connectivity indices.

    Parameters
    ----------
    ch_names:
        Channel names available in the Epochs object.

    pairs:
        Flattened pair definitions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[dict[str, str]], list[dict[str, str]]]
        Source indices, target indices, valid pairs and missing pairs.
    """
    source_indices: list[int] = []
    target_indices: list[int] = []
    valid_pairs: list[dict[str, str]] = []
    missing_pairs: list[dict[str, str]] = []

    for pair in pairs:
        source = pair["source"]
        target = pair["target"]

        if source in ch_names and target in ch_names:
            source_indices.append(ch_names.index(source))
            target_indices.append(ch_names.index(target))
            valid_pairs.append(pair)
        else:
            missing_pairs.append(pair)

    return (
        np.asarray(source_indices, dtype=int),
        np.asarray(target_indices, dtype=int),
        valid_pairs,
        missing_pairs,
    )


def fisher_z_transform(
    value: float,
    lower: float = 0.0,
    upper: float = 0.999999,
) -> float:
    """
    Apply Fisher z transformation to a bounded connectivity value.

    Parameters
    ----------
    value:
        Connectivity value.

    lower:
        Lower clipping limit.

    upper:
        Upper clipping limit.

    Returns
    -------
    float
        Fisher z-transformed value.
    """
    clipped = np.clip(value, lower, upper)
    return float(np.arctanh(clipped))


def _extract_compact_connectivity_value(
    compact_data: np.ndarray,
    pair_index: int,
) -> float:
    """
    Extract a scalar connectivity value from compact MNE output.

    Parameters
    ----------
    compact_data:
        Output of ``con.get_data()``.

    pair_index:
        Pair index.

    Returns
    -------
    float
        Connectivity value.
    """
    value = np.asarray(compact_data[pair_index]).squeeze()

    if value.size == 0:
        return float("nan")

    if value.ndim == 0:
        return float(value)

    return float(np.nanmean(value))


def compute_connectivity_for_band(
    epochs: mne.Epochs,
    method: str,
    fmin: float,
    fmax: float,
    pairs: list[dict[str, str]],
    tmin: float | None = None,
    tmax: float | None = None,
    mode: str = "multitaper",
    faverage: bool = True,
    fisher_z: bool = False,
) -> dict[str, float]:
    """
    Compute connectivity for one method and one frequency band.

    Parameters
    ----------
    epochs:
        MNE Epochs object.

    method:
        Connectivity method supported by MNE-Connectivity, for example
        ``coh``, ``wpli`` or ``ciplv``.

    fmin:
        Minimum frequency in Hz.

    fmax:
        Maximum frequency in Hz.

    pairs:
        Flattened channel-pair definitions.

    tmin:
        Optional start time for connectivity calculation.

    tmax:
        Optional end time for connectivity calculation.

    mode:
        Spectral estimation mode.

    faverage:
        Whether to average connectivity values across the frequency band.

    fisher_z:
        Whether to apply Fisher z transformation.

    Returns
    -------
    dict[str, float]
        Connectivity values for all requested pairs.
    """
    from mne_connectivity import spectral_connectivity_epochs

    source_indices, target_indices, valid_pairs, missing_pairs = (
        build_indices_from_pairs(
            ch_names=epochs.ch_names,
            pairs=pairs,
        )
    )

    results: dict[str, float] = {}

    for pair in missing_pairs:
        pair_key = make_pair_key(
            pair_group=pair["group"],
            source=pair["source"],
            target=pair["target"],
        )
        results[pair_key] = np.nan

    if len(valid_pairs) == 0:
        return results

    indices = (
        source_indices,
        target_indices,
    )

    connectivity = spectral_connectivity_epochs(
        epochs,
        method=method,
        mode=mode,
        sfreq=epochs.info["sfreq"],
        fmin=fmin,
        fmax=fmax,
        faverage=faverage,
        indices=indices,
        tmin=tmin,
        tmax=tmax,
        verbose=False,
    )

    compact_data = connectivity.get_data()

    for pair_index, pair in enumerate(valid_pairs):
        pair_key = make_pair_key(
            pair_group=pair["group"],
            source=pair["source"],
            target=pair["target"],
        )

        value = _extract_compact_connectivity_value(
            compact_data=compact_data,
            pair_index=pair_index,
        )

        if fisher_z and not np.isnan(value):
            value = fisher_z_transform(value)

        results[pair_key] = value

    return results


def compute_connectivity_summary(
    epochs: mne.Epochs,
    connectivity_config: dict[str, Any],
    tmin: float | None = None,
    tmax: float | None = None,
    mode: str = "multitaper",
    faverage: bool = True,
    fisher_z: bool = False,
) -> dict[str, float]:
    """
    Compute connectivity summary from YAML connectivity configuration.

    Parameters
    ----------
    epochs:
        MNE Epochs object.

    connectivity_config:
        Connectivity section from experiment YAML.

    tmin:
        Optional start time for connectivity calculation.

    tmax:
        Optional end time for connectivity calculation.

    mode:
        Spectral estimation mode.

    faverage:
        Whether to average connectivity values across each frequency band.

    fisher_z:
        Whether to apply Fisher z transformation.

    Returns
    -------
    dict[str, float]
        Connectivity result dictionary.
    """
    methods = connectivity_config["methods"]
    frequency_bands = connectivity_config["frequency_bands"]
    pairs = flatten_connectivity_pairs(connectivity_config["pairs"])

    results: dict[str, float] = {}

    for method in methods:
        method_name = str(method).lower()

        for band_name, band_config in frequency_bands.items():
            fmin, fmax = parse_frequency_band(band_config)

            band_results = compute_connectivity_for_band(
                epochs=epochs,
                method=method_name,
                fmin=fmin,
                fmax=fmax,
                pairs=pairs,
                tmin=tmin,
                tmax=tmax,
                mode=mode,
                faverage=faverage,
                fisher_z=fisher_z,
            )

            for pair_key, value in band_results.items():
                if fisher_z:
                    output_key = f"z_{method_name}_{band_name}_{pair_key}"
                else:
                    output_key = f"{method_name}_{band_name}_{pair_key}"

                results[output_key] = value

    return results