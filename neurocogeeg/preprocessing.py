"""
EEG preprocessing utilities for NeuroCogEEG.

This module contains reusable preprocessing functions for Emotiv EPOC X
recordings. It handles EDF loading, channel selection, montage assignment,
referencing, filtering, ICA-based artifact removal and marker-channel handling.

Experiment-specific epoching, ERP analysis, PSD analysis and connectivity
analysis are not performed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
from mne_icalabel import label_components

from neurocogeeg.utils import load_yaml


VALID_EXPERIMENTS = (
    "flanker",
    "gonogo",
    "readysetgo",
    "tmt",
)


def get_project_root() -> Path:
    """
    Return the NeuroCogEEG repository root directory.

    Returns
    -------
    Path
        Path to the repository root.
    """
    return Path(__file__).resolve().parents[1]


def get_default_device_config_path() -> Path:
    """
    Return the default Emotiv EPOC X configuration file path.

    Returns
    -------
    Path
        Path to ``configs/emotiv_epocx.yaml``.
    """
    return get_project_root() / "configs" / "emotiv_epocx.yaml"


def get_default_experiment_config_path(experiment: str) -> Path:
    """
    Return the default experiment configuration file path.

    Parameters
    ----------
    experiment:
        Experiment name. Must be one of ``flanker``, ``gonogo``,
        ``readysetgo`` or ``tmt``.

    Returns
    -------
    Path
        Path to the corresponding experiment YAML file.
    """
    if experiment not in VALID_EXPERIMENTS:
        valid = ", ".join(VALID_EXPERIMENTS)
        raise ValueError(
            f"Unknown experiment: {experiment!r}. "
            f"Valid experiments are: {valid}"
        )

    return get_project_root() / "configs" / f"{experiment}.yaml"


def load_device_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load device-level YAML configuration.

    Parameters
    ----------
    config_path:
        Optional path to a device YAML file. If not provided,
        ``configs/emotiv_epocx.yaml`` is used.

    Returns
    -------
    dict[str, Any]
        Device configuration dictionary.
    """
    path = (
        Path(config_path)
        if config_path is not None
        else get_default_device_config_path()
    )
    return load_yaml(path)


def load_experiment_config(
    experiment: str,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load experiment-level YAML configuration.

    Parameters
    ----------
    experiment:
        Experiment name. Must be one of ``flanker``, ``gonogo``,
        ``readysetgo`` or ``tmt``.

    config_path:
        Optional path to an experiment YAML file. If not provided,
        ``configs/<experiment>.yaml`` is used.

    Returns
    -------
    dict[str, Any]
        Experiment configuration dictionary.
    """
    path = (
        Path(config_path)
        if config_path is not None
        else get_default_experiment_config_path(experiment)
    )
    return load_yaml(path)


def read_edf(edf_path: str | Path) -> mne.io.BaseRaw:
    """
    Read an EDF file with preload enabled.

    Parameters
    ----------
    edf_path:
        Path to EDF file.

    Returns
    -------
    mne.io.BaseRaw
        Loaded raw object.
    """
    path = Path(edf_path)

    if not path.exists():
        raise FileNotFoundError(f"EDF file not found: {path}")

    return mne.io.read_raw_edf(path, preload=True, verbose=False)


def find_marker_channel(
    raw: mne.io.BaseRaw,
    marker_config: dict[str, Any],
) -> str:
    """
    Find the marker channel in a raw EDF recording.

    Parameters
    ----------
    raw:
        Raw EEG object.

    marker_config:
        Marker channel configuration from ``emotiv_epocx.yaml``.
        Expected keys are ``candidates`` and optionally ``fallback_index``.

    Returns
    -------
    str
        Selected marker channel name.

    Raises
    ------
    ValueError
        If no marker channel can be identified.
    """
    marker_candidates = marker_config.get("candidates", [])

    for marker_name in marker_candidates:
        if marker_name in raw.ch_names:
            return marker_name

    fallback_index = marker_config.get("fallback_index", None)

    if fallback_index is not None:
        if 0 <= fallback_index < len(raw.ch_names):
            fallback_channel = raw.ch_names[fallback_index]
            print(
                "⚠️ UYARI: Marker kanalı isimle bulunamadı. "
                f"Fallback index kullanılıyor: {fallback_index} "
                f"({fallback_channel})"
            )
            return fallback_channel

    raise ValueError(
        "No marker channel found. "
        f"Expected one of: {marker_candidates}. "
        f"Fallback index: {fallback_index}. "
        f"Available channels: {raw.ch_names}"
    )


def pick_emotiv_eeg_channels(
    raw: mne.io.BaseRaw,
    eeg_channels: list[str],
) -> mne.io.BaseRaw:
    """
    Select Emotiv EEG channels from a raw recording.

    Parameters
    ----------
    raw:
        Raw EEG object.

    eeg_channels:
        Expected Emotiv EEG channel names.

    Returns
    -------
    mne.io.BaseRaw
        Raw object containing only selected EEG channels.
    """
    available_channels = [ch for ch in eeg_channels if ch in raw.ch_names]

    if not available_channels:
        raise ValueError(
            "No expected Emotiv EEG channels were found in the EDF file."
        )

    missing_channels = sorted(set(eeg_channels) - set(available_channels))

    if missing_channels:
        print(
            "⚠️ UYARI: Bazı Emotiv kanalları dosyada bulunamadı: "
            f"{missing_channels}"
        )

    return raw.copy().pick(picks=available_channels, verbose=False)


def set_standard_montage(
    raw_eeg: mne.io.BaseRaw,
    montage_name: str,
    coordinate_frame: str = "head",
) -> mne.io.BaseRaw:
    """
    Set a standard montage for available EEG channels.

    Parameters
    ----------
    raw_eeg:
        Raw object containing EEG channels.

    montage_name:
        MNE montage name, for example ``standard_1020``.

    coordinate_frame:
        Coordinate frame for digitization montage.

    Returns
    -------
    mne.io.BaseRaw
        Raw object with montage assigned.
    """
    montage = mne.channels.make_standard_montage(montage_name)
    montage_positions = montage.get_positions()["ch_pos"]

    ch_pos = {
        ch_name: montage_positions[ch_name]
        for ch_name in raw_eeg.ch_names
        if ch_name in montage_positions
    }

    if not ch_pos:
        raise ValueError(
            "None of the EEG channels were found in the selected montage."
        )

    dig_montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        coord_frame=coordinate_frame,
    )

    raw_eeg.set_montage(dig_montage)
    return raw_eeg


def set_eog_channels(
    raw_eeg: mne.io.BaseRaw,
    eog_channels: list[str],
) -> mne.io.BaseRaw:
    """
    Mark selected channels as EOG.

    Parameters
    ----------
    raw_eeg:
        Raw object containing EEG channels.

    eog_channels:
        Channels to mark as EOG.

    Returns
    -------
    mne.io.BaseRaw
        Raw object with EOG channel types updated.
    """
    channel_types = {
        ch_name: "eog"
        for ch_name in eog_channels
        if ch_name in raw_eeg.ch_names
    }

    if channel_types:
        raw_eeg.set_channel_types(channel_types, verbose=False)

    return raw_eeg


def apply_reference(
    raw_eeg: mne.io.BaseRaw,
    reference_config: dict[str, Any],
) -> mne.io.BaseRaw:
    """
    Apply EEG reference according to device configuration.

    Parameters
    ----------
    raw_eeg:
        Raw EEG object.

    reference_config:
        Reference configuration from ``emotiv_epocx.yaml``.

    Returns
    -------
    mne.io.BaseRaw
        Referenced raw EEG object.
    """
    reference_type = reference_config.get("type", "average")

    if reference_type == "average":
        raw_eeg.set_eeg_reference(
            "average",
            projection=False,
            verbose=False,
        )
    else:
        raise ValueError(f"Unsupported reference type: {reference_type!r}")

    return raw_eeg


def apply_basic_filtering(
    raw_eeg: mne.io.BaseRaw,
    preprocessing_config: dict[str, Any],
) -> mne.io.BaseRaw:
    """
    Apply analysis filter and notch filter.

    Parameters
    ----------
    raw_eeg:
        Raw object to filter.

    preprocessing_config:
        Preprocessing section from experiment YAML.

    Returns
    -------
    mne.io.BaseRaw
        Filtered raw object.
    """
    analysis_filter = preprocessing_config["filtering"]["analysis"]

    raw_eeg.filter(
        l_freq=analysis_filter["l_freq"],
        h_freq=analysis_filter["h_freq"],
        fir_design="firwin",
        verbose=False,
    )

    notch_config = preprocessing_config.get("notch_filter", {})

    if notch_config.get("enabled", False):
        raw_eeg.notch_filter(
            freqs=notch_config["frequency"],
            fir_design="firwin",
            verbose=False,
        )

    return raw_eeg


def prepare_ica_raw(
    raw_eeg: mne.io.BaseRaw,
    preprocessing_config: dict[str, Any],
) -> mne.io.BaseRaw:
    """
    Prepare a filtered copy of raw data for ICA fitting.

    Parameters
    ----------
    raw_eeg:
        Raw EEG object.

    preprocessing_config:
        Preprocessing section from experiment YAML.

    Returns
    -------
    mne.io.BaseRaw
        Filtered raw object for ICA fitting.
    """
    ica_filter = preprocessing_config["filtering"]["ica"]

    raw_for_ica = raw_eeg.copy().filter(
        l_freq=ica_filter["l_freq"],
        h_freq=ica_filter["h_freq"],
        fir_design="firwin",
        verbose=False,
    )

    return raw_for_ica


def run_ica_cleaning(
    raw_eeg: mne.io.BaseRaw,
    raw_for_ica: mne.io.BaseRaw,
    preprocessing_config: dict[str, Any],
) -> tuple[mne.io.BaseRaw, mne.preprocessing.ICA, dict[str, Any]]:
    """
    Run ICA and remove non-brain components using ICLabel.

    Parameters
    ----------
    raw_eeg:
        Filtered raw EEG object used for final analysis.

    raw_for_ica:
        Filtered raw EEG object used only for ICA fitting.

    preprocessing_config:
        Preprocessing section from experiment YAML.

    Returns
    -------
    tuple[mne.io.BaseRaw, mne.preprocessing.ICA, dict[str, Any]]
        Cleaned raw object, fitted ICA object and ICLabel output.
    """
    ica_config = preprocessing_config["ica"]

    ica = mne.preprocessing.ICA(
        n_components=ica_config["n_components"],
        random_state=ica_config["random_state"],
        method=ica_config["method"],
        fit_params=ica_config.get("fit_params", None),
    )

    ica.fit(raw_for_ica, verbose=False)

    iclabel_config = preprocessing_config.get("iclabel", {})
    ic_labels: dict[str, Any] = {}

    if iclabel_config.get("enabled", False):
        ic_labels = label_components(
            raw_for_ica,
            ica,
            method=iclabel_config.get("method", "iclabel"),
        )

        if iclabel_config.get("exclude_non_brain", True):
            labels = np.array(ic_labels["labels"])
            ica.exclude = np.where(labels != "brain")[0].tolist()

    raw_clean = ica.apply(
        raw_eeg.copy(),
        exclude=ica.exclude,
        verbose=False,
    )

    return raw_clean, ica, ic_labels


def preprocess_emotiv_edf(
    edf_path: str | Path,
    experiment: str,
    device_config_path: str | Path | None = None,
    experiment_config_path: str | Path | None = None,
) -> tuple[mne.io.BaseRaw, mne.preprocessing.ICA | None, dict[str, Any]]:
    """
    Full preprocessing pipeline for one Emotiv EPOC X EDF file.

    This function returns a cleaned raw object with the marker channel attached.

    Parameters
    ----------
    edf_path:
        Path to EDF file.

    experiment:
        Experiment name. Must be one of ``flanker``, ``gonogo``,
        ``readysetgo`` or ``tmt``.

    device_config_path:
        Optional path to device YAML file. If not provided,
        ``configs/emotiv_epocx.yaml`` is used.

    experiment_config_path:
        Optional path to experiment YAML file. If not provided,
        ``configs/<experiment>.yaml`` is used.

    Returns
    -------
    tuple[mne.io.BaseRaw, mne.preprocessing.ICA | None, dict[str, Any]]
        Cleaned raw object, ICA object if ICA was enabled, and a preprocessing
        information dictionary.
    """
    device_config = load_device_config(device_config_path)
    experiment_config = load_experiment_config(
        experiment=experiment,
        config_path=experiment_config_path,
    )

    preprocessing_config = experiment_config["preprocessing"]

    raw = read_edf(edf_path)

    eeg_channels = device_config["channels"]["eeg"]
    eog_channels = device_config["channels"].get("eog", [])

    marker_channel = find_marker_channel(
        raw=raw,
        marker_config=device_config["marker_channels"],
    )

    stim_channel = raw.copy().pick(
        picks=[marker_channel],
        verbose=False,
    )

    raw_eeg = pick_emotiv_eeg_channels(
        raw=raw,
        eeg_channels=eeg_channels,
    )

    raw_eeg = set_standard_montage(
        raw_eeg=raw_eeg,
        montage_name=device_config["montage"]["name"],
        coordinate_frame=device_config["montage"].get(
            "coordinate_frame",
            "head",
        ),
    )

    raw_eeg = set_eog_channels(
        raw_eeg=raw_eeg,
        eog_channels=eog_channels,
    )

    raw_eeg = apply_reference(
        raw_eeg=raw_eeg,
        reference_config=device_config["reference"],
    )

    raw_for_ica = prepare_ica_raw(
        raw_eeg=raw_eeg,
        preprocessing_config=preprocessing_config,
    )

    raw_eeg = apply_basic_filtering(
        raw_eeg=raw_eeg,
        preprocessing_config=preprocessing_config,
    )

    ica_object: mne.preprocessing.ICA | None = None
    ic_labels: dict[str, Any] = {}

    if preprocessing_config["ica"].get("enabled", False):
        raw_clean, ica_object, ic_labels = run_ica_cleaning(
            raw_eeg=raw_eeg,
            raw_for_ica=raw_for_ica,
            preprocessing_config=preprocessing_config,
        )
    else:
        raw_clean = raw_eeg.copy()

    raw_clean.add_channels(
        [stim_channel],
        force_update_info=True,
    )

    preprocessing_info = {
        "edf_path": str(edf_path),
        "experiment": experiment,
        "marker_channel": marker_channel,
        "analysis_filter": preprocessing_config["filtering"]["analysis"],
        "ica_filter": preprocessing_config["filtering"]["ica"],
        "notch_filter": preprocessing_config.get("notch_filter", {}),
        "ica_enabled": preprocessing_config["ica"].get("enabled", False),
        "ica_excluded_components": (
            ica_object.exclude if ica_object is not None else []
        ),
        "ic_labels": ic_labels,
    }

    return raw_clean, ica_object, preprocessing_info
