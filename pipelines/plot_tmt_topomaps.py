"""
TMT topomap figure pipeline for NeuroCogEEG.

This script creates descriptive scalp topomap figures for TMT relative power.

It reuses the validated preprocessing logic and computes channel-level
relative PSD values from the continuous TMT task segment.

Outputs:
outputs/tmt/figures/topomaps/
outputs/qc/tmt_topomap_manifest.csv
outputs/tmt/csv/tmt_topomap_channel_relative_power.csv

Important:
These figures are descriptive scalp visualizations. They should not be treated
as high-density source-localization results because Emotiv EPOC X has a limited
number of EEG channels.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from neurocogeeg.behavior import get_event_code
from neurocogeeg.dataset import SubjectRecord, get_dataset
from neurocogeeg.epochs import extract_events_from_marker_channel
from neurocogeeg.preprocessing import (
    load_device_config,
    load_experiment_config,
    preprocess_emotiv_edf,
)


EXPERIMENT = "tmt"

GROUP_ORDER = [
    "control",
    "experimental",
]

VARIANT_ORDER = [
    "tmt1",
    "tmt2",
]

BAND_ORDER = [
    "theta",
    "alpha",
    "beta",
]


def get_task_segment_samples(events, experiment_config):
    """
    Return TMT task start and end samples.

    Parameters
    ----------
    events:
        MNE events array.

    experiment_config:
        Loaded tmt.yaml configuration.

    Returns
    -------
    tuple[int | None, int | None]
        Start and end sample. Returns (None, None) if the segment cannot be
        identified.
    """
    events_config = experiment_config["events"]
    segment_config = experiment_config["continuous_eeg"]["segment"]

    start_code = get_event_code(
        events_config=events_config,
        event_name=segment_config["start_event"],
    )

    end_codes = [
        get_event_code(
            events_config=events_config,
            event_name=end_event,
        )
        for end_event in segment_config["end_events"]
    ]

    start_samples = events[events[:, 2] == start_code, 0]

    if start_samples.size == 0:
        return None, None

    end_mask = np.isin(events[:, 2], end_codes)
    end_samples = events[end_mask, 0]

    if end_samples.size == 0:
        return None, None

    start_sample = int(start_samples[0])
    end_sample = int(end_samples[-1])

    if end_sample <= start_sample:
        return None, None

    return start_sample, end_sample


def get_task_segment_times(events, sfreq, experiment_config):
    """
    Return TMT task start and end times in seconds.
    """
    start_sample, end_sample = get_task_segment_samples(
        events=events,
        experiment_config=experiment_config,
    )

    if start_sample is None or end_sample is None:
        return None, None

    return float(start_sample / sfreq), float(end_sample / sfreq)


def crop_task_segment(raw_clean, events, experiment_config):
    """
    Crop raw object to the continuous TMT task segment.
    """
    sfreq = float(raw_clean.info["sfreq"])

    start_time, end_time = get_task_segment_times(
        events=events,
        sfreq=sfreq,
        experiment_config=experiment_config,
    )

    if start_time is None or end_time is None:
        return None

    if end_time <= start_time:
        return None

    return raw_clean.copy().crop(
        tmin=start_time,
        tmax=end_time,
        include_tmax=True,
    )


def prepare_eeg_raw_for_topomap(raw_segment):
    """
    Pick EEG channels and attach a standard 10-20 montage if possible.
    """
    eeg_raw = raw_segment.copy().pick("eeg")

    if len(eeg_raw.ch_names) == 0:
        return None

    try:
        eeg_raw.set_montage(
            "standard_1020",
            match_case=False,
            on_missing="ignore",
        )
    except Exception:
        pass

    return eeg_raw


def integrate_band_power(psds, freqs, fmin, fmax):
    """
    Integrate PSD values within one frequency band.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(mask):
        return np.full(psds.shape[0], np.nan)

    return np.trapz(
        psds[:, mask],
        freqs[mask],
        axis=1,
    )


def compute_channel_relative_power(eeg_raw, psd_config):
    """
    Compute channel-level relative power for theta, alpha and beta bands.

    Returns
    -------
    dict[str, np.ndarray]
        Band name to relative power vector in percent.
    """
    data = eeg_raw.get_data()
    sfreq = float(eeg_raw.info["sfreq"])
    n_times = data.shape[1]

    n_fft_seconds = float(psd_config.get("n_fft_seconds", 2.0))
    n_fft = int(round(n_fft_seconds * sfreq))
    n_fft = max(8, min(n_fft, n_times))

    psds, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=float(psd_config["total_band"]["fmin"]),
        fmax=float(psd_config["total_band"]["fmax"]),
        n_fft=n_fft,
        n_per_seg=n_fft,
        n_overlap=n_fft // 2,
        average="mean",
        verbose=False,
    )

    total_power = integrate_band_power(
        psds=psds,
        freqs=freqs,
        fmin=float(psd_config["total_band"]["fmin"]),
        fmax=float(psd_config["total_band"]["fmax"]),
    )

    results = {}

    for band_name in BAND_ORDER:
        band_config = psd_config["frequency_bands"][band_name]

        band_power = integrate_band_power(
            psds=psds,
            freqs=freqs,
            fmin=float(band_config["fmin"]),
            fmax=float(band_config["fmax"]),
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            relative_power = (band_power / total_power) * 100.0

        results[band_name] = relative_power

    return results


def initialize_topomap_store():
    """
    Create empty storage for group/variant/band topomap values.
    """
    store = {}

    for variant in VARIANT_ORDER:
        for group in GROUP_ORDER:
            for band in BAND_ORDER:
                store[(variant, group, band)] = {
                    "vectors": [],
                    "info": None,
                    "ch_names": None,
                }

    return store


def add_subject_to_store(
    store,
    subject,
    eeg_raw,
    band_values,
):
    """
    Add one subject's channel-level band vectors to the topomap store.
    """
    rows = []

    for band_name, values in band_values.items():
        key = (subject.variant, subject.group, band_name)

        if key not in store:
            rows.append(
                {
                    "subject_id": subject.subject_id,
                    "group": subject.group,
                    "variant": subject.variant,
                    "band": band_name,
                    "status": "skipped_unexpected_group_or_variant",
                    "n_channels": len(eeg_raw.ch_names),
                }
            )
            continue

        entry = store[key]

        if entry["ch_names"] is None:
            entry["ch_names"] = list(eeg_raw.ch_names)
            entry["info"] = eeg_raw.info.copy()

        if list(eeg_raw.ch_names) != entry["ch_names"]:
            rows.append(
                {
                    "subject_id": subject.subject_id,
                    "group": subject.group,
                    "variant": subject.variant,
                    "band": band_name,
                    "status": "skipped_channel_mismatch",
                    "n_channels": len(eeg_raw.ch_names),
                }
            )
            continue

        if np.all(np.isnan(values)):
            rows.append(
                {
                    "subject_id": subject.subject_id,
                    "group": subject.group,
                    "variant": subject.variant,
                    "band": band_name,
                    "status": "skipped_all_nan_values",
                    "n_channels": len(eeg_raw.ch_names),
                }
            )
            continue

        entry["vectors"].append(values)

        rows.append(
            {
                "subject_id": subject.subject_id,
                "group": subject.group,
                "variant": subject.variant,
                "band": band_name,
                "status": "added",
                "n_channels": len(eeg_raw.ch_names),
            }
        )

    return rows


def process_subject(
    subject: SubjectRecord,
    device_config,
    experiment_config,
    store,
):
    """
    Process one TMT subject for topomap values.
    """
    print(
        f"İşleniyor: {subject.subject_id} "
        f"({subject.group}, {subject.variant})"
    )

    try:
        raw_clean, _ica, preprocessing_info = preprocess_emotiv_edf(
            edf_path=subject.edf_path,
            experiment=EXPERIMENT,
        )

        events = extract_events_from_marker_channel(
            raw=raw_clean,
            marker_channel=preprocessing_info["marker_channel"],
            marker_config=device_config["marker_channels"],
        )

        raw_segment = crop_task_segment(
            raw_clean=raw_clean,
            events=events,
            experiment_config=experiment_config,
        )

        if raw_segment is None:
            return [
                {
                    "subject_id": subject.subject_id,
                    "group": subject.group,
                    "variant": subject.variant,
                    "band": "",
                    "status": "skipped_no_task_segment",
                    "n_channels": "",
                }
            ], []

        eeg_raw = prepare_eeg_raw_for_topomap(raw_segment)

        if eeg_raw is None:
            return [
                {
                    "subject_id": subject.subject_id,
                    "group": subject.group,
                    "variant": subject.variant,
                    "band": "",
                    "status": "skipped_no_eeg_channels",
                    "n_channels": "",
                }
            ], []

        band_values = compute_channel_relative_power(
            eeg_raw=eeg_raw,
            psd_config=experiment_config["psd"],
        )

        manifest_rows = add_subject_to_store(
            store=store,
            subject=subject,
            eeg_raw=eeg_raw,
            band_values=band_values,
        )

        channel_rows = []

        for band_name, values in band_values.items():
            for channel, value in zip(eeg_raw.ch_names, values):
                channel_rows.append(
                    {
                        "subject_id": subject.subject_id,
                        "group": subject.group,
                        "variant": subject.variant,
                        "band": band_name,
                        "channel": channel,
                        "relative_power_percent": float(value),
                    }
                )

        return manifest_rows, channel_rows

    except Exception as error:
        return [
            {
                "subject_id": subject.subject_id,
                "group": subject.group,
                "variant": subject.variant,
                "band": "",
                "status": "processing_error",
                "n_channels": "",
                "error": str(error),
            }
        ], []


def compute_mean_vector(vectors):
    """
    Compute mean channel vector.
    """
    array = np.asarray(vectors, dtype=float)

    if array.size == 0:
        return None

    return np.nanmean(array, axis=0)


def compute_band_vlim(store, band):
    """
    Compute a common color range for one band across all groups and variants.
    """
    mean_vectors = []

    for variant in VARIANT_ORDER:
        for group in GROUP_ORDER:
            entry = store[(variant, group, band)]
            mean_vector = compute_mean_vector(entry["vectors"])

            if mean_vector is not None:
                mean_vectors.append(mean_vector)

    if not mean_vectors:
        return None, None

    combined = np.concatenate(mean_vectors)

    vmin = float(np.nanmin(combined))
    vmax = float(np.nanmax(combined))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None

    if vmin == vmax:
        padding = abs(vmin) * 0.05 if vmin != 0 else 1.0
        vmin -= padding
        vmax += padding

    return vmin, vmax


def plot_topomap_compatible(values, info, axes, vmin, vmax):
    """
    Plot a topomap across MNE versions.

    Some MNE versions do not support show_names or vlim. Therefore this
    function avoids channel-name drawing and falls back from vlim to vmin/vmax
    when needed.
    """
    try:
        image, _ = mne.viz.plot_topomap(
            values,
            info,
            axes=axes,
            show=False,
            sensors=True,
            contours=6,
            vlim=(vmin, vmax),
        )
    except TypeError:
        image, _ = mne.viz.plot_topomap(
            values,
            info,
            axes=axes,
            show=False,
            sensors=True,
            contours=6,
            vmin=vmin,
            vmax=vmax,
        )

    return image


def save_one_topomap(
    store,
    variant,
    group,
    band,
    vmin,
    vmax,
    output_dir,
):
    """
    Save one group/variant/band topomap.
    """
    entry = store[(variant, group, band)]

    mean_vector = compute_mean_vector(entry["vectors"])

    if mean_vector is None:
        return {
            "variant": variant,
            "group": group,
            "band": band,
            "status": "skipped_no_group_data",
            "n_subjects": 0,
            "output_path": "",
        }

    if entry["info"] is None:
        return {
            "variant": variant,
            "group": group,
            "band": band,
            "status": "skipped_no_info",
            "n_subjects": len(entry["vectors"]),
            "output_path": "",
        }

    fig, ax = plt.subplots(figsize=(6, 5))

    image = plot_topomap_compatible(
        values=mean_vector,
        info=entry["info"],
        axes=ax,
        vmin=vmin,
        vmax=vmax,
    )

    title = (
        f"TMT {variant.upper()} - {group}\n"
        f"{band} relative power (n={len(entry['vectors'])})"
    )

    ax.set_title(title)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("Relative power (%)")

    fig.tight_layout()

    filename = f"tmt_{variant}_{group}_{band}_relative_power_topomap.png"
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return {
        "variant": variant,
        "group": group,
        "band": band,
        "status": "created",
        "n_subjects": len(entry["vectors"]),
        "output_path": str(output_path),
    }


def save_all_topomaps(store):
    """
    Save all group/variant/band topomaps.
    """
    output_dir = PROJECT_ROOT / "outputs" / "tmt" / "figures" / "topomaps"

    rows = []

    for band in BAND_ORDER:
        vmin, vmax = compute_band_vlim(
            store=store,
            band=band,
        )

        if vmin is None or vmax is None:
            for variant in VARIANT_ORDER:
                for group in GROUP_ORDER:
                    rows.append(
                        {
                            "variant": variant,
                            "group": group,
                            "band": band,
                            "status": "skipped_no_band_data",
                            "n_subjects": 0,
                            "output_path": "",
                        }
                    )
            continue

        for variant in VARIANT_ORDER:
            for group in GROUP_ORDER:
                rows.append(
                    save_one_topomap(
                        store=store,
                        variant=variant,
                        group=group,
                        band=band,
                        vmin=vmin,
                        vmax=vmax,
                        output_dir=output_dir,
                    )
                )

    return rows


def write_dataframe(dataframe, path):
    """
    Write a semicolon-separated CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )


def main():
    dataset = get_dataset(EXPERIMENT)
    device_config = load_device_config()
    experiment_config = load_experiment_config(EXPERIMENT)

    subjects = dataset.list_subjects()

    if not subjects:
        print("TMT EDF dosyası bulunamadı.")
        return

    store = initialize_topomap_store()

    manifest_rows = []
    channel_rows = []

    for subject in subjects:
        subject_manifest_rows, subject_channel_rows = process_subject(
            subject=subject,
            device_config=device_config,
            experiment_config=experiment_config,
            store=store,
        )

        manifest_rows.extend(subject_manifest_rows)
        channel_rows.extend(subject_channel_rows)

    plot_rows = save_all_topomaps(store)

    for row in plot_rows:
        manifest_rows.append(
            {
                "subject_id": "",
                "group": row["group"],
                "variant": row["variant"],
                "band": row["band"],
                "status": row["status"],
                "n_channels": "",
                "n_subjects": row["n_subjects"],
                "output_path": row["output_path"],
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    channel_values = pd.DataFrame(channel_rows)

    manifest_path = PROJECT_ROOT / "outputs" / "qc" / "tmt_topomap_manifest.csv"
    channel_values_path = (
        PROJECT_ROOT
        / "outputs"
        / "tmt"
        / "csv"
        / "tmt_topomap_channel_relative_power.csv"
    )

    write_dataframe(manifest, manifest_path)
    write_dataframe(channel_values, channel_values_path)

    print("")
    print("TMT topomap grafikleri tamamlandı.")
    print(f"Manifest dosyası: {manifest_path}")
    print(f"Kanal değerleri: {channel_values_path}")
    print("")
    print("Durum özeti:")
    print(manifest["status"].value_counts(dropna=False))

    created = manifest[manifest["status"] == "created"]

    print("")
    print(f"Oluşturulan topomap sayısı: {len(created)}")

    if not created.empty:
        print(
            created[
                ["variant", "group", "band", "n_subjects", "output_path"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()