"""
TMT waveform figure pipeline for NeuroCogEEG.

This script creates group- and variant-level grand-average ROI waveform figures
for TMT response-locked RP/PMP analyses.

It reuses the validated preprocessing and epoching logic. It does not compute
new CSV outcomes and does not modify existing result files.

Outputs:
outputs/tmt/figures/waveforms/
outputs/qc/tmt_waveform_manifest.csv
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from neurocogeeg.behavior import get_event_code
from neurocogeeg.dataset import SubjectRecord, get_dataset
from neurocogeeg.epochs import (
    create_epochs,
    extract_events_from_marker_channel,
    get_reject_criteria,
)
from neurocogeeg.preprocessing import (
    load_device_config,
    load_experiment_config,
    preprocess_emotiv_edf,
)


EXPERIMENT = "tmt"


def create_response_locked_epochs(raw_clean, events, experiment_config):
    events_config = experiment_config["events"]

    correct_click_code = get_event_code(
        events_config=events_config,
        event_name="correct_click",
    )

    epoch_config = experiment_config["epochs"]["response_locked"]
    reject_criteria = get_reject_criteria(
        experiment_config["quality_control"]
    )

    return create_epochs(
        raw=raw_clean,
        events=events,
        event_id={"correct_click": correct_click_code},
        epoch_config=epoch_config,
        reject_criteria=reject_criteria,
        picks="eeg",
    )


def extract_roi_waveform_uv(
    epochs,
    condition,
    roi_channels,
    minimum_trials,
):
    if len(epochs) < minimum_trials:
        return None, None, "skipped_minimum_trials_fail"

    if condition not in epochs.event_id:
        return None, None, "skipped_condition_missing"

    try:
        condition_epochs = epochs[condition]
    except Exception:
        return None, None, "skipped_condition_error"

    if len(condition_epochs) < minimum_trials:
        return None, None, "skipped_minimum_trials_fail"

    available_channels = [
        channel for channel in roi_channels
        if channel in condition_epochs.ch_names
    ]

    if not available_channels:
        return None, None, "skipped_no_roi_channels"

    roi_epochs = condition_epochs.copy().pick(available_channels)

    data_uv = roi_epochs.get_data() * 1e6
    waveform_uv = data_uv.mean(axis=0).mean(axis=0)

    return roi_epochs.times.copy(), waveform_uv, "added"


def get_plot_specs(experiment_config):
    rp = experiment_config["response_locked"]["rp"]
    pmp = experiment_config["response_locked"]["pmp"]

    return [
        {
            "name": "response_locked_rp_roi",
            "condition": rp["condition"],
            "roi": rp["roi"],
            "window": rp["window"],
            "title": "TMT response-locked RP ROI",
            "filename": "tmt_response_locked_rp_roi.png",
        },
        {
            "name": "response_locked_pmp_roi",
            "condition": pmp["condition"],
            "roi": pmp["roi"],
            "window": pmp["window"],
            "title": "TMT response-locked PMP ROI",
            "filename": "tmt_response_locked_pmp_roi.png",
        },
    ]


def get_series_labels():
    return [
        "control / tmt1",
        "control / tmt2",
        "experimental / tmt1",
        "experimental / tmt2",
    ]


def get_subject_series_label(subject):
    return f"{subject.group} / {subject.variant}"


def initialize_waveform_store(plot_specs):
    store = {}

    for spec in plot_specs:
        store[spec["name"]] = {
            "series": {
                label: []
                for label in get_series_labels()
            },
            "times": None,
        }

    return store


def add_subject_waveforms_to_store(
    waveform_store,
    subject,
    response_epochs,
    plot_specs,
    minimum_trials,
):
    rows = []
    series_label = get_subject_series_label(subject)

    for spec in plot_specs:
        times, waveform, status = extract_roi_waveform_uv(
            epochs=response_epochs,
            condition=spec["condition"],
            roi_channels=spec["roi"],
            minimum_trials=minimum_trials,
        )

        if waveform is None:
            rows.append(
                {
                    "subject_id": subject.subject_id,
                    "group": subject.group,
                    "variant": subject.variant,
                    "plot_name": spec["name"],
                    "status": status,
                    "epoch_count": len(response_epochs),
                }
            )
            continue

        store_entry = waveform_store[spec["name"]]

        if store_entry["times"] is None:
            store_entry["times"] = times

        if len(times) != len(store_entry["times"]):
            rows.append(
                {
                    "subject_id": subject.subject_id,
                    "group": subject.group,
                    "variant": subject.variant,
                    "plot_name": spec["name"],
                    "status": "skipped_time_length_mismatch",
                    "epoch_count": len(response_epochs),
                }
            )
            continue

        if series_label not in store_entry["series"]:
            rows.append(
                {
                    "subject_id": subject.subject_id,
                    "group": subject.group,
                    "variant": subject.variant,
                    "plot_name": spec["name"],
                    "status": "skipped_unexpected_series_label",
                    "epoch_count": len(response_epochs),
                }
            )
            continue

        store_entry["series"][series_label].append(waveform)

        rows.append(
            {
                "subject_id": subject.subject_id,
                "group": subject.group,
                "variant": subject.variant,
                "plot_name": spec["name"],
                "status": "added",
                "epoch_count": len(response_epochs),
            }
        )

    return rows


def process_subject(
    subject: SubjectRecord,
    device_config,
    experiment_config,
    waveform_store,
    plot_specs,
):
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

        response_epochs = create_response_locked_epochs(
            raw_clean=raw_clean,
            events=events,
            experiment_config=experiment_config,
        )

        minimum_trials = experiment_config["quality_control"][
            "minimum_trials"
        ]["response_locked"]

        return add_subject_waveforms_to_store(
            waveform_store=waveform_store,
            subject=subject,
            response_epochs=response_epochs,
            plot_specs=plot_specs,
            minimum_trials=minimum_trials,
        )

    except Exception as error:
        return [
            {
                "subject_id": subject.subject_id,
                "group": subject.group,
                "variant": subject.variant,
                "plot_name": "all",
                "status": "processing_error",
                "epoch_count": "",
                "error": str(error),
            }
        ]


def compute_mean_and_sem(waveforms):
    array = np.asarray(waveforms, dtype=float)

    if array.size == 0:
        return None, None

    mean = np.nanmean(array, axis=0)

    if array.shape[0] <= 1:
        sem = np.zeros_like(mean)
    else:
        sem = np.nanstd(array, axis=0, ddof=1) / np.sqrt(array.shape[0])

    return mean, sem


def save_waveform_plot(spec, store_entry, output_dir):
    times = store_entry["times"]

    if times is None:
        return {
            "plot_name": spec["name"],
            "status": "skipped_no_time_axis",
            "output_path": "",
        }

    fig, ax = plt.subplots(figsize=(10, 5))

    plotted_series = []

    for series_label in get_series_labels():
        waveforms = store_entry["series"][series_label]

        mean, sem = compute_mean_and_sem(waveforms)

        if mean is None:
            continue

        ax.plot(times, mean, label=f"{series_label} (n={len(waveforms)})")
        ax.fill_between(times, mean - sem, mean + sem, alpha=0.2)

        plotted_series.append(series_label)

    if not plotted_series:
        plt.close(fig)
        return {
            "plot_name": spec["name"],
            "status": "skipped_no_series_data",
            "output_path": "",
        }

    window = spec["window"]
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.axvspan(window["start"], window["end"], alpha=0.12)

    ax.set_title(spec["title"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()

    output_path = output_dir / spec["filename"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return {
        "plot_name": spec["name"],
        "status": "created",
        "output_path": str(output_path),
    }


def save_all_waveform_plots(waveform_store, plot_specs):
    output_dir = PROJECT_ROOT / "outputs" / "tmt" / "figures" / "waveforms"

    rows = []

    for spec in plot_specs:
        rows.append(
            save_waveform_plot(
                spec=spec,
                store_entry=waveform_store[spec["name"]],
                output_dir=output_dir,
            )
        )

    return rows


def write_manifest(rows):
    manifest = pd.DataFrame(rows)

    output_path = PROJECT_ROOT / "outputs" / "qc" / "tmt_waveform_manifest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(
        output_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )

    return output_path, manifest


def main():
    dataset = get_dataset(EXPERIMENT)
    device_config = load_device_config()
    experiment_config = load_experiment_config(EXPERIMENT)

    subjects = dataset.list_subjects()

    if not subjects:
        print("TMT EDF dosyası bulunamadı.")
        return

    plot_specs = get_plot_specs(experiment_config)
    waveform_store = initialize_waveform_store(plot_specs)

    manifest_rows = []

    for subject in subjects:
        manifest_rows.extend(
            process_subject(
                subject=subject,
                device_config=device_config,
                experiment_config=experiment_config,
                waveform_store=waveform_store,
                plot_specs=plot_specs,
            )
        )

    plot_rows = save_all_waveform_plots(
        waveform_store=waveform_store,
        plot_specs=plot_specs,
    )

    for row in plot_rows:
        manifest_rows.append(
            {
                "subject_id": "",
                "group": "",
                "variant": "",
                "plot_name": row["plot_name"],
                "status": row["status"],
                "epoch_count": "",
                "output_path": row["output_path"],
            }
        )

    manifest_path, manifest = write_manifest(manifest_rows)

    print("")
    print("TMT waveform grafikleri tamamlandı.")
    print(f"Manifest dosyası: {manifest_path}")
    print("")
    print("Durum özeti:")
    print(manifest["status"].value_counts(dropna=False))

    created = manifest[manifest["status"] == "created"]

    print("")
    print(f"Oluşturulan grafik sayısı: {len(created)}")

    if not created.empty:
        print(created[["plot_name", "output_path"]].to_string(index=False))


if __name__ == "__main__":
    main()