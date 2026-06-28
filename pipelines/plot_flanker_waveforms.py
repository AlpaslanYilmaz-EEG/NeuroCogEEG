"""
Flanker waveform figure pipeline for NeuroCogEEG.

This script creates group-level grand-average ROI waveform figures for Flanker.

It reuses the validated preprocessing and epoching logic. It does not compute
new CSV outcomes and does not modify existing result files.

Outputs:
outputs/flanker/figures/waveforms/
outputs/qc/flanker_waveform_manifest.csv
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
    filter_event_sequence,
    get_reject_criteria,
)
from neurocogeeg.preprocessing import (
    load_device_config,
    load_experiment_config,
    preprocess_emotiv_edf,
)


EXPERIMENT = "flanker"


def create_stimulus_locked_correct_epochs(raw_clean, events, experiment_config):
    events_config = experiment_config["events"]

    stimulus_code = get_event_code(
        events_config=events_config,
        event_name="stimulus",
    )
    correct_code = get_event_code(
        events_config=events_config,
        event_name="correct_response",
    )

    correct_stimulus_events = filter_event_sequence(
        events=events,
        first_event_code=stimulus_code,
        next_event_code=correct_code,
    )

    epoch_config = experiment_config["epochs"]["stimulus_locked"]
    reject_criteria = get_reject_criteria(
        experiment_config["quality_control"]
    )

    return create_epochs(
        raw=raw_clean,
        events=correct_stimulus_events,
        event_id={"stimulus": stimulus_code},
        epoch_config=epoch_config,
        reject_criteria=reject_criteria,
        picks="eeg",
    )


def create_response_locked_epochs(raw_clean, events, experiment_config):
    events_config = experiment_config["events"]

    correct_code = get_event_code(
        events_config=events_config,
        event_name="correct_response",
    )
    error_code = get_event_code(
        events_config=events_config,
        event_name="error_response",
    )

    epoch_config = experiment_config["epochs"]["response_locked"]
    reject_criteria = get_reject_criteria(
        experiment_config["quality_control"]
    )

    return create_epochs(
        raw=raw_clean,
        events=events,
        event_id={
            "correct_response": correct_code,
            "error_response": error_code,
        },
        epoch_config=epoch_config,
        reject_criteria=reject_criteria,
        picks="eeg",
    )


def extract_roi_waveform_uv(epochs, condition, roi_channels):
    if len(epochs) == 0:
        return None, None

    if condition not in epochs.event_id:
        return None, None

    try:
        condition_epochs = epochs[condition]
    except Exception:
        return None, None

    if len(condition_epochs) == 0:
        return None, None

    available_channels = [
        channel for channel in roi_channels
        if channel in condition_epochs.ch_names
    ]

    if not available_channels:
        return None, None

    roi_epochs = condition_epochs.copy().pick(available_channels)

    data_uv = roi_epochs.get_data() * 1e6
    waveform_uv = data_uv.mean(axis=0).mean(axis=0)

    return roi_epochs.times.copy(), waveform_uv


def get_plot_specs(experiment_config):
    n2 = experiment_config["erp_components"]["n2"]
    p3 = experiment_config["erp_components"]["p3"]
    rp = experiment_config["response_locked"]["rp"]
    pmp = experiment_config["response_locked"]["pmp"]

    return [
        {
            "name": "stimulus_locked_n2_roi",
            "epoch_type": "stimulus_locked",
            "condition": n2["condition"],
            "roi": n2["roi"],
            "window": n2["window"],
            "title": "Flanker stimulus-locked N2 ROI",
            "filename": "flanker_stimulus_locked_n2_roi.png",
        },
        {
            "name": "stimulus_locked_p3_roi",
            "epoch_type": "stimulus_locked",
            "condition": p3["condition"],
            "roi": p3["roi"],
            "window": p3["window"],
            "title": "Flanker stimulus-locked P3 ROI",
            "filename": "flanker_stimulus_locked_p3_roi.png",
        },
        {
            "name": "response_locked_rp_roi",
            "epoch_type": "response_locked",
            "condition": rp["condition"],
            "roi": rp["roi"],
            "window": rp["window"],
            "title": "Flanker response-locked RP ROI",
            "filename": "flanker_response_locked_rp_roi.png",
        },
        {
            "name": "response_locked_pmp_roi",
            "epoch_type": "response_locked",
            "condition": pmp["condition"],
            "roi": pmp["roi"],
            "window": pmp["window"],
            "title": "Flanker response-locked PMP ROI",
            "filename": "flanker_response_locked_pmp_roi.png",
        },
    ]


def initialize_waveform_store(plot_specs):
    store = {}

    for spec in plot_specs:
        store[spec["name"]] = {
            "control": [],
            "experimental": [],
            "times": None,
        }

    return store


def add_subject_waveforms_to_store(
    waveform_store,
    subject,
    stimulus_epochs,
    response_epochs,
    plot_specs,
):
    rows = []

    for spec in plot_specs:
        if spec["epoch_type"] == "stimulus_locked":
            epochs = stimulus_epochs
        else:
            epochs = response_epochs

        times, waveform = extract_roi_waveform_uv(
            epochs=epochs,
            condition=spec["condition"],
            roi_channels=spec["roi"],
        )

        if waveform is None:
            rows.append(
                {
                    "subject_id": subject.subject_id,
                    "group": subject.group,
                    "plot_name": spec["name"],
                    "status": "skipped_no_waveform",
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
                    "plot_name": spec["name"],
                    "status": "skipped_time_length_mismatch",
                }
            )
            continue

        store_entry[subject.group].append(waveform)

        rows.append(
            {
                "subject_id": subject.subject_id,
                "group": subject.group,
                "plot_name": spec["name"],
                "status": "added",
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
    print(f"İşleniyor: {subject.subject_id} ({subject.group})")

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

        stimulus_epochs = create_stimulus_locked_correct_epochs(
            raw_clean=raw_clean,
            events=events,
            experiment_config=experiment_config,
        )

        response_epochs = create_response_locked_epochs(
            raw_clean=raw_clean,
            events=events,
            experiment_config=experiment_config,
        )

        return add_subject_waveforms_to_store(
            waveform_store=waveform_store,
            subject=subject,
            stimulus_epochs=stimulus_epochs,
            response_epochs=response_epochs,
            plot_specs=plot_specs,
        )

    except Exception as error:
        return [
            {
                "subject_id": subject.subject_id,
                "group": subject.group,
                "plot_name": "all",
                "status": "processing_error",
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

    fig, ax = plt.subplots(figsize=(9, 5))

    plotted_groups = []

    for group in ["control", "experimental"]:
        waveforms = store_entry[group]

        mean, sem = compute_mean_and_sem(waveforms)

        if mean is None:
            continue

        ax.plot(times, mean, label=f"{group} (n={len(waveforms)})")
        ax.fill_between(times, mean - sem, mean + sem, alpha=0.2)

        plotted_groups.append(group)

    if not plotted_groups:
        plt.close(fig)
        return {
            "plot_name": spec["name"],
            "status": "skipped_no_group_data",
            "output_path": "",
        }

    window = spec["window"]
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.axvspan(window["start"], window["end"], alpha=0.12)

    ax.set_title(spec["title"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

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
    output_dir = PROJECT_ROOT / "outputs" / "flanker" / "figures" / "waveforms"

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

    output_path = PROJECT_ROOT / "outputs" / "qc" / "flanker_waveform_manifest.csv"
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
        print("Flanker EDF dosyası bulunamadı.")
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
                "plot_name": row["plot_name"],
                "status": row["status"],
                "output_path": row["output_path"],
            }
        )

    manifest_path, manifest = write_manifest(manifest_rows)

    print("")
    print("Flanker waveform grafikleri tamamlandı.")
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