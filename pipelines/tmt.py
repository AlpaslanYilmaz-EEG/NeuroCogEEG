"""
Trail Making Test analysis pipeline for NeuroCogEEG.

This pipeline performs:

- EDF preprocessing
- event extraction
- behavioral TMT analysis
- continuous EEG PSD analysis
- early/late PSD comparison
- response-locked RP/PMP analysis
- quality-control summaries
- SPSS-compatible CSV export

Numeric event values are never hard-coded. Event meanings are read from
configs/tmt.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurocogeeg.behavior import (  # noqa: E402
    compute_inter_event_intervals_seconds,
    compute_task_duration_seconds,
    count_events_from_config,
    get_event_code,
    summarize_intervals_seconds,
)
from neurocogeeg.dataset import SubjectRecord, get_dataset  # noqa: E402
from neurocogeeg.epochs import (  # noqa: E402
    create_epochs,
    extract_events_from_marker_channel,
    get_reject_criteria,
)
from neurocogeeg.export import (  # noqa: E402
    add_subject_metadata,
    merge_result_dicts,
    save_config_snapshot,
    save_spss_csv,
)
from neurocogeeg.preprocessing import (  # noqa: E402
    load_device_config,
    load_experiment_config,
    preprocess_emotiv_edf,
)
from neurocogeeg.psd import (  # noqa: E402
    compute_early_late_psd_summary,
    compute_roi_psd_summary,
    crop_raw_segment,
)
from neurocogeeg.qc import (  # noqa: E402
    merge_qc_summaries,
    summarize_event_counts,
    summarize_epochs,
    summarize_epochs_by_condition,
    summarize_minimum_trial_requirement,
    summarize_preprocessing_info,
    summarize_raw_info,
)
from neurocogeeg.rp import compute_response_locked_metrics_from_epochs  # noqa: E402


EXPERIMENT = "tmt"


def get_task_segment_samples(
    events: np.ndarray,
    experiment_config: dict[str, Any],
) -> tuple[int | None, int | None]:
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
        Start sample and end sample. Returns ``(None, None)`` if the segment
        cannot be identified.
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


def get_task_segment_times(
    events: np.ndarray,
    sfreq: float,
    experiment_config: dict[str, Any],
) -> tuple[float | None, float | None]:
    """
    Return TMT task start and end times in seconds.

    Parameters
    ----------
    events:
        MNE events array.

    sfreq:
        Sampling frequency in Hz.

    experiment_config:
        Loaded tmt.yaml configuration.

    Returns
    -------
    tuple[float | None, float | None]
        Start and end time in seconds.
    """
    start_sample, end_sample = get_task_segment_samples(
        events=events,
        experiment_config=experiment_config,
    )

    if start_sample is None or end_sample is None:
        return None, None

    return float(start_sample / sfreq), float(end_sample / sfreq)


def empty_psd_results(
    psd_config: dict[str, Any],
) -> dict[str, float]:
    """
    Create NaN-filled PSD result columns.

    Parameters
    ----------
    psd_config:
        PSD section from tmt.yaml.

    Returns
    -------
    dict[str, float]
        NaN-filled PSD result dictionary.
    """
    suffix = (
        "relative_percent"
        if psd_config.get("relative_power", True)
        else "absolute_power"
    )

    results: dict[str, float] = {}

    for roi_name in psd_config["rois"]:
        for band_name in psd_config["frequency_bands"]:
            results[f"{roi_name}_{band_name}_{suffix}"] = np.nan

    return results


def empty_early_late_psd_results(
    psd_config: dict[str, Any],
) -> dict[str, float]:
    """
    Create NaN-filled early/late PSD result columns.

    Parameters
    ----------
    psd_config:
        PSD section from tmt.yaml.

    Returns
    -------
    dict[str, float]
        NaN-filled early/late PSD result dictionary.
    """
    base_results = empty_psd_results(psd_config)
    results: dict[str, float] = {}

    for key in base_results:
        results[f"early_{key}"] = np.nan
        results[f"late_{key}"] = np.nan
        results[f"change_{key}"] = np.nan

    return results


def empty_rp_results() -> dict[str, float | int]:
    """
    Create NaN-filled RP/PMP result columns.

    Returns
    -------
    dict[str, float | int]
        Empty RP/PMP metrics.
    """
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


def compute_behavioral_results(
    events: np.ndarray,
    sfreq: float,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute TMT behavioral outcomes.

    Parameters
    ----------
    events:
        MNE events array.

    sfreq:
        Sampling frequency.

    experiment_config:
        Loaded tmt.yaml configuration.

    Returns
    -------
    dict[str, Any]
        Behavioral result dictionary.
    """
    events_config = experiment_config["events"]
    behavior_config = experiment_config["behavior"]

    event_counts = count_events_from_config(
        events=events,
        events_config=events_config,
    )

    duration_config = behavior_config["task_duration"]

    start_code = get_event_code(
        events_config=events_config,
        event_name=duration_config["start_event"],
    )

    end_codes = [
        get_event_code(
            events_config=events_config,
            event_name=end_event,
        )
        for end_event in duration_config["end_events"]
    ]

    task_duration_s = compute_task_duration_seconds(
        events=events,
        start_code=start_code,
        end_codes=end_codes,
        sfreq=sfreq,
    )

    clicks_config = behavior_config["clicks"]

    correct_click_code = get_event_code(
        events_config=events_config,
        event_name=clicks_config["correct_event"],
    )

    error_click_code = get_event_code(
        events_config=events_config,
        event_name=clicks_config["error_event"],
    )

    correct_click_count = int(np.sum(events[:, 2] == correct_click_code))
    error_click_count = int(np.sum(events[:, 2] == error_click_code))
    total_click_count = correct_click_count + error_click_count

    if total_click_count == 0:
        error_percent = np.nan
    else:
        error_percent = float((error_click_count / total_click_count) * 100.0)

    interval_config = behavior_config["inter_click_interval"]

    interval_codes = [
        get_event_code(
            events_config=events_config,
            event_name=event_name,
        )
        for event_name in interval_config["events"]
    ]

    intervals_s = compute_inter_event_intervals_seconds(
        events=events,
        event_codes=interval_codes,
        sfreq=sfreq,
    )

    interval_summary = summarize_intervals_seconds(
        intervals_seconds=intervals_s,
        prefix="inter_click_interval",
    )

    return {
        **event_counts,
        "task_duration_s": task_duration_s,
        "correct_click_count": correct_click_count,
        "error_click_count": error_click_count,
        "total_click_count": total_click_count,
        "error_percent": error_percent,
        **interval_summary,
    }


def create_response_locked_epochs(
    raw_clean,
    events: np.ndarray,
    experiment_config: dict[str, Any],
):
    """
    Create response-locked epochs around correct TMT clicks.

    Parameters
    ----------
    raw_clean:
        Preprocessed raw object.

    events:
        MNE events array.

    experiment_config:
        Loaded tmt.yaml configuration.

    Returns
    -------
    mne.Epochs
        Response-locked epochs.
    """
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


def compute_psd_results(
    raw_clean,
    events: np.ndarray,
    sfreq: float,
    experiment_config: dict[str, Any],
) -> dict[str, float]:
    """
    Compute continuous TMT PSD results.

    Parameters
    ----------
    raw_clean:
        Preprocessed raw object.

    events:
        MNE events array.

    sfreq:
        Sampling frequency.

    experiment_config:
        Loaded tmt.yaml configuration.

    Returns
    -------
    dict[str, float]
        PSD result dictionary.
    """
    psd_config = experiment_config["psd"]

    start_time, end_time = get_task_segment_times(
        events=events,
        sfreq=sfreq,
        experiment_config=experiment_config,
    )

    if start_time is None or end_time is None:
        return empty_psd_results(psd_config)

    try:
        task_raw = crop_raw_segment(
            raw=raw_clean,
            start_time=start_time,
            end_time=end_time,
        )

        return compute_roi_psd_summary(
            raw=task_raw,
            roi_config=psd_config["rois"],
            frequency_bands=psd_config["frequency_bands"],
            total_band_config=psd_config["total_band"],
            relative=psd_config.get("relative_power", True),
            n_fft_seconds=psd_config.get("n_fft_seconds", 2.0),
        )

    except Exception:
        return empty_psd_results(psd_config)


def compute_early_late_results(
    raw_clean,
    events: np.ndarray,
    sfreq: float,
    experiment_config: dict[str, Any],
) -> dict[str, float]:
    """
    Compute early/late PSD comparison results.

    Parameters
    ----------
    raw_clean:
        Preprocessed raw object.

    events:
        MNE events array.

    sfreq:
        Sampling frequency.

    experiment_config:
        Loaded tmt.yaml configuration.

    Returns
    -------
    dict[str, float]
        Early/late PSD result dictionary.
    """
    psd_config = experiment_config["psd"]
    early_late_config = experiment_config.get("early_late_psd", {})

    if not early_late_config.get("enabled", False):
        return empty_early_late_psd_results(psd_config)

    start_time, end_time = get_task_segment_times(
        events=events,
        sfreq=sfreq,
        experiment_config=experiment_config,
    )

    if start_time is None or end_time is None:
        return empty_early_late_psd_results(psd_config)

    try:
        return compute_early_late_psd_summary(
            raw=raw_clean,
            start_time=start_time,
            end_time=end_time,
            roi_config=psd_config["rois"],
            frequency_bands=psd_config["frequency_bands"],
            total_band_config=psd_config["total_band"],
            relative=psd_config.get("relative_power", True),
            n_fft_seconds=psd_config.get("n_fft_seconds", 2.0),
        )

    except Exception:
        return empty_early_late_psd_results(psd_config)


def compute_response_locked_results(
    response_epochs,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute RP and PMP from response-locked TMT epochs.

    Parameters
    ----------
    response_epochs:
        Response-locked epochs.

    experiment_config:
        Loaded tmt.yaml configuration.

    Returns
    -------
    dict[str, Any]
        RP and PMP results.
    """
    rp_config = experiment_config["response_locked"]["rp"]
    pmp_config = experiment_config["response_locked"]["pmp"]

    return compute_response_locked_metrics_from_epochs(
        epochs=response_epochs,
        roi_channels=rp_config["roi"],
        rp_window_config=rp_config["window"],
        pmp_window_config=pmp_config["window"],
        condition=rp_config["condition"],
    )


def process_subject(
    subject: SubjectRecord,
    device_config: dict[str, Any],
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Process one TMT participant.

    Parameters
    ----------
    subject:
        Subject record.

    device_config:
        Loaded Emotiv device configuration.

    experiment_config:
        Loaded TMT experiment configuration.

    Returns
    -------
    dict[str, Any]
        One SPSS-compatible result row.
    """
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

        sfreq = raw_clean.info["sfreq"]

        behavioral_results = compute_behavioral_results(
            events=events,
            sfreq=sfreq,
            experiment_config=experiment_config,
        )

        event_id = {
            name: int(code)
            for name, code in experiment_config["events"].items()
        }

        qc_event_results = summarize_event_counts(
            events=events,
            event_id=event_id,
        )

        qc_preprocessing_results = summarize_preprocessing_info(
            preprocessing_info
        )

        qc_raw_results = summarize_raw_info(raw_clean)

        psd_results = compute_psd_results(
            raw_clean=raw_clean,
            events=events,
            sfreq=sfreq,
            experiment_config=experiment_config,
        )

        early_late_results = compute_early_late_results(
            raw_clean=raw_clean,
            events=events,
            sfreq=sfreq,
            experiment_config=experiment_config,
        )

        response_epochs = create_response_locked_epochs(
            raw_clean=raw_clean,
            events=events,
            experiment_config=experiment_config,
        )

        qc_response_epochs = summarize_epochs(
            response_epochs,
            prefix="response_locked_epochs",
        )

        qc_response_conditions = summarize_epochs_by_condition(
            response_epochs,
            prefix="response_condition",
        )

        minimum_trials = experiment_config["quality_control"][
            "minimum_trials"
        ]

        response_minimum = summarize_minimum_trial_requirement(
            observed_count=len(response_epochs),
            minimum_count=minimum_trials["response_locked"],
            label="response_locked_trials",
        )

        if len(response_epochs) >= minimum_trials["response_locked"]:
            response_results = compute_response_locked_results(
                response_epochs=response_epochs,
                experiment_config=experiment_config,
            )
        else:
            response_results = empty_rp_results()

        qc_results = merge_qc_summaries(
            qc_event_results,
            qc_preprocessing_results,
            qc_raw_results,
            qc_response_epochs,
            qc_response_conditions,
            response_minimum,
        )

        result_row = merge_result_dicts(
            behavioral_results,
            psd_results,
            early_late_results,
            response_results,
            qc_results,
        )

        return add_subject_metadata(
            result_row=result_row,
            subject_id=subject.subject_id,
            group=subject.group,
            experiment=subject.experiment,
        )

    except Exception as error:
        print(f"❌ Hata: {subject.subject_id}: {error}")

        return add_subject_metadata(
            result_row={
                "processing_error": str(error),
            },
            subject_id=subject.subject_id,
            group=subject.group,
            experiment=subject.experiment,
        )


def run_pipeline() -> Path | None:
    """
    Run the TMT pipeline for all available participants.

    Returns
    -------
    Path | None
        Path to the saved CSV file, or None if no subjects were found.
    """
    dataset = get_dataset(EXPERIMENT)
    dataset.ensure_output_dirs()

    device_config = load_device_config()
    experiment_config = load_experiment_config(EXPERIMENT)

    save_config_snapshot(
        config={
            "device": device_config,
            "experiment": experiment_config,
        },
        output_path=dataset.reports_dir / "used_tmt_config.yaml",
    )

    subjects = dataset.list_subjects()

    if not subjects:
        print("Hiç EDF dosyası bulunamadı.")
        print("Beklenen klasörler:")
        print(dataset.group_raw_dir("control"))
        print(dataset.group_raw_dir("experimental"))
        return None

    rows: list[dict[str, Any]] = []

    for subject in subjects:
        rows.append(
            process_subject(
                subject=subject,
                device_config=device_config,
                experiment_config=experiment_config,
            )
        )

    output_path = dataset.csv_dir / "tmt_results.csv"

    save_spss_csv(
        rows=rows,
        output_path=output_path,
    )

    print(f"Sonuç dosyası kaydedildi: {output_path}")

    return output_path


if __name__ == "__main__":
    run_pipeline()