"""
Check response-marker sequence anomalies for NeuroCogEEG non-TMT tasks.

Version note: Go/No-Go uses false_alarm (marker 53) rather than incorrect_response.

This diagnostic script scans Flanker, Go/No-Go and ReadySetGo EDF marker
sequences and asks three practical questions:

1. Are there consecutive response markers after the current de-duplication?
2. Are there response markers without an immediately preceding task stimulus?
3. Would such response markers be response-locked epoch candidates before
   artifact/boundary rejection?

It does not modify raw EDF files or pipeline outputs.

Run from the project root:
    python pipelines/check_non_tmt_response_marker_sequences.py

Outputs:
    outputs/qc/non_tmt_response_marker_sequence_qc.csv
    outputs/qc/non_tmt_response_marker_sequence_details.csv
    outputs/qc/non_tmt_response_marker_duplicate_details.csv
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurocogeeg.dataset import SubjectRecord, get_dataset  # noqa: E402
from neurocogeeg.epochs import deduplicate_close_repeated_events  # noqa: E402
from neurocogeeg.preprocessing import (  # noqa: E402
    load_device_config,
    load_experiment_config,
    preprocess_emotiv_edf,
)


@dataclass(frozen=True)
class TaskRule:
    experiment: str
    response_event_names: tuple[str, ...]
    generic_stimulus_event_names: tuple[str, ...]
    valid_preceding_stimulus_names_by_response: dict[str, tuple[str, ...]]


TASK_RULES: dict[str, TaskRule] = {
    "flanker": TaskRule(
        experiment="flanker",
        response_event_names=("correct_response", "error_response"),
        generic_stimulus_event_names=("stimulus",),
        valid_preceding_stimulus_names_by_response={
            "correct_response": ("stimulus",),
            "error_response": ("stimulus",),
        },
    ),
    "gonogo": TaskRule(
        experiment="gonogo",
        response_event_names=("correct_response", "false_alarm"),
        generic_stimulus_event_names=("go_stimulus", "nogo_stimulus"),
        valid_preceding_stimulus_names_by_response={
            "correct_response": ("go_stimulus",),
            "false_alarm": ("nogo_stimulus",),
        },
    ),
    "readysetgo": TaskRule(
        experiment="readysetgo",
        response_event_names=("go_response",),
        generic_stimulus_event_names=("go",),
        valid_preceding_stimulus_names_by_response={
            "go_response": ("go",),
        },
    ),
}


def extract_events_without_dedup(raw, marker_channel: str, marker_config: dict[str, Any]) -> np.ndarray:
    """Mirror the project marker extraction, but keep all repeated markers."""
    if marker_channel not in raw.ch_names:
        raise ValueError(f"Marker channel not found in raw: {marker_channel}")

    scale_factor = marker_config.get("scale_factor", 1)
    zero_value = marker_config.get("zero_value", 0)

    marker_data = raw.copy().pick(picks=[marker_channel], verbose=False).get_data()[0]
    marker_values = np.round(marker_data * scale_factor).astype(int)
    marker_changes = np.diff(marker_values)
    event_indices = np.where(np.abs(marker_changes) > 0)[0] + 1

    if event_indices.size == 0:
        return np.empty((0, 3), dtype=int)

    events = np.column_stack(
        (
            event_indices,
            np.zeros_like(event_indices),
            marker_values[event_indices],
        )
    ).astype(int)

    events = events[events[:, 2] != int(zero_value)]
    return events.astype(int)


def code_lookup(events_config: dict[str, Any]) -> dict[str, int]:
    return {name: int(code) for name, code in events_config.items()}


def names_by_code(events_config: dict[str, Any]) -> dict[int, str]:
    mapping: dict[int, list[str]] = {}
    for name, code in events_config.items():
        mapping.setdefault(int(code), []).append(str(name))
    return {code: "/".join(names) for code, names in mapping.items()}


def safe_event_name(code: int | None, lookup: dict[int, str]) -> str:
    if code is None:
        return ""
    return lookup.get(int(code), f"code_{int(code)}")


def event_bounds_ok(sample: int, sfreq: float, n_times: int, tmin: float, tmax: float) -> bool:
    start = int(round(sample + tmin * sfreq))
    stop = int(round(sample + tmax * sfreq))
    return start >= 0 and stop < n_times and stop > start


def get_response_epoch_window(experiment_config: dict[str, Any]) -> tuple[float, float] | None:
    epochs_config = experiment_config.get("epochs", {})
    response_config = epochs_config.get("response_locked")
    if not response_config:
        return None
    return float(response_config["tmin"]), float(response_config["tmax"])


def make_detail_row(
    *,
    experiment: str,
    subject: SubjectRecord,
    anomaly_type: str,
    event_index: int,
    event: np.ndarray,
    previous_event: np.ndarray | None,
    next_event: np.ndarray | None,
    sfreq: float,
    name_lookup: dict[int, str],
    would_be_epoch_candidate: bool | None = None,
) -> dict[str, Any]:
    event_sample = int(event[0])
    event_code = int(event[2])

    prev_sample = int(previous_event[0]) if previous_event is not None else None
    prev_code = int(previous_event[2]) if previous_event is not None else None
    next_sample = int(next_event[0]) if next_event is not None else None
    next_code = int(next_event[2]) if next_event is not None else None

    return {
        "experiment": experiment,
        "subject_id": subject.subject_id,
        "group": subject.group,
        "anomaly_type": anomaly_type,
        "event_index_post_dedup": int(event_index),
        "event_sample": event_sample,
        "event_time_s": float(event_sample / sfreq),
        "event_code": event_code,
        "event_name": safe_event_name(event_code, name_lookup),
        "previous_sample": prev_sample,
        "previous_code": prev_code,
        "previous_name": safe_event_name(prev_code, name_lookup),
        "previous_delta_ms": float(((event_sample - prev_sample) / sfreq) * 1000.0)
        if prev_sample is not None
        else np.nan,
        "next_sample": next_sample,
        "next_code": next_code,
        "next_name": safe_event_name(next_code, name_lookup),
        "next_delta_ms": float(((next_sample - event_sample) / sfreq) * 1000.0)
        if next_sample is not None
        else np.nan,
        "would_be_response_locked_epoch_candidate_pre_rejection": would_be_epoch_candidate,
    }


def analyze_subject(
    subject: SubjectRecord,
    task_rule: TaskRule,
    device_config: dict[str, Any],
    experiment_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    experiment = task_rule.experiment

    raw_clean, _ica, preprocessing_info = preprocess_emotiv_edf(
        edf_path=subject.edf_path,
        experiment=experiment,
    )

    sfreq = float(raw_clean.info["sfreq"])
    n_times = int(raw_clean.n_times)
    marker_channel = preprocessing_info["marker_channel"]
    marker_config = device_config["marker_channels"]
    duplicate_window_ms = float(marker_config.get("duplicate_window_ms", 100))

    events_raw = extract_events_without_dedup(
        raw=raw_clean,
        marker_channel=marker_channel,
        marker_config=marker_config,
    )

    events_post, n_dropped_total = deduplicate_close_repeated_events(
        events=events_raw,
        sfreq=sfreq,
        duplicate_window_ms=duplicate_window_ms,
    )

    events_config = experiment_config["events"]
    code_by_name = code_lookup(events_config)
    name_by_event_code = names_by_code(events_config)

    missing_response_names = [
        name for name in task_rule.response_event_names if name not in code_by_name
    ]
    missing_stimulus_names = [
        name for name in task_rule.generic_stimulus_event_names if name not in code_by_name
    ]
    if missing_response_names or missing_stimulus_names:
        raise KeyError(
            f"Missing event names for {experiment}: "
            f"responses={missing_response_names}, stimuli={missing_stimulus_names}"
        )

    response_codes = {code_by_name[name] for name in task_rule.response_event_names}
    stimulus_codes = {code_by_name[name] for name in task_rule.generic_stimulus_event_names}

    valid_prev_codes_by_response_code: dict[int, set[int]] = {}
    for response_name, stimulus_names in task_rule.valid_preceding_stimulus_names_by_response.items():
        if response_name not in code_by_name:
            continue
        valid_prev_codes_by_response_code[code_by_name[response_name]] = {
            code_by_name[name]
            for name in stimulus_names
            if name in code_by_name
        }

    response_epoch_window = get_response_epoch_window(experiment_config)
    response_tmin_tmax_available = response_epoch_window is not None

    details: list[dict[str, Any]] = []
    duplicate_details: list[dict[str, Any]] = []

    # Raw repeated response markers that the current deduplication would remove.
    duplicate_window_samples = int(round((duplicate_window_ms / 1000.0) * sfreq))
    last_kept_sample_by_code: dict[int, int] = {}
    n_duplicate_response_dropped = 0

    for raw_index, event in enumerate(events_raw):
        event_code = int(event[2])
        event_sample = int(event[0])
        last_sample = last_kept_sample_by_code.get(event_code)

        would_drop = False
        if last_sample is not None and event_sample - last_sample <= duplicate_window_samples:
            would_drop = True

        if would_drop:
            if event_code in response_codes:
                n_duplicate_response_dropped += 1
                duplicate_details.append(
                    {
                        "experiment": experiment,
                        "subject_id": subject.subject_id,
                        "group": subject.group,
                        "raw_event_index": int(raw_index),
                        "event_sample": event_sample,
                        "event_time_s": float(event_sample / sfreq),
                        "event_code": event_code,
                        "event_name": safe_event_name(event_code, name_by_event_code),
                        "previous_same_code_kept_sample": int(last_sample),
                        "delta_from_previous_same_code_ms": float(((event_sample - last_sample) / sfreq) * 1000.0),
                        "duplicate_window_ms": duplicate_window_ms,
                    }
                )
            continue

        last_kept_sample_by_code[event_code] = event_sample

    # Post-dedup consecutive response markers and response-context anomalies.
    n_consecutive_response_pairs = 0
    n_response_without_immediate_stimulus = 0
    n_mismatched_response_context = 0
    n_response_epoch_candidates_without_immediate_stimulus = 0
    n_response_epoch_candidates_mismatched_context = 0

    response_mask_post = np.isin(events_post[:, 2], list(response_codes))
    stimulus_mask_post = np.isin(events_post[:, 2], list(stimulus_codes))

    n_response_markers_post = int(np.sum(response_mask_post))
    n_stimulus_markers_post = int(np.sum(stimulus_mask_post))

    for index, event in enumerate(events_post):
        event_code = int(event[2])
        if event_code not in response_codes:
            continue

        previous_event = events_post[index - 1] if index > 0 else None
        next_event = events_post[index + 1] if index + 1 < len(events_post) else None
        previous_code = int(previous_event[2]) if previous_event is not None else None

        would_be_epoch_candidate: bool | None = None
        if response_epoch_window is not None:
            tmin, tmax = response_epoch_window
            would_be_epoch_candidate = event_bounds_ok(
                sample=int(event[0]),
                sfreq=sfreq,
                n_times=n_times,
                tmin=tmin,
                tmax=tmax,
            )

        if next_event is not None and int(next_event[2]) in response_codes:
            n_consecutive_response_pairs += 1
            details.append(
                make_detail_row(
                    experiment=experiment,
                    subject=subject,
                    anomaly_type="consecutive_response_pair",
                    event_index=index,
                    event=event,
                    previous_event=previous_event,
                    next_event=next_event,
                    sfreq=sfreq,
                    name_lookup=name_by_event_code,
                    would_be_epoch_candidate=would_be_epoch_candidate,
                )
            )

        if previous_code not in stimulus_codes:
            n_response_without_immediate_stimulus += 1
            if would_be_epoch_candidate:
                n_response_epoch_candidates_without_immediate_stimulus += 1
            details.append(
                make_detail_row(
                    experiment=experiment,
                    subject=subject,
                    anomaly_type="response_without_immediate_stimulus",
                    event_index=index,
                    event=event,
                    previous_event=previous_event,
                    next_event=next_event,
                    sfreq=sfreq,
                    name_lookup=name_by_event_code,
                    would_be_epoch_candidate=would_be_epoch_candidate,
                )
            )

        valid_prev_codes = valid_prev_codes_by_response_code.get(event_code)
        if valid_prev_codes is not None and previous_code not in valid_prev_codes:
            n_mismatched_response_context += 1
            if would_be_epoch_candidate:
                n_response_epoch_candidates_mismatched_context += 1
            details.append(
                make_detail_row(
                    experiment=experiment,
                    subject=subject,
                    anomaly_type="mismatched_response_context",
                    event_index=index,
                    event=event,
                    previous_event=previous_event,
                    next_event=next_event,
                    sfreq=sfreq,
                    name_lookup=name_by_event_code,
                    would_be_epoch_candidate=would_be_epoch_candidate,
                )
            )

    status_flags: list[str] = []
    if n_duplicate_response_dropped > 0:
        status_flags.append("RAW_DUPLICATE_RESPONSES_DROPPED")
    if n_consecutive_response_pairs > 0:
        status_flags.append("CONSECUTIVE_RESPONSES_POST_DEDUP")
    if n_response_without_immediate_stimulus > 0:
        status_flags.append("RESPONSE_WITHOUT_IMMEDIATE_STIMULUS")
    if n_mismatched_response_context > 0:
        status_flags.append("MISMATCHED_RESPONSE_CONTEXT")
    if not status_flags:
        status_flags.append("OK")

    qc_row = {
        "experiment": experiment,
        "subject_id": subject.subject_id,
        "group": subject.group,
        "n_events_raw_pre_dedup": int(len(events_raw)),
        "n_events_post_dedup": int(len(events_post)),
        "n_events_dropped_by_current_deduplication": int(n_dropped_total),
        "n_stimulus_markers_post_dedup": n_stimulus_markers_post,
        "n_response_markers_post_dedup": n_response_markers_post,
        "n_duplicate_response_markers_dropped_by_current_pipeline": int(n_duplicate_response_dropped),
        "n_consecutive_response_pairs_post_dedup": int(n_consecutive_response_pairs),
        "n_response_without_immediate_stimulus_post_dedup": int(n_response_without_immediate_stimulus),
        "response_locked_candidates_without_immediate_stimulus_pre_rejection": int(n_response_epoch_candidates_without_immediate_stimulus),
        "n_mismatched_response_context_post_dedup": int(n_mismatched_response_context),
        "response_locked_candidates_mismatched_context_pre_rejection": int(n_response_epoch_candidates_mismatched_context),
        "response_epoch_window_available": bool(response_tmin_tmax_available),
        "status": "|".join(status_flags),
    }

    return qc_row, details, duplicate_details


def run() -> None:
    device_config = load_device_config()

    all_qc_rows: list[dict[str, Any]] = []
    all_detail_rows: list[dict[str, Any]] = []
    all_duplicate_rows: list[dict[str, Any]] = []
    all_error_rows: list[dict[str, Any]] = []

    for experiment, task_rule in TASK_RULES.items():
        print(f"\n=== {experiment} ===")
        dataset = get_dataset(experiment)
        dataset.ensure_output_dirs()
        experiment_config = load_experiment_config(experiment)
        subjects = dataset.list_subjects()

        if not subjects:
            print("No EDF files found.")
            continue

        for subject in subjects:
            try:
                qc_row, details, duplicate_details = analyze_subject(
                    subject=subject,
                    task_rule=task_rule,
                    device_config=device_config,
                    experiment_config=experiment_config,
                )
                all_qc_rows.append(qc_row)
                all_detail_rows.extend(details)
                all_duplicate_rows.extend(duplicate_details)
                print(f"{subject.subject_id}: {qc_row['status']}")
            except Exception as exc:  # keep diagnostics robust
                error_row = {
                    "experiment": experiment,
                    "subject_id": subject.subject_id,
                    "group": subject.group,
                    "error": str(exc),
                    "status": "PROCESSING_ERROR",
                }
                all_error_rows.append(error_row)
                all_qc_rows.append(error_row)
                print(f"{subject.subject_id}: PROCESSING_ERROR: {exc}")

    qc_dir = PROJECT_ROOT / "outputs" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    qc_path = qc_dir / "non_tmt_response_marker_sequence_qc.csv"
    details_path = qc_dir / "non_tmt_response_marker_sequence_details.csv"
    duplicates_path = qc_dir / "non_tmt_response_marker_duplicate_details.csv"
    errors_path = qc_dir / "non_tmt_response_marker_sequence_errors.csv"

    qc_df = pd.DataFrame(all_qc_rows)
    details_df = pd.DataFrame(all_detail_rows)
    duplicates_df = pd.DataFrame(all_duplicate_rows)
    errors_df = pd.DataFrame(all_error_rows)

    qc_df.to_csv(qc_path, sep=";", decimal=",", index=False)
    details_df.to_csv(details_path, sep=";", decimal=",", index=False)
    duplicates_df.to_csv(duplicates_path, sep=";", decimal=",", index=False)
    errors_df.to_csv(errors_path, sep=";", decimal=",", index=False)

    print("\nStatus counts:")
    if not qc_df.empty and "status" in qc_df:
        print(qc_df["status"].value_counts(dropna=False).to_string())
    else:
        print("No QC rows.")

    key_columns = [
        "n_duplicate_response_markers_dropped_by_current_pipeline",
        "n_consecutive_response_pairs_post_dedup",
        "n_response_without_immediate_stimulus_post_dedup",
        "response_locked_candidates_without_immediate_stimulus_pre_rejection",
        "n_mismatched_response_context_post_dedup",
        "response_locked_candidates_mismatched_context_pre_rejection",
    ]

    print("\nKey totals by experiment:")
    if not qc_df.empty and "experiment" in qc_df:
        available_key_columns = [c for c in key_columns if c in qc_df.columns]
        numeric_df = qc_df.copy()
        for column in available_key_columns:
            numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce").fillna(0)
        totals = numeric_df.groupby("experiment")[available_key_columns].sum()
        print(totals.to_string())

    print("\nSaved:")
    print(qc_path)
    print(details_path)
    print(duplicates_path)
    print(errors_path)
    print("FINAL RESULT: PASS")


if __name__ == "__main__":
    run()
