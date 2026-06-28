"""
Flanker analysis pipeline for NeuroCogEEG.

This pipeline orchestrates the complete Flanker workflow:

- EDF preprocessing
- event extraction
- behavioral analysis
- stimulus-locked ERP analysis
- response-locked ERP/RP analysis
- spectral connectivity analysis
- quality-control summaries
- SPSS-compatible CSV export

The pipeline does not hard-code numeric event values. Event meanings are read
from configs/flanker.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow running this script directly as:
# python pipelines/flanker.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurocogeeg.behavior import (  # noqa: E402
    compute_accuracy_percent,
    compute_transition_latencies_ms,
    count_events_from_config,
    get_event_code,
    summarize_latencies,
)
from neurocogeeg.connectivity import (  # noqa: E402
    compute_connectivity_summary,
    flatten_connectivity_pairs,
    make_pair_key,
)
from neurocogeeg.dataset import SubjectRecord, get_dataset  # noqa: E402
from neurocogeeg.epochs import (  # noqa: E402
    create_epochs,
    extract_events_from_marker_channel,
    filter_event_sequence,
    get_reject_criteria,
)
from neurocogeeg.erp import (  # noqa: E402
    compute_component_from_epochs,
    compute_difference_component_from_epochs,
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


EXPERIMENT = "flanker"


def empty_connectivity_results(
    connectivity_config: dict[str, Any],
) -> dict[str, float]:
    """
    Create NaN-filled connectivity result columns.

    Parameters
    ----------
    connectivity_config:
        Connectivity section from flanker.yaml.

    Returns
    -------
    dict[str, float]
        NaN-filled connectivity result dictionary.
    """
    results: dict[str, float] = {}

    methods = connectivity_config["methods"]
    frequency_bands = connectivity_config["frequency_bands"]
    pairs = flatten_connectivity_pairs(connectivity_config["pairs"])
    use_fisher_z = connectivity_config.get("fisher_z", False)

    for method in methods:
        method_name = str(method).lower()

        for band_name in frequency_bands:
            for pair in pairs:
                pair_key = make_pair_key(
                    pair_group=pair["group"],
                    source=pair["source"],
                    target=pair["target"],
                )

                if use_fisher_z:
                    output_key = f"z_{method_name}_{band_name}_{pair_key}"
                else:
                    output_key = f"{method_name}_{band_name}_{pair_key}"

                results[output_key] = np.nan

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
    Compute Flanker behavioral outcomes.

    Parameters
    ----------
    events:
        MNE events array.

    sfreq:
        Sampling frequency.

    experiment_config:
        Loaded flanker.yaml configuration.

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

    rt_config = behavior_config["reaction_time"]

    rt_start_code = get_event_code(
        events_config=events_config,
        event_name=rt_config["start_event"],
    )
    rt_end_code = get_event_code(
        events_config=events_config,
        event_name=rt_config["end_event"],
    )

    rt_values = compute_transition_latencies_ms(
        events=events,
        start_code=rt_start_code,
        end_code=rt_end_code,
        sfreq=sfreq,
        min_latency_ms=rt_config.get("min_ms"),
        max_latency_ms=rt_config.get("max_ms"),
    )

    rt_summary = summarize_latencies(
        latencies_ms=rt_values,
        prefix="correct_rt",
    )

    accuracy_config = behavior_config["accuracy"]

    correct_code = get_event_code(
        events_config=events_config,
        event_name=accuracy_config["correct_event"],
    )
    total_code = get_event_code(
        events_config=events_config,
        event_name=accuracy_config["total_event"],
    )

    correct_count = int(np.sum(events[:, 2] == correct_code))
    total_count = int(np.sum(events[:, 2] == total_code))

    accuracy = compute_accuracy_percent(
        correct_count=correct_count,
        total_count=total_count,
    )

    return {
        **event_counts,
        **rt_summary,
        "accuracy_percent": accuracy,
    }


def create_stimulus_locked_correct_epochs(
    raw_clean,
    events: np.ndarray,
    experiment_config: dict[str, Any],
):
    """
    Create stimulus-locked epochs for stimuli followed by correct responses.

    Parameters
    ----------
    raw_clean:
        Preprocessed raw object.

    events:
        MNE events array.

    experiment_config:
        Loaded flanker.yaml configuration.

    Returns
    -------
    mne.Epochs
        Stimulus-locked epochs.
    """
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


def create_response_locked_epochs(
    raw_clean,
    events: np.ndarray,
    experiment_config: dict[str, Any],
):
    """
    Create response-locked epochs for correct and error responses.

    Parameters
    ----------
    raw_clean:
        Preprocessed raw object.

    events:
        MNE events array.

    experiment_config:
        Loaded flanker.yaml configuration.

    Returns
    -------
    mne.Epochs
        Response-locked epochs.
    """
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


def compute_stimulus_locked_erp_results(
    stimulus_epochs,
    experiment_config: dict[str, Any],
) -> dict[str, float]:
    """
    Compute N2 and P3 from stimulus-locked epochs.

    Parameters
    ----------
    stimulus_epochs:
        Stimulus-locked epochs.

    experiment_config:
        Loaded flanker.yaml configuration.

    Returns
    -------
    dict[str, float]
        N2 and P3 results.
    """
    components = experiment_config["erp_components"]

    n2_config = components["n2"]
    p3_config = components["p3"]

    n2_results = compute_component_from_epochs(
        epochs=stimulus_epochs,
        condition=n2_config["condition"],
        component_name="n2",
        window_config=n2_config["window"],
        roi_channels=n2_config["roi"],
        method=n2_config["method"],
        polarity=n2_config["polarity"],
    )

    p3_results = compute_component_from_epochs(
        epochs=stimulus_epochs,
        condition=p3_config["condition"],
        component_name="p3",
        window_config=p3_config["window"],
        roi_channels=p3_config["roi"],
        method=p3_config["method"],
        polarity=p3_config["polarity"],
    )

    return {
        **n2_results,
        **p3_results,
    }


def compute_response_locked_results(
    response_epochs,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute ERN, RP and PMP from response-locked epochs.

    Parameters
    ----------
    response_epochs:
        Response-locked epochs.

    experiment_config:
        Loaded flanker.yaml configuration.

    Returns
    -------
    dict[str, Any]
        ERN, RP and PMP results.
    """
    ern_config = experiment_config["erp_components"]["ern"]

    ern_results = compute_difference_component_from_epochs(
        epochs=response_epochs,
        condition_a=ern_config["condition_a"],
        condition_b=ern_config["condition_b"],
        component_name="ern",
        window_config=ern_config["window"],
        roi_channels=ern_config["roi"],
        method=ern_config["method"],
        polarity=ern_config["polarity"],
        weight_a=ern_config["weights"]["condition_a"],
        weight_b=ern_config["weights"]["condition_b"],
    )

    rp_config = experiment_config["response_locked"]["rp"]
    pmp_config = experiment_config["response_locked"]["pmp"]

    rp_results = compute_response_locked_metrics_from_epochs(
        epochs=response_epochs,
        roi_channels=rp_config["roi"],
        rp_window_config=rp_config["window"],
        pmp_window_config=pmp_config["window"],
        condition=rp_config["condition"],
    )

    return {
        **ern_results,
        **rp_results,
    }


def compute_connectivity_results(
    stimulus_epochs,
    experiment_config: dict[str, Any],
) -> dict[str, float]:
    """
    Compute Flanker connectivity results.

    Parameters
    ----------
    stimulus_epochs:
        Stimulus-locked epochs.

    experiment_config:
        Loaded flanker.yaml configuration.

    Returns
    -------
    dict[str, float]
        Connectivity result dictionary.
    """
    connectivity_config = experiment_config["connectivity"]
    time_window = connectivity_config["time_window"]

    return compute_connectivity_summary(
        epochs=stimulus_epochs,
        connectivity_config=connectivity_config,
        tmin=time_window["start"],
        tmax=time_window["end"],
        mode=connectivity_config.get("mode", "multitaper"),
        faverage=connectivity_config.get("faverage", True),
        fisher_z=connectivity_config.get("fisher_z", False),
    )


def process_subject(
    subject: SubjectRecord,
    device_config: dict[str, Any],
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Process one participant.

    Parameters
    ----------
    subject:
        Subject record.

    device_config:
        Loaded Emotiv device configuration.

    experiment_config:
        Loaded Flanker experiment configuration.

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

        qc_stimulus_epochs = summarize_epochs(
            stimulus_epochs,
            prefix="stimulus_locked_epochs",
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

        stimulus_minimum = summarize_minimum_trial_requirement(
            observed_count=len(stimulus_epochs),
            minimum_count=minimum_trials["stimulus_locked"],
            label="stimulus_locked_trials",
        )

        response_minimum = summarize_minimum_trial_requirement(
            observed_count=len(response_epochs),
            minimum_count=minimum_trials["response_locked"],
            label="response_locked_trials",
        )

        erp_results: dict[str, Any]

        if len(stimulus_epochs) >= minimum_trials["stimulus_locked"]:
            erp_results = compute_stimulus_locked_erp_results(
                stimulus_epochs=stimulus_epochs,
                experiment_config=experiment_config,
            )
        else:
            erp_results = {
                "n2_amplitude_uv": np.nan,
                "p3_amplitude_uv": np.nan,
            }

        response_results = compute_response_locked_results(
            response_epochs=response_epochs,
            experiment_config=experiment_config,
        )

        if len(stimulus_epochs) >= minimum_trials["connectivity"]:
            connectivity_results = compute_connectivity_results(
                stimulus_epochs=stimulus_epochs,
                experiment_config=experiment_config,
            )
        else:
            connectivity_results = empty_connectivity_results(
                experiment_config["connectivity"]
            )

        qc_results = merge_qc_summaries(
            qc_event_results,
            qc_preprocessing_results,
            qc_raw_results,
            qc_stimulus_epochs,
            qc_response_epochs,
            qc_response_conditions,
            stimulus_minimum,
            response_minimum,
        )

        result_row = merge_result_dicts(
            behavioral_results,
            erp_results,
            response_results,
            connectivity_results,
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
    Run the Flanker pipeline for all available participants.

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
        output_path=dataset.reports_dir / "used_flanker_config.yaml",
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

    output_path = dataset.csv_dir / "flanker_results.csv"

    save_spss_csv(
        rows=rows,
        output_path=output_path,
    )

    print(f"Sonuç dosyası kaydedildi: {output_path}")

    return output_path


if __name__ == "__main__":
    run_pipeline()