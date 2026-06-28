"""
Variable dictionary draft generator for NeuroCogEEG outputs.

This script reads outputs/qc/variable_inventory.csv and creates a first-pass
variable dictionary draft.

It does not run EEG analysis and does not modify result CSV files.
"""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


INVENTORY_PATH = PROJECT_ROOT / "outputs" / "qc" / "variable_inventory.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "qc" / "variable_dictionary_draft.csv"


def infer_unit(column_name, variable_family):
    name = column_name.lower()

    if variable_family == "metadata":
        return "text"

    if variable_family == "processing_error":
        return "text"

    if variable_family == "event_count":
        return "count"

    if column_name == "qc_marker_channel":
        return "text"

    if column_name == "qc_ica_excluded_components":
        return "text_list"

    if name.endswith("_n") or "count" in name or name.endswith("_kept"):
        return "count"

    if name.endswith("_total") or name.endswith("_dropped") or name.endswith("_ignored"):
        return "count"

    if "minimum_required" in name:
        return "count"

    if "minimum_met" in name or name.endswith("_enabled"):
        return "boolean"

    if "percent" in name:
        return "percent"

    if name.endswith("_ms") or "_ms" in name:
        return "ms"

    if name.endswith("_s") or "duration_s" in name:
        return "s"

    if name.endswith("_uv") or "amplitude_uv" in name or name.endswith("_peak_uv"):
        return "microvolt"

    if name.endswith("_uv_s") or "area_uv_s" in name:
        return "microvolt_second"

    if name.endswith("_uv_per_s") or "slope_uv_per_s" in name:
        return "microvolt_per_second"

    if variable_family == "connectivity":
        if name.startswith("z_"):
            return "fisher_z"
        return "connectivity_index"

    if variable_family == "psd":
        if "relative_percent" in name:
            return "percent"
        return "power"

    if name.endswith("_sfreq"):
        return "Hz"

    if name.endswith("_samples"):
        return "samples"

    if name.endswith("_channels"):
        return "count"

    return "unspecified"

def infer_analysis_role(column_name, variable_family):
    name = column_name.lower()

    if variable_family == "metadata":
        return "identifier_or_grouping"

    if variable_family == "processing_error":
        return "processing_qc"

    if variable_family == "event_count":
        return "event_count_qc"

    if variable_family == "behavior_reaction_time":
        return "behavioral_outcome"

    if variable_family == "behavior":
        return "behavioral_outcome"

    if variable_family == "erp":
        return "erp_outcome"

    if variable_family == "cnv":
        return "cnv_outcome"

    if variable_family == "response_locked":
        if "trial_count" in name:
            return "response_locked_qc"
        return "response_locked_outcome"

    if variable_family == "connectivity":
        return "connectivity_outcome"

    if variable_family == "psd":
        return "psd_outcome"

    if variable_family == "quality_control":
        return "quality_control"

    return "needs_review"


def infer_spss_include(column_name, variable_family):
    name = column_name.lower()

    if variable_family in [
        "metadata",
        "behavior_reaction_time",
        "behavior",
        "erp",
        "cnv",
        "response_locked",
        "connectivity",
        "psd",
    ]:
        return "yes"

    if variable_family in ["event_count", "quality_control", "processing_error"]:
        return "qc_only"

    if name.startswith("raw_"):
        return "qc_only"

    return "review"


def infer_missing_allowed(column_name, variable_family):
    name = column_name.lower()

    if variable_family == "metadata":
        return "no"

    if column_name == "qc_ica_excluded_components":
        return "yes_empty_means_no_excluded_components"

    if column_name == "qc_marker_channel":
        return "no"

    if variable_family == "processing_error":
        return "yes"

    if column_name == "ern_amplitude_uv":
        return "yes"

    if column_name == "cnv_amplitude_uv":
        return "yes"

    if variable_family == "connectivity":
        return "yes_if_minimum_trials_fail"

    if variable_family == "response_locked":
        if name == "rp_trial_count":
            return "no"
        return "yes_if_minimum_trials_fail"

    if variable_family in ["erp", "behavior", "behavior_reaction_time", "psd"]:
        return "no_or_review"

    if variable_family in ["event_count", "quality_control"]:
        return "no"

    return "review"

def infer_missing_reason(column_name, variable_family):
    name = column_name.lower()

    if column_name == "qc_ica_excluded_components":
        return "Empty value usually means that no ICA component was excluded."

    if column_name == "ern_amplitude_uv":
        return "May be missing when participant has no usable error-response epochs."

    if column_name == "cnv_amplitude_uv":
        return "May be missing when set-locked minimum trial criterion is not met."

    if variable_family == "connectivity":
        return "May be missing when the epoch set used for connectivity does not meet minimum trial criterion."

    if variable_family == "response_locked" and name != "rp_trial_count":
        return "May be missing when response-locked minimum trial criterion is not met."

    if variable_family == "processing_error":
        return "Empty when processing completed without error."

    return ""


def infer_notes(column_name, variable_family):
    name = column_name.lower()

    if column_name == "subject_id":
        return "Participant identifier derived from EDF filename."

    if column_name == "group":
        return "Between-subject group label."

    if column_name == "experiment":
        return "Experiment/task name."

    if column_name == "tmt_variant":
        return "TMT task variant: tmt1 or tmt2."

    if variable_family == "connectivity":
        return "Computed from configured channel pairs and frequency bands."

    if variable_family == "psd":
        return "Computed from continuous TMT task segment."

    if variable_family == "quality_control":
        return "Use for data-quality filtering and transparent reporting, not as primary outcome."

    if "drop_percent" in name:
        return "Percentage of selected epochs dropped after rejection criteria."

    return ""


def build_dictionary(inventory):
    rows = []

    for _, row in inventory.iterrows():
        column = row["column"]
        variable_family = row["variable_family"]

        rows.append(
            {
                "experiment": row["experiment"],
                "column": column,
                "variable_family": variable_family,
                "unit": infer_unit(column, variable_family),
                "analysis_role": infer_analysis_role(column, variable_family),
                "spss_include": infer_spss_include(column, variable_family),
                "missing_allowed": infer_missing_allowed(column, variable_family),
                "missing_reason": infer_missing_reason(column, variable_family),
                "notes": infer_notes(column, variable_family),
                "missing_count": row["missing_count"],
                "missing_percent": row["missing_percent"],
                "sample_values": row["sample_values"],
            }
        )

    return pd.DataFrame(rows)


def main():
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Variable inventory not found: {INVENTORY_PATH}. "
            "Run pipelines/variable_inventory.py first."
        )

    inventory = pd.read_csv(
        INVENTORY_PATH,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    dictionary = build_dictionary(inventory)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dictionary.to_csv(
        OUTPUT_PATH,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )

    print("Değişken sözlüğü taslağı oluşturuldu.")
    print(f"Dosya: {OUTPUT_PATH}")
    print("")
    print("SPSS dahil etme önerisi:")
    print(dictionary["spss_include"].value_counts(dropna=False))
    print("")
    print("Birim özeti:")
    print(dictionary["unit"].value_counts(dropna=False))
    print("")
    print("Missing allowed özeti:")
    print(dictionary["missing_allowed"].value_counts(dropna=False))


if __name__ == "__main__":
    main()