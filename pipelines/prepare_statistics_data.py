"""
Statistics preparation pipeline for NeuroCogEEG.

This script prepares analysis-ready CSV files for later inferential statistics.

It does not run statistical tests.

Main tasks:
- Read validated SPSS main CSV files.
- Add stable participant_id fields.
- Validate TMT tmt1/tmt2 pairing.
- Add analysis-specific inclusion flags.
- Write statistics-ready CSV files.
- Write validation reports.

Outputs:
- outputs/statistics/flanker_statistics_ready.csv
- outputs/statistics/gonogo_statistics_ready.csv
- outputs/statistics/readysetgo_statistics_ready.csv
- outputs/statistics/tmt_statistics_ready.csv
- outputs/qc/statistics_preparation_report.csv
- outputs/qc/statistics_preparation_summary.csv
"""

import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SPSS_MAIN_FILES = {
    "flanker": PROJECT_ROOT / "outputs" / "flanker" / "csv" / "flanker_spss_main.csv",
    "gonogo": PROJECT_ROOT / "outputs" / "gonogo" / "csv" / "gonogo_spss_main.csv",
    "readysetgo": PROJECT_ROOT / "outputs" / "readysetgo" / "csv" / "readysetgo_spss_main.csv",
    "tmt": PROJECT_ROOT / "outputs" / "tmt" / "csv" / "tmt_spss_main.csv",
}


STATISTICS_OUTPUT_FILES = {
    "flanker": PROJECT_ROOT / "outputs" / "statistics" / "flanker_statistics_ready.csv",
    "gonogo": PROJECT_ROOT / "outputs" / "statistics" / "gonogo_statistics_ready.csv",
    "readysetgo": PROJECT_ROOT / "outputs" / "statistics" / "readysetgo_statistics_ready.csv",
    "tmt": PROJECT_ROOT / "outputs" / "statistics" / "tmt_statistics_ready.csv",
}


VALID_GROUPS = {
    "control",
    "experimental",
}

VALID_TMT_VARIANTS = {
    "tmt1",
    "tmt2",
}


def read_csv(path):
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )


def write_csv(dataframe, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )


def as_binary_flag(series):
    text = series.astype(str).str.strip().str.lower()

    mapped = text.map(
        {
            "1": 1,
            "1.0": 1,
            "true": 1,
            "yes": 1,
            "pass": 1,
            "0": 0,
            "0.0": 0,
            "false": 0,
            "no": 0,
            "fail": 0,
        }
    )

    return mapped.astype("Int64")


def numeric_not_missing_flag(dataframe, column):
    if column not in dataframe.columns:
        return pd.Series([0] * len(dataframe), index=dataframe.index).astype("Int64")

    values = pd.to_numeric(dataframe[column], errors="coerce")

    return values.notna().astype("Int64")


def minimum_flag(dataframe, column):
    if column not in dataframe.columns:
        return pd.Series([0] * len(dataframe), index=dataframe.index).astype("Int64")

    return as_binary_flag(dataframe[column])


def add_participant_id_after_subject_id(dataframe, participant_ids):
    output = dataframe.copy()

    if "participant_id" in output.columns:
        output = output.drop(columns=["participant_id"])

    subject_index = list(output.columns).index("subject_id")

    output.insert(
        subject_index + 1,
        "participant_id",
        participant_ids,
    )

    return output


def derive_tmt_participant_id(subject_id):
    """
    Derive stable TMT participant_id from variant-specific subject_id.

    Rule:
    - xxxxxizol1 -> xxxxxizol
    - xxxxxizol2 -> xxxxxizol

    The final 1/2 is removed only when the subject_id ends with izol1 or izol2.
    """
    text = str(subject_id).strip()

    if re.search(r"izol[12]$", text, flags=re.IGNORECASE):
        return text[:-1]

    return text


def validate_basic_metadata(experiment, dataframe):
    rows = []

    required_columns = ["subject_id", "group", "experiment"]

    for column in required_columns:
        if column not in dataframe.columns:
            rows.append(
                {
                    "experiment": experiment,
                    "severity": "ERROR",
                    "issue": "required_metadata_column_missing",
                    "subject_id": "",
                    "participant_id": "",
                    "details": column,
                }
            )

    if "group" in dataframe.columns:
        observed_groups = set(dataframe["group"].dropna().astype(str).unique())
        invalid_groups = sorted(observed_groups - VALID_GROUPS)

        for group in invalid_groups:
            rows.append(
                {
                    "experiment": experiment,
                    "severity": "ERROR",
                    "issue": "invalid_group_value",
                    "subject_id": "",
                    "participant_id": "",
                    "details": group,
                }
            )

    if "subject_id" in dataframe.columns:
        duplicated = dataframe[
            dataframe["subject_id"].duplicated(keep=False)
        ]

        if experiment != "tmt" and not duplicated.empty:
            for subject_id in duplicated["subject_id"].astype(str).unique():
                rows.append(
                    {
                        "experiment": experiment,
                        "severity": "ERROR",
                        "issue": "duplicate_subject_id",
                        "subject_id": subject_id,
                        "participant_id": subject_id,
                        "details": "",
                    }
                )

    return rows


def validate_tmt_participant_ids(dataframe):
    rows = []

    if "tmt_variant" not in dataframe.columns:
        rows.append(
            {
                "experiment": "tmt",
                "severity": "ERROR",
                "issue": "tmt_variant_column_missing",
                "subject_id": "",
                "participant_id": "",
                "details": "",
            }
        )
        return rows

    observed_variants = set(dataframe["tmt_variant"].dropna().astype(str).unique())
    invalid_variants = sorted(observed_variants - VALID_TMT_VARIANTS)

    for variant in invalid_variants:
        rows.append(
            {
                "experiment": "tmt",
                "severity": "ERROR",
                "issue": "invalid_tmt_variant_value",
                "subject_id": "",
                "participant_id": "",
                "details": variant,
            }
        )

    for _, row in dataframe.iterrows():
        subject_id = str(row["subject_id"])
        variant = str(row["tmt_variant"])
        participant_id = str(row["participant_id"])

        if not re.search(r"izol[12]$", subject_id, flags=re.IGNORECASE):
            rows.append(
                {
                    "experiment": "tmt",
                    "severity": "ERROR",
                    "issue": "subject_id_does_not_match_tmt_suffix_rule",
                    "subject_id": subject_id,
                    "participant_id": participant_id,
                    "details": "Expected subject_id to end with izol1 or izol2.",
                }
            )

        if variant == "tmt1" and not re.search(r"izol1$", subject_id, flags=re.IGNORECASE):
            rows.append(
                {
                    "experiment": "tmt",
                    "severity": "ERROR",
                    "issue": "tmt1_subject_id_suffix_mismatch",
                    "subject_id": subject_id,
                    "participant_id": participant_id,
                    "details": "Expected tmt1 subject_id to end with izol1.",
                }
            )

        if variant == "tmt2" and not re.search(r"izol2$", subject_id, flags=re.IGNORECASE):
            rows.append(
                {
                    "experiment": "tmt",
                    "severity": "ERROR",
                    "issue": "tmt2_subject_id_suffix_mismatch",
                    "subject_id": subject_id,
                    "participant_id": participant_id,
                    "details": "Expected tmt2 subject_id to end with izol2.",
                }
            )

    duplicate_rows = dataframe[
        dataframe.duplicated(
            subset=["participant_id", "tmt_variant"],
            keep=False,
        )
    ]

    for _, row in duplicate_rows.iterrows():
        rows.append(
            {
                "experiment": "tmt",
                "severity": "ERROR",
                "issue": "duplicate_participant_variant_row",
                "subject_id": row["subject_id"],
                "participant_id": row["participant_id"],
                "details": str(row["tmt_variant"]),
            }
        )

    group_counts = dataframe.groupby("participant_id")["group"].nunique()

    for participant_id, group_count in group_counts.items():
        if int(group_count) > 1:
            rows.append(
                {
                    "experiment": "tmt",
                    "severity": "ERROR",
                    "issue": "participant_id_has_multiple_groups",
                    "subject_id": "",
                    "participant_id": participant_id,
                    "details": f"group_count={group_count}",
                }
            )

    variant_sets = dataframe.groupby("participant_id")["tmt_variant"].apply(
        lambda values: set(values.astype(str))
    )

    for participant_id, variants in variant_sets.items():
        if variants != VALID_TMT_VARIANTS:
            rows.append(
                {
                    "experiment": "tmt",
                    "severity": "ERROR",
                    "issue": "participant_missing_tmt_pair",
                    "subject_id": "",
                    "participant_id": participant_id,
                    "details": f"observed_variants={sorted(variants)}",
                }
            )

    return rows


def add_common_inclusion_flags(experiment, dataframe):
    output = dataframe.copy()

    output["include_behavior"] = 1

    if "stimulus_locked_trials_minimum_met" in output.columns:
        output["include_stimulus_locked"] = minimum_flag(
            output,
            "stimulus_locked_trials_minimum_met",
        )

    if "set_locked_trials_minimum_met" in output.columns:
        output["include_set_locked"] = minimum_flag(
            output,
            "set_locked_trials_minimum_met",
        )

    if "response_locked_trials_minimum_met" in output.columns:
        output["include_response_locked"] = minimum_flag(
            output,
            "response_locked_trials_minimum_met",
        )

    return output


def prepare_flanker(dataframe):
    output = add_participant_id_after_subject_id(
        dataframe=dataframe,
        participant_ids=dataframe["subject_id"].astype(str),
    )

    output = add_common_inclusion_flags(
        experiment="flanker",
        dataframe=output,
    )

    output["include_n2_p3"] = output["include_stimulus_locked"]
    output["include_connectivity"] = output["include_stimulus_locked"]

    output["include_ern"] = (
        (output["include_response_locked"] == 1)
        & (numeric_not_missing_flag(output, "ern_amplitude_uv") == 1)
    ).astype("Int64")

    output["include_rp_pmp"] = output["include_response_locked"]

    return output


def prepare_gonogo(dataframe):
    output = add_participant_id_after_subject_id(
        dataframe=dataframe,
        participant_ids=dataframe["subject_id"].astype(str),
    )

    output = add_common_inclusion_flags(
        experiment="gonogo",
        dataframe=output,
    )

    output["include_n2_p3"] = output["include_stimulus_locked"]
    output["include_connectivity"] = output["include_stimulus_locked"]
    output["include_rp_pmp"] = output["include_response_locked"]

    return output


def prepare_readysetgo(dataframe):
    output = add_participant_id_after_subject_id(
        dataframe=dataframe,
        participant_ids=dataframe["subject_id"].astype(str),
    )

    output = add_common_inclusion_flags(
        experiment="readysetgo",
        dataframe=output,
    )

    output["include_cnv"] = (
        (output["include_set_locked"] == 1)
        & (numeric_not_missing_flag(output, "cnv_amplitude_uv") == 1)
    ).astype("Int64")

    output["include_set_locked_connectivity"] = output["include_set_locked"]
    output["include_rp_pmp"] = output["include_response_locked"]

    return output


def prepare_tmt(dataframe):
    participant_ids = dataframe["subject_id"].apply(derive_tmt_participant_id)

    output = add_participant_id_after_subject_id(
        dataframe=dataframe,
        participant_ids=participant_ids,
    )

    output = add_common_inclusion_flags(
        experiment="tmt",
        dataframe=output,
    )

    output["include_psd"] = 1
    output["include_rp_pmp"] = output["include_response_locked"]

    pair_counts = output.groupby("participant_id")["tmt_variant"].transform(
        lambda values: len(set(values.astype(str)))
    )

    output["include_tmt_complete_pair"] = (pair_counts == 2).astype("Int64")

    return output


def prepare_one_experiment(experiment):
    path = SPSS_MAIN_FILES[experiment]

    if not path.exists():
        return None, [
            {
                "experiment": experiment,
                "severity": "ERROR",
                "issue": "spss_main_file_missing",
                "subject_id": "",
                "participant_id": "",
                "details": str(path),
            }
        ]

    dataframe = read_csv(path)

    report_rows = validate_basic_metadata(
        experiment=experiment,
        dataframe=dataframe,
    )

    if experiment == "flanker":
        prepared = prepare_flanker(dataframe)
    elif experiment == "gonogo":
        prepared = prepare_gonogo(dataframe)
    elif experiment == "readysetgo":
        prepared = prepare_readysetgo(dataframe)
    elif experiment == "tmt":
        prepared = prepare_tmt(dataframe)
        report_rows.extend(validate_tmt_participant_ids(prepared))
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    return prepared, report_rows


def summarize_prepared_data(experiment, dataframe):
    rows = []

    include_columns = [
        column for column in dataframe.columns
        if column.startswith("include_")
    ]

    for column in include_columns:
        values = as_binary_flag(dataframe[column])

        rows.append(
            {
                "experiment": experiment,
                "summary_type": "include_flag",
                "variable": column,
                "total_rows": int(len(dataframe)),
                "included_count": int((values == 1).sum()),
                "excluded_count": int((values == 0).sum()),
                "missing_count": int(values.isna().sum()),
            }
        )

    if experiment == "tmt":
        pair_summary = (
            dataframe.groupby("participant_id")["tmt_variant"]
            .nunique()
            .value_counts()
            .sort_index()
        )

        for variant_count, participant_count in pair_summary.items():
            rows.append(
                {
                    "experiment": "tmt",
                    "summary_type": "participant_pairing",
                    "variable": f"variant_count_{variant_count}",
                    "total_rows": int(len(dataframe)),
                    "included_count": int(participant_count),
                    "excluded_count": "",
                    "missing_count": "",
                }
            )

    return rows


def main():
    all_report_rows = []
    all_summary_rows = []

    for experiment in SPSS_MAIN_FILES:
        prepared, report_rows = prepare_one_experiment(experiment)

        all_report_rows.extend(report_rows)

        if prepared is None:
            continue

        output_path = STATISTICS_OUTPUT_FILES[experiment]
        write_csv(prepared, output_path)

        all_summary_rows.extend(
            summarize_prepared_data(
                experiment=experiment,
                dataframe=prepared,
            )
        )

        print(
            f"{experiment}: statistics-ready dosya yazıldı "
            f"({len(prepared)} satır, {len(prepared.columns)} sütun): "
            f"{output_path}"
        )

    report = pd.DataFrame(all_report_rows)

    if report.empty:
        report = pd.DataFrame(
            columns=[
                "experiment",
                "severity",
                "issue",
                "subject_id",
                "participant_id",
                "details",
            ]
        )

    summary = pd.DataFrame(all_summary_rows)

    report_path = PROJECT_ROOT / "outputs" / "qc" / "statistics_preparation_report.csv"
    summary_path = PROJECT_ROOT / "outputs" / "qc" / "statistics_preparation_summary.csv"

    write_csv(report, report_path)
    write_csv(summary, summary_path)

    errors = report[report["severity"] == "ERROR"]

    print("")
    print("Statistics preparation summary:")
    print(summary.to_string(index=False))
    print("")
    print(f"Report file: {report_path}")
    print(f"Summary file: {summary_path}")
    print("")
    print(f"ERROR sayısı: {len(errors)}")

    if not errors.empty:
        print("")
        print(errors.to_string(index=False))
        raise SystemExit(1)

    print("")
    print("Sonuç: PASS")


if __name__ == "__main__":
    main()