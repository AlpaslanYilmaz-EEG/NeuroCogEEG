"""
Validation script for NeuroCogEEG SPSS main files.

This script checks whether reduced SPSS-ready main analysis CSV files are
structurally safe and consistent with the variable dictionary.

It does not modify analysis result files.
"""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DICTIONARY_PATH = PROJECT_ROOT / "outputs" / "qc" / "variable_dictionary_draft.csv"

RESULT_FILES = {
    "flanker": PROJECT_ROOT / "outputs" / "flanker" / "csv" / "flanker_results.csv",
    "gonogo": PROJECT_ROOT / "outputs" / "gonogo" / "csv" / "gonogo_results.csv",
    "readysetgo": PROJECT_ROOT / "outputs" / "readysetgo" / "csv" / "readysetgo_results.csv",
    "tmt": PROJECT_ROOT / "outputs" / "tmt" / "csv" / "tmt_results.csv",
}

SPSS_MAIN_FILES = {
    "flanker": PROJECT_ROOT / "outputs" / "flanker" / "csv" / "flanker_spss_main.csv",
    "gonogo": PROJECT_ROOT / "outputs" / "gonogo" / "csv" / "gonogo_spss_main.csv",
    "readysetgo": PROJECT_ROOT / "outputs" / "readysetgo" / "csv" / "readysetgo_spss_main.csv",
    "tmt": PROJECT_ROOT / "outputs" / "tmt" / "csv" / "tmt_spss_main.csv",
}


CRITICAL_COLUMNS = {
    "flanker": [
        "subject_id",
        "group",
        "experiment",
        "correct_rt_mean_ms",
        "accuracy_percent",
        "n2_amplitude_uv",
        "p3_amplitude_uv",
        "ern_amplitude_uv",
        "rp_mean_uv",
        "pmp_peak_uv",
        "stimulus_locked_trials_minimum_met",
        "response_locked_trials_minimum_met",
    ],
    "gonogo": [
        "subject_id",
        "group",
        "experiment",
        "correct_rt_mean_ms",
        "accuracy_percent",
        "n2_amplitude_uv",
        "p3_amplitude_uv",
        "rp_mean_uv",
        "pmp_peak_uv",
        "stimulus_locked_trials_minimum_met",
        "response_locked_trials_minimum_met",
    ],
    "readysetgo": [
        "subject_id",
        "group",
        "experiment",
        "response_time_mean_ms",
        "accuracy_percent",
        "cnv_amplitude_uv",
        "rp_mean_uv",
        "pmp_peak_uv",
        "set_locked_trials_minimum_met",
        "response_locked_trials_minimum_met",
    ],
    "tmt": [
        "subject_id",
        "group",
        "experiment",
        "tmt_variant",
        "task_duration_s",
        "correct_click_count",
        "error_click_count",
        "error_percent",
        "frontal_theta_relative_percent",
        "parietal_occipital_theta_relative_percent",
        "rp_mean_uv",
        "pmp_peak_uv",
        "response_locked_trials_minimum_met",
    ],
}


SUSPICIOUS_EVENT_COLUMNS = {
    "stimulus",
    "correct_response",
    "error_response",
    "missed_response",
    "ready",
    "set",
    "go_response",
    "task_start",
    "correct_click",
    "error_click",
    "task_end",
}


def read_csv(path):
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )


def find_suspicious_columns(columns):
    suspicious = []

    for column in columns:
        if column.startswith("raw_"):
            suspicious.append(column)
        elif column.startswith("qc_"):
            suspicious.append(column)
        elif column.startswith("event_count_"):
            suspicious.append(column)
        elif column in SUSPICIOUS_EVENT_COLUMNS:
            suspicious.append(column)

    return suspicious


def validate_minimum_flag_columns(dataframe):
    issues = []

    flag_columns = [
        column
        for column in dataframe.columns
        if column.endswith("_minimum_met")
    ]

    for column in flag_columns:
        values = dataframe[column].dropna().unique()
        invalid_values = [
            value
            for value in values
            if value not in [0, 1, 0.0, 1.0]
        ]

        if invalid_values:
            issues.append(
                {
                    "column": column,
                    "issue": "minimum_flag_not_numeric_0_1",
                    "details": str(invalid_values),
                }
            )

    return issues


def get_minimum_flag_for_measure(experiment, column):
    """Return the QC minimum-trial flag that can explain a missing measure."""
    response_locked_tokens = (
        "ern_",
        "rp_",
        "pmp_",
        "left_frontocentral_rp_",
        "left_frontocentral_pmp_",
    )

    if column.startswith(response_locked_tokens):
        return "response_locked_trials_minimum_met"

    if experiment in {"flanker", "gonogo"} and column.startswith(("n2_", "p3_")):
        return "stimulus_locked_trials_minimum_met"

    if experiment == "readysetgo" and (
        column.startswith("cnv_")
        or column.startswith("left_frontocentral_cnv_")
    ):
        return "set_locked_trials_minimum_met"

    return None


def validate_no_missing_required_columns(experiment, dataframe, dictionary):
    """Validate missing values while allowing QC-explained analysis missingness.

    A measure marked ``missing_allowed=no`` in the variable dictionary remains
    required. However, an analysis-derived value may legitimately be missing
    when its corresponding minimum-trial QC flag is 0. Such rows are reported
    as WARNING rather than ERROR. Missing values not explained by the QC flag
    remain ERRORs.
    """
    issues = []

    experiment_dictionary = dictionary[
        dictionary["experiment"] == experiment
    ].copy()

    dictionary_by_column = {
        row["column"]: row
        for _, row in experiment_dictionary.iterrows()
    }

    for column in dataframe.columns:
        row = dictionary_by_column.get(column)

        if row is None:
            continue

        missing_allowed = str(row["missing_allowed"]).strip().lower()
        missing_mask = dataframe[column].isna()
        missing_count = int(missing_mask.sum())

        if missing_allowed != "no" or missing_count == 0:
            continue

        minimum_flag = get_minimum_flag_for_measure(
            experiment=experiment,
            column=column,
        )

        if minimum_flag is None or minimum_flag not in dataframe.columns:
            issues.append(
                {
                    "severity": "ERROR",
                    "column": column,
                    "issue": "unexpected_missing_values",
                    "details": f"missing_count={missing_count}",
                }
            )
            continue

        flag_values = pd.to_numeric(
            dataframe[minimum_flag],
            errors="coerce",
        )
        expected_mask = missing_mask & flag_values.eq(0)
        unexpected_mask = missing_mask & ~flag_values.eq(0)

        expected_count = int(expected_mask.sum())
        unexpected_count = int(unexpected_mask.sum())

        if expected_count > 0:
            subject_details = ""
            if "subject_id" in dataframe.columns:
                subjects = dataframe.loc[
                    expected_mask,
                    "subject_id",
                ].astype(str).tolist()
                subject_details = f", subjects={subjects}"

            issues.append(
                {
                    "severity": "WARNING",
                    "column": column,
                    "issue": "expected_missing_due_to_minimum_trials",
                    "details": (
                        f"missing_count={expected_count}, "
                        f"minimum_flag={minimum_flag}{subject_details}"
                    ),
                }
            )

        if unexpected_count > 0:
            subject_details = ""
            if "subject_id" in dataframe.columns:
                subjects = dataframe.loc[
                    unexpected_mask,
                    "subject_id",
                ].astype(str).tolist()
                subject_details = f", subjects={subjects}"

            issues.append(
                {
                    "severity": "ERROR",
                    "column": column,
                    "issue": "unexpected_missing_values",
                    "details": (
                        f"missing_count={unexpected_count}, "
                        f"minimum_flag={minimum_flag}{subject_details}"
                    ),
                }
            )

    return issues


def validate_critical_columns(experiment, dataframe):
    issues = []

    expected = CRITICAL_COLUMNS[experiment]

    for column in expected:
        if column not in dataframe.columns:
            issues.append(
                {
                    "column": column,
                    "issue": "critical_column_missing",
                    "details": "",
                }
            )

    return issues


def validate_tmt_variant(experiment, dataframe):
    issues = []

    if experiment != "tmt":
        if "tmt_variant" in dataframe.columns:
            issues.append(
                {
                    "column": "tmt_variant",
                    "issue": "tmt_variant_present_in_non_tmt_file",
                    "details": "",
                }
            )
        return issues

    if "tmt_variant" not in dataframe.columns:
        issues.append(
            {
                "column": "tmt_variant",
                "issue": "tmt_variant_missing",
                "details": "",
            }
        )
        return issues

    variants = set(dataframe["tmt_variant"].dropna().astype(str).unique())

    if variants != {"tmt1", "tmt2"}:
        issues.append(
            {
                "column": "tmt_variant",
                "issue": "unexpected_tmt_variant_values",
                "details": str(sorted(variants)),
            }
        )

    return issues


def validate_one_experiment(experiment, dictionary):
    rows = []

    result_path = RESULT_FILES[experiment]
    spss_path = SPSS_MAIN_FILES[experiment]

    if not result_path.exists():
        rows.append(
            {
                "experiment": experiment,
                "severity": "ERROR",
                "column": "",
                "issue": "result_file_missing",
                "details": str(result_path),
            }
        )
        return rows

    if not spss_path.exists():
        rows.append(
            {
                "experiment": experiment,
                "severity": "ERROR",
                "column": "",
                "issue": "spss_main_file_missing",
                "details": str(spss_path),
            }
        )
        return rows

    result_df = read_csv(result_path)
    spss_df = read_csv(spss_path)

    if len(result_df) != len(spss_df):
        rows.append(
            {
                "experiment": experiment,
                "severity": "ERROR",
                "column": "",
                "issue": "row_count_mismatch",
                "details": f"result={len(result_df)}, spss={len(spss_df)}",
            }
        )

    suspicious_columns = find_suspicious_columns(spss_df.columns)

    for column in suspicious_columns:
        rows.append(
            {
                "experiment": experiment,
                "severity": "ERROR",
                "column": column,
                "issue": "suspicious_technical_column_in_spss_main",
                "details": "",
            }
        )

    for issue in validate_minimum_flag_columns(spss_df):
        rows.append(
            {
                "experiment": experiment,
                "severity": "ERROR",
                **issue,
            }
        )

    for issue in validate_no_missing_required_columns(
        experiment=experiment,
        dataframe=spss_df,
        dictionary=dictionary,
    ):
        rows.append(
            {
                "experiment": experiment,
                **issue,
            }
        )

    for issue in validate_critical_columns(
        experiment=experiment,
        dataframe=spss_df,
    ):
        rows.append(
            {
                "experiment": experiment,
                "severity": "ERROR",
                **issue,
            }
        )

    for issue in validate_tmt_variant(
        experiment=experiment,
        dataframe=spss_df,
    ):
        rows.append(
            {
                "experiment": experiment,
                "severity": "ERROR",
                **issue,
            }
        )

    rows.append(
        {
            "experiment": experiment,
            "severity": "INFO",
            "column": "",
            "issue": "file_summary",
            "details": (
                f"rows={len(spss_df)}, "
                f"columns={len(spss_df.columns)}"
            ),
        }
    )

    return rows


def main():
    if not DICTIONARY_PATH.exists():
        raise FileNotFoundError(
            f"Variable dictionary not found: {DICTIONARY_PATH}. "
            "Run pipelines/variable_dictionary_draft.py first."
        )

    dictionary = read_csv(DICTIONARY_PATH)

    validation_rows = []

    for experiment in SPSS_MAIN_FILES:
        validation_rows.extend(
            validate_one_experiment(
                experiment=experiment,
                dictionary=dictionary,
            )
        )

    validation = pd.DataFrame(validation_rows)

    output_dir = PROJECT_ROOT / "outputs" / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "spss_main_validation_report.csv"

    validation.to_csv(
        output_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )

    errors = validation[validation["severity"] == "ERROR"]
    warnings = validation[validation["severity"] == "WARNING"]
    info = validation[validation["severity"] == "INFO"]

    print("SPSS ana dosya doğrulama raporu")
    print("")
    print("Dosya özetleri:")
    print(info[["experiment", "details"]].to_string(index=False))
    print("")
    print(f"WARNING sayısı: {len(warnings)}")

    if not warnings.empty:
        print("")
        print("QC ile açıklanan beklenen eksiklikler:")
        print(
            warnings[
                ["experiment", "column", "issue", "details"]
            ].to_string(index=False)
        )

    print("")
    print(f"ERROR sayısı: {len(errors)}")

    if not errors.empty:
        print("")
        print(errors[["experiment", "column", "issue", "details"]].to_string(index=False))
        raise SystemExit(1)

    print("")
    print("Sonuç: PASS")
    print(f"Rapor dosyası: {output_path}")


if __name__ == "__main__":
    main()