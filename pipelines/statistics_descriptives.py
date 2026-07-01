"""
Descriptive statistics pipeline for NeuroCogEEG.

This script creates descriptive statistics tables from statistics-ready CSV files.

It does not run inferential statistical tests.

Outputs:
- outputs/statistics/descriptives/all_descriptives.csv
- outputs/statistics/descriptives/<experiment>_descriptives.csv
- outputs/qc/statistics_descriptives_report.csv
- outputs/qc/statistics_descriptives_summary.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STATISTICS_READY_FILES = {
    "flanker": PROJECT_ROOT / "outputs" / "statistics" / "flanker_statistics_ready.csv",
    "gonogo": PROJECT_ROOT / "outputs" / "statistics" / "gonogo_statistics_ready.csv",
    "readysetgo": PROJECT_ROOT / "outputs" / "statistics" / "readysetgo_statistics_ready.csv",
    "tmt": PROJECT_ROOT / "outputs" / "statistics" / "tmt_statistics_ready.csv",
}


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "statistics" / "descriptives"
QC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qc"


CONNECTIVITY_PREFIXES = (
    "coh_",
    "wpli_",
    "ciplv_",
    "z_coh_",
    "z_wpli_",
    "z_ciplv_",
)


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


def existing_columns(dataframe, columns):
    return [column for column in columns if column in dataframe.columns]


def missing_columns(dataframe, columns):
    return [column for column in columns if column not in dataframe.columns]


def connectivity_columns(dataframe):
    return [
        column
        for column in dataframe.columns
        if column.startswith(CONNECTIVITY_PREFIXES)
    ]


def tmt_psd_columns(dataframe):
    return [
        column
        for column in dataframe.columns
        if "relative_percent" in column
    ]


def get_experiment_plan(experiment, dataframe):
    if experiment == "flanker":
        return [
            {
                "analysis_family": "behavior",
                "grouping_columns": ["group"],
                "include_flags": ["include_behavior"],
                "variables": [
                    "correct_rt_mean_ms",
                    "correct_rt_median_ms",
                    "correct_rt_sd_ms",
                    "accuracy_percent",
                ],
            },
            {
                "analysis_family": "erp_n2_p3",
                "grouping_columns": ["group"],
                "include_flags": ["include_n2_p3"],
                "variables": [
                    "n2_amplitude_uv",
                    "p3_amplitude_uv",
                ],
            },
            {
                "analysis_family": "erp_ern",
                "grouping_columns": ["group"],
                "include_flags": ["include_ern"],
                "variables": [
                    "ern_amplitude_uv",
                ],
            },
            {
                "analysis_family": "response_locked",
                "grouping_columns": ["group"],
                "include_flags": ["include_rp_pmp"],
                "variables": [
                    "rp_mean_uv",
                    "rp_peak_uv",
                    "rp_area_uv_s",
                    "rp_slope_uv_per_s",
                    "pmp_mean_uv",
                    "pmp_peak_uv",
                    "pmp_area_uv_s",
                ],
            },
            {
                "analysis_family": "connectivity",
                "grouping_columns": ["group"],
                "include_flags": ["include_connectivity"],
                "variables": connectivity_columns(dataframe),
            },
        ]

    if experiment == "gonogo":
        return [
            {
                "analysis_family": "behavior",
                "grouping_columns": ["group"],
                "include_flags": ["include_behavior"],
                "variables": [
                    "correct_rt_mean_ms",
                    "correct_rt_median_ms",
                    "correct_rt_sd_ms",
                    "accuracy_percent",
                ],
            },
            {
                "analysis_family": "erp_n2_p3",
                "grouping_columns": ["group"],
                "include_flags": ["include_n2_p3"],
                "variables": [
                    "n2_amplitude_uv",
                    "p3_amplitude_uv",
                ],
            },
            {
                "analysis_family": "response_locked",
                "grouping_columns": ["group"],
                "include_flags": ["include_rp_pmp"],
                "variables": [
                    "rp_mean_uv",
                    "rp_peak_uv",
                    "rp_area_uv_s",
                    "rp_slope_uv_per_s",
                    "pmp_mean_uv",
                    "pmp_peak_uv",
                    "pmp_area_uv_s",
                ],
            },
            {
                "analysis_family": "connectivity",
                "grouping_columns": ["group"],
                "include_flags": ["include_connectivity"],
                "variables": connectivity_columns(dataframe),
            },
        ]

    if experiment == "readysetgo":
        return [
            {
                "analysis_family": "behavior",
                "grouping_columns": ["group"],
                "include_flags": ["include_behavior"],
                "variables": [
                    "response_time_mean_ms",
                    "response_time_median_ms",
                    "response_time_sd_ms",
                    "accuracy_percent",
                ],
            },
            {
                "analysis_family": "cnv",
                "grouping_columns": ["group"],
                "include_flags": ["include_cnv"],
                "variables": [
                    "cnv_amplitude_uv",
                ],
            },
            {
                "analysis_family": "response_locked",
                "grouping_columns": ["group"],
                "include_flags": ["include_rp_pmp"],
                "variables": [
                    "rp_mean_uv",
                    "rp_peak_uv",
                    "rp_area_uv_s",
                    "rp_slope_uv_per_s",
                    "pmp_mean_uv",
                    "pmp_peak_uv",
                    "pmp_area_uv_s",
                ],
            },
            {
                "analysis_family": "set_locked_connectivity",
                "grouping_columns": ["group"],
                "include_flags": ["include_set_locked_connectivity"],
                "variables": connectivity_columns(dataframe),
            },
        ]

    if experiment == "tmt":
        return [
            {
                "analysis_family": "behavior",
                "grouping_columns": ["group", "tmt_variant"],
                "include_flags": [
                    "include_behavior",
                    "include_tmt_complete_pair",
                ],
                "variables": [
                    "task_duration_s",
                    "correct_click_count",
                    "error_click_count",
                    "total_click_count",
                    "error_percent",
                    "inter_click_interval_mean_s",
                    "inter_click_interval_median_s",
                    "inter_click_interval_sd_s",
                ],
            },
            {
                "analysis_family": "psd",
                "grouping_columns": ["group", "tmt_variant"],
                "include_flags": [
                    "include_psd",
                    "include_tmt_complete_pair",
                ],
                "variables": tmt_psd_columns(dataframe),
            },
            {
                "analysis_family": "response_locked",
                "grouping_columns": ["group", "tmt_variant"],
                "include_flags": [
                    "include_rp_pmp",
                    "include_tmt_complete_pair",
                ],
                "variables": [
                    "rp_mean_uv",
                    "rp_peak_uv",
                    "rp_area_uv_s",
                    "rp_slope_uv_per_s",
                    "pmp_mean_uv",
                    "pmp_peak_uv",
                    "pmp_area_uv_s",
                ],
            },
        ]

    raise ValueError(f"Unknown experiment: {experiment}")


def validate_plan(experiment, dataframe, plan):
    rows = []

    for family in plan:
        analysis_family = family["analysis_family"]

        for column in family["grouping_columns"]:
            if column not in dataframe.columns:
                rows.append(
                    {
                        "experiment": experiment,
                        "severity": "ERROR",
                        "analysis_family": analysis_family,
                        "variable": column,
                        "issue": "grouping_column_missing",
                        "details": "",
                    }
                )

        for flag in family["include_flags"]:
            if flag not in dataframe.columns:
                rows.append(
                    {
                        "experiment": experiment,
                        "severity": "ERROR",
                        "analysis_family": analysis_family,
                        "variable": flag,
                        "issue": "include_flag_missing",
                        "details": "",
                    }
                )

        for variable in family["variables"]:
            if variable not in dataframe.columns:
                rows.append(
                    {
                        "experiment": experiment,
                        "severity": "ERROR",
                        "analysis_family": analysis_family,
                        "variable": variable,
                        "issue": "analysis_variable_missing",
                        "details": "",
                    }
                )

        if len(family["variables"]) == 0:
            rows.append(
                {
                    "experiment": experiment,
                    "severity": "WARNING",
                    "analysis_family": analysis_family,
                    "variable": "",
                    "issue": "no_variables_in_family",
                    "details": "",
                }
            )

    return rows


def make_include_mask(dataframe, include_flags):
    mask = pd.Series(True, index=dataframe.index)

    for flag in include_flags:
        values = as_binary_flag(dataframe[flag])
        mask = mask & (values == 1)

    return mask


def summarize_values(values):
    numeric = pd.to_numeric(values, errors="coerce")

    valid = numeric.dropna()

    if len(valid) == 0:
        return {
            "valid_n": 0,
            "missing_n": int(numeric.isna().sum()),
            "mean": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    std_value = float(valid.std(ddof=1)) if len(valid) > 1 else np.nan
    sem_value = float(std_value / np.sqrt(len(valid))) if len(valid) > 1 else np.nan

    return {
        "valid_n": int(len(valid)),
        "missing_n": int(numeric.isna().sum()),
        "mean": float(valid.mean()),
        "std": std_value,
        "sem": sem_value,
        "median": float(valid.median()),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


def summarize_variable(
    experiment,
    dataframe,
    analysis_family,
    variable,
    grouping_columns,
    include_flags,
):
    include_mask = make_include_mask(
        dataframe=dataframe,
        include_flags=include_flags,
    )

    included = dataframe.loc[include_mask].copy()

    rows = []

    grouped = included.groupby(
        grouping_columns,
        dropna=False,
        sort=True,
    )

    for group_values, group_df in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_info = {
            column: value
            for column, value in zip(grouping_columns, group_values)
        }

        stats = summarize_values(group_df[variable])

        row = {
            "experiment": experiment,
            "analysis_family": analysis_family,
            "variable": variable,
            "include_flags": ", ".join(include_flags),
            "group": group_info.get("group", ""),
            "tmt_variant": group_info.get("tmt_variant", ""),
            "included_n": int(len(group_df)),
            **stats,
        }

        rows.append(row)

    return rows


def create_descriptives_for_experiment(experiment, dataframe):
    plan = get_experiment_plan(
        experiment=experiment,
        dataframe=dataframe,
    )

    report_rows = validate_plan(
        experiment=experiment,
        dataframe=dataframe,
        plan=plan,
    )

    if any(row["severity"] == "ERROR" for row in report_rows):
        return pd.DataFrame(), report_rows

    descriptive_rows = []

    for family in plan:
        for variable in family["variables"]:
            descriptive_rows.extend(
                summarize_variable(
                    experiment=experiment,
                    dataframe=dataframe,
                    analysis_family=family["analysis_family"],
                    variable=variable,
                    grouping_columns=family["grouping_columns"],
                    include_flags=family["include_flags"],
                )
            )

    descriptives = pd.DataFrame(descriptive_rows)

    return descriptives, report_rows


def create_summary(descriptives):
    if descriptives.empty:
        return pd.DataFrame(
            columns=[
                "experiment",
                "analysis_family",
                "variable_count",
                "row_count",
                "minimum_included_n",
                "minimum_valid_n",
            ]
        )

    summary = (
        descriptives.groupby(["experiment", "analysis_family"])
        .agg(
            variable_count=("variable", "nunique"),
            row_count=("variable", "count"),
            minimum_included_n=("included_n", "min"),
            minimum_valid_n=("valid_n", "min"),
        )
        .reset_index()
    )

    return summary


def main():
    all_descriptives = []
    all_report_rows = []

    for experiment, path in STATISTICS_READY_FILES.items():
        if not path.exists():
            all_report_rows.append(
                {
                    "experiment": experiment,
                    "severity": "ERROR",
                    "analysis_family": "",
                    "variable": "",
                    "issue": "statistics_ready_file_missing",
                    "details": str(path),
                }
            )
            continue

        dataframe = read_csv(path)

        descriptives, report_rows = create_descriptives_for_experiment(
            experiment=experiment,
            dataframe=dataframe,
        )

        all_report_rows.extend(report_rows)

        if descriptives.empty:
            continue

        all_descriptives.append(descriptives)

        experiment_output_path = OUTPUT_DIR / f"{experiment}_descriptives.csv"
        write_csv(descriptives, experiment_output_path)

        print(
            f"{experiment}: descriptives yazıldı "
            f"({len(descriptives)} satır): {experiment_output_path}"
        )

    if all_descriptives:
        combined = pd.concat(all_descriptives, ignore_index=True)
    else:
        combined = pd.DataFrame()

    all_output_path = OUTPUT_DIR / "all_descriptives.csv"
    write_csv(combined, all_output_path)

    report = pd.DataFrame(all_report_rows)

    if report.empty:
        report = pd.DataFrame(
            columns=[
                "experiment",
                "severity",
                "analysis_family",
                "variable",
                "issue",
                "details",
            ]
        )

    summary = create_summary(combined)

    report_path = QC_OUTPUT_DIR / "statistics_descriptives_report.csv"
    summary_path = QC_OUTPUT_DIR / "statistics_descriptives_summary.csv"

    write_csv(report, report_path)
    write_csv(summary, summary_path)

    errors = report[report["severity"] == "ERROR"]

    print("")
    print("Descriptive statistics summary:")
    if summary.empty:
        print("Özet boş.")
    else:
        print(summary.to_string(index=False))

    print("")
    print(f"All descriptives: {all_output_path}")
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