"""
Report table generator for NeuroCogEEG statistical outputs.

This script converts extracted SPSS results into cleaner report-ready tables.

It does not run statistics.
It does not run SPSS.

Inputs:
- outputs/statistics/spss_extracted/group_statistics.csv
- outputs/statistics/spss_extracted/independent_samples_tests_reporting.csv
- outputs/statistics/spss_extracted/tmt_mixed_fixed_effects_reporting.csv
- outputs/statistics/spss_extracted/tmt_estimated_marginal_means.csv

Outputs:
- outputs/statistics/report_tables/between_subject_results_table.csv
- outputs/statistics/report_tables/tmt_mixed_fixed_effects_table.csv
- outputs/statistics/report_tables/tmt_estimated_marginal_means_table.csv
- outputs/qc/report_tables_manifest.csv
"""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXTRACTED_DIR = PROJECT_ROOT / "outputs" / "statistics" / "spss_extracted"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "statistics" / "report_tables"
QC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qc"


GROUP_STATISTICS_PATH = EXTRACTED_DIR / "group_statistics.csv"
INDEPENDENT_REPORTING_PATH = EXTRACTED_DIR / "independent_samples_tests_reporting.csv"
TMT_FIXED_REPORTING_PATH = EXTRACTED_DIR / "tmt_mixed_fixed_effects_reporting.csv"
TMT_EMMEANS_PATH = EXTRACTED_DIR / "tmt_estimated_marginal_means.csv"


EXPERIMENT_LABELS = {
    "flanker": "Flanker",
    "gonogo": "Go/No-Go",
    "readysetgo": "ReadySetGo",
    "tmt": "TMT",
}


VARIABLE_LABELS = {
    "correct_rt_mean_ms": "Mean correct RT (ms)",
    "response_time_mean_ms": "Mean response time (ms)",
    "accuracy_percent": "Accuracy (%)",
    "n2_amplitude_uv": "N2 amplitude (µV)",
    "p3_amplitude_uv": "P3 amplitude (µV)",
    "ern_amplitude_uv": "ERN amplitude (µV)",
    "cnv_amplitude_uv": "CNV amplitude (µV)",
    "rp_mean_uv": "RP mean amplitude (µV)",
    "pmp_peak_uv": "PMP peak amplitude (µV)",
    "task_duration_s": "Task duration (s)",
    "error_percent": "Error (%)",
    "frontal_theta_relative_percent": "Frontal theta relative power (%)",
    "frontal_alpha_relative_percent": "Frontal alpha relative power (%)",
    "frontal_beta_relative_percent": "Frontal beta relative power (%)",
    "parietal_occipital_theta_relative_percent": (
        "Parietal-occipital theta relative power (%)"
    ),
    "parietal_occipital_alpha_relative_percent": (
        "Parietal-occipital alpha relative power (%)"
    ),
    "parietal_occipital_beta_relative_percent": (
        "Parietal-occipital beta relative power (%)"
    ),
}


EFFECT_LABELS = {
    "group_code": "Group",
    "tmt_variant_code": "TMT variant",
    "group_code * tmt_variant_code": "Group × TMT variant",
}


GROUP_LABELS = {
    "1": "control",
    "2": "experimental",
    "control": "control",
    "experimental": "experimental",
}


VARIANT_LABELS = {
    "1": "tmt1",
    "2": "tmt2",
    "tmt1": "tmt1",
    "tmt2": "tmt2",
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


def safe_float(value):
    if pd.isna(value):
        return None

    try:
        return float(value)
    except Exception:
        return None


def round_or_blank(value, digits=3):
    number = safe_float(value)

    if number is None:
        return ""

    return round(number, digits)


def format_p(value):
    number = safe_float(value)

    if number is None:
        return ""

    if number < 0.001:
        return "< .001"

    return f"{number:.3f}"


def format_mean_sd(mean, sd, n):
    mean_value = safe_float(mean)
    sd_value = safe_float(sd)
    n_value = safe_float(n)

    if mean_value is None or sd_value is None or n_value is None:
        return ""

    return f"{mean_value:.3f} ± {sd_value:.3f} (n={int(n_value)})"


def normalize_group_name(value):
    text = str(value).strip()

    return GROUP_LABELS.get(text, text)


def normalize_variant_name(value):
    text = str(value).strip()

    return VARIANT_LABELS.get(text, text)


def get_group_stats_row(group_statistics, experiment, variable, group):
    group_name = normalize_group_name(group)

    rows = group_statistics[
        (group_statistics["experiment"] == experiment)
        & (group_statistics["variable"] == variable)
        & (group_statistics["group"].astype(str) == group_name)
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def create_between_subject_results(group_statistics, independent_reporting):
    rows = []

    if independent_reporting.empty:
        return pd.DataFrame()

    for _, test_row in independent_reporting.iterrows():
        experiment = test_row["experiment"]
        variable = test_row["variable"]

        control_stats = get_group_stats_row(
            group_statistics=group_statistics,
            experiment=experiment,
            variable=variable,
            group="control",
        )

        experimental_stats = get_group_stats_row(
            group_statistics=group_statistics,
            experiment=experiment,
            variable=variable,
            group="experimental",
        )

        control_summary = ""
        experimental_summary = ""
        control_n = ""
        experimental_n = ""

        if control_stats is not None:
            control_summary = format_mean_sd(
                control_stats["mean"],
                control_stats["std_deviation"],
                control_stats["n"],
            )
            control_n = int(safe_float(control_stats["n"]))

        if experimental_stats is not None:
            experimental_summary = format_mean_sd(
                experimental_stats["mean"],
                experimental_stats["std_deviation"],
                experimental_stats["n"],
            )
            experimental_n = int(safe_float(experimental_stats["n"]))

        rows.append(
            {
                "experiment": experiment,
                "experiment_label": EXPERIMENT_LABELS.get(experiment, experiment),
                "analysis_family": test_row["analysis_family"],
                "variable": variable,
                "variable_label": VARIABLE_LABELS.get(variable, variable),
                "control_n": control_n,
                "control_mean_sd": control_summary,
                "experimental_n": experimental_n,
                "experimental_mean_sd": experimental_summary,
                "assumption": test_row["assumption"],
                "levene_p": round_or_blank(test_row["levene_sig"], 3),
                "t": round_or_blank(test_row["t"], 3),
                "df": round_or_blank(test_row["df"], 3),
                "p": round_or_blank(test_row["p_two_tailed"], 3),
                "p_formatted": format_p(test_row["p_two_tailed"]),
                "mean_difference": round_or_blank(test_row["mean_difference"], 3),
                "ci95_lower": round_or_blank(test_row["ci95_lower"], 3),
                "ci95_upper": round_or_blank(test_row["ci95_upper"], 3),
                "assumption_rule": test_row["assumption_rule"],
            }
        )

    return pd.DataFrame(rows)


def create_tmt_fixed_effects_table(tmt_fixed):
    if tmt_fixed.empty:
        return pd.DataFrame()

    output = tmt_fixed.copy()

    output["experiment_label"] = "TMT"
    output["variable_label"] = output["variable"].map(
        lambda value: VARIABLE_LABELS.get(value, value)
    )
    output["effect_label"] = output["effect"].map(
        lambda value: EFFECT_LABELS.get(value, value)
    )

    keep_columns = [
        "experiment",
        "experiment_label",
        "analysis_family",
        "variable",
        "variable_label",
        "effect",
        "effect_label",
        "numerator_df",
        "denominator_df",
        "f",
        "p",
    ]

    output = output[keep_columns].copy()

    output["numerator_df"] = output["numerator_df"].map(
        lambda value: round_or_blank(value, 3)
    )
    output["denominator_df"] = output["denominator_df"].map(
        lambda value: round_or_blank(value, 3)
    )
    output["f"] = output["f"].map(
        lambda value: round_or_blank(value, 3)
    )
    output["p_raw"] = output["p"]
    output["p"] = output["p"].map(
        lambda value: round_or_blank(value, 3)
    )
    output["p_formatted"] = output["p_raw"].map(format_p)
    output = output.drop(columns=["p_raw"])

    return output


def create_tmt_emmeans_table(tmt_emmeans):
    if tmt_emmeans.empty:
        return pd.DataFrame()

    output = tmt_emmeans.copy()

    output["experiment_label"] = "TMT"
    output["variable_label"] = output["variable"].map(
        lambda value: VARIABLE_LABELS.get(value, value)
    )
    output["group_label"] = output["group"].map(normalize_group_name)
    output["tmt_variant_label"] = output["tmt_variant"].map(normalize_variant_name)

    keep_columns = [
        "experiment",
        "experiment_label",
        "analysis_family",
        "variable",
        "variable_label",
        "group",
        "group_label",
        "tmt_variant",
        "tmt_variant_label",
        "mean",
        "std_error",
        "df",
        "ci95_lower",
        "ci95_upper",
    ]

    output = output[keep_columns].copy()

    for column in ["mean", "std_error", "df", "ci95_lower", "ci95_upper"]:
        output[column] = output[column].map(
            lambda value: round_or_blank(value, 3)
        )

    return output


def validate_inputs():
    rows = []

    input_files = {
        "group_statistics": GROUP_STATISTICS_PATH,
        "independent_reporting": INDEPENDENT_REPORTING_PATH,
        "tmt_fixed_reporting": TMT_FIXED_REPORTING_PATH,
        "tmt_emmeans": TMT_EMMEANS_PATH,
    }

    for name, path in input_files.items():
        if not path.exists():
            rows.append(
                {
                    "file_key": name,
                    "severity": "ERROR",
                    "issue": "input_file_missing",
                    "path": str(path),
                    "details": "",
                }
            )
        elif path.stat().st_size == 0:
            rows.append(
                {
                    "file_key": name,
                    "severity": "ERROR",
                    "issue": "input_file_empty",
                    "path": str(path),
                    "details": "",
                }
            )
        else:
            rows.append(
                {
                    "file_key": name,
                    "severity": "INFO",
                    "issue": "input_file_ok",
                    "path": str(path),
                    "details": f"size_bytes={path.stat().st_size}",
                }
            )

    return rows


def main():
    manifest_rows = validate_inputs()

    errors = [
        row for row in manifest_rows
        if row["severity"] == "ERROR"
    ]

    if errors:
        manifest = pd.DataFrame(manifest_rows)
        manifest_path = QC_OUTPUT_DIR / "report_tables_manifest.csv"
        write_csv(manifest, manifest_path)
        print(manifest.to_string(index=False))
        raise SystemExit(1)

    group_statistics = read_csv(GROUP_STATISTICS_PATH)
    independent_reporting = read_csv(INDEPENDENT_REPORTING_PATH)
    tmt_fixed = read_csv(TMT_FIXED_REPORTING_PATH)
    tmt_emmeans = read_csv(TMT_EMMEANS_PATH)

    between_subject = create_between_subject_results(
        group_statistics=group_statistics,
        independent_reporting=independent_reporting,
    )

    tmt_fixed_table = create_tmt_fixed_effects_table(tmt_fixed)
    tmt_emmeans_table = create_tmt_emmeans_table(tmt_emmeans)

    between_path = OUTPUT_DIR / "between_subject_results_table.csv"
    tmt_fixed_path = OUTPUT_DIR / "tmt_mixed_fixed_effects_table.csv"
    tmt_emmeans_path = OUTPUT_DIR / "tmt_estimated_marginal_means_table.csv"

    write_csv(between_subject, between_path)
    write_csv(tmt_fixed_table, tmt_fixed_path)
    write_csv(tmt_emmeans_table, tmt_emmeans_path)

    output_files = {
        "between_subject_results_table": between_path,
        "tmt_mixed_fixed_effects_table": tmt_fixed_path,
        "tmt_estimated_marginal_means_table": tmt_emmeans_path,
    }

    for name, path in output_files.items():
        manifest_rows.append(
            {
                "file_key": name,
                "severity": "INFO",
                "issue": "output_file_created",
                "path": str(path),
                "details": f"size_bytes={path.stat().st_size}",
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = QC_OUTPUT_DIR / "report_tables_manifest.csv"
    write_csv(manifest, manifest_path)

    print("Raporlanabilir sonuç tabloları oluşturuldu.")
    print("")
    print(f"Between-subject rows: {len(between_subject)}")
    print(f"TMT fixed-effect rows: {len(tmt_fixed_table)}")
    print(f"TMT EMM rows: {len(tmt_emmeans_table)}")
    print("")
    print(f"Between-subject table: {between_path}")
    print(f"TMT fixed effects table: {tmt_fixed_path}")
    print(f"TMT EMM table: {tmt_emmeans_path}")
    print(f"Manifest: {manifest_path}")
    print("")
    print("Sonuç: PASS")


if __name__ == "__main__":
    main()