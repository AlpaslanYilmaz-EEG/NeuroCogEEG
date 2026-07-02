"""
SPSS output extraction pipeline for NeuroCogEEG.

This script extracts selected SPSS output tables from exported .xlsx files.

It does not run SPSS.
It does not run statistical tests.

Supported:
- Group Statistics
- Independent Samples Test
- TMT MIXED Type III Tests of Fixed Effects
- TMT Estimated Marginal Means

Outputs:
- outputs/statistics/spss_extracted/group_statistics.csv
- outputs/statistics/spss_extracted/independent_samples_tests.csv
- outputs/statistics/spss_extracted/independent_samples_tests_reporting.csv
- outputs/statistics/spss_extracted/tmt_mixed_fixed_effects.csv
- outputs/statistics/spss_extracted/tmt_mixed_fixed_effects_reporting.csv
- outputs/statistics/spss_extracted/tmt_estimated_marginal_means.csv
- outputs/qc/spss_result_extraction_manifest.csv
"""

import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SPSS_OUTPUT_FILES = {
    "flanker": PROJECT_ROOT
    / "outputs"
    / "statistics"
    / "spss_output"
    / "flanker_analysis_output.xlsx",
    "gonogo": PROJECT_ROOT
    / "outputs"
    / "statistics"
    / "spss_output"
    / "gonogo_analysis_output.xlsx",
    "readysetgo": PROJECT_ROOT
    / "outputs"
    / "statistics"
    / "spss_output"
    / "readysetgo_analysis_output.xlsx",
    "tmt": PROJECT_ROOT
    / "outputs"
    / "statistics"
    / "spss_output"
    / "tmt_analysis_output.xlsx",
}


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "statistics" / "spss_extracted"
QC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qc"


VARIABLE_FAMILIES = {
    "correct_rt_mean_ms": "behavior",
    "accuracy_percent": "behavior",
    "response_time_mean_ms": "behavior",
    "n2_amplitude_uv": "erp_n2_p3",
    "p3_amplitude_uv": "erp_n2_p3",
    "ern_amplitude_uv": "erp_ern",
    "cnv_amplitude_uv": "cnv",
    "rp_mean_uv": "response_locked",
    "pmp_peak_uv": "response_locked",
    "task_duration_s": "tmt_behavior",
    "error_percent": "tmt_behavior",
    "frontal_theta_relative_percent": "tmt_psd",
    "frontal_alpha_relative_percent": "tmt_psd",
    "frontal_beta_relative_percent": "tmt_psd",
    "parietal_occipital_theta_relative_percent": "tmt_psd",
    "parietal_occipital_alpha_relative_percent": "tmt_psd",
    "parietal_occipital_beta_relative_percent": "tmt_psd",
}


TMT_REPORTABLE_EFFECTS = {
    "group_code",
    "tmt_variant_code",
    "group_code * tmt_variant_code",
}


def write_csv(dataframe, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )


def read_spss_export_xlsx(path):
    return pd.read_excel(
        path,
        sheet_name=0,
        header=None,
    )


def is_blank(value):
    if pd.isna(value):
        return True

    return str(value).strip() == ""


def cell_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def to_float(value):
    if pd.isna(value):
        return None

    text = str(value).strip().replace(",", ".")

    text = re.sub(
        r"^([-+]?\d+(?:\.\d+)?)([A-Za-z]+)$",
        r"\1",
        text,
    )

    try:
        return float(text)
    except Exception:
        return None


def find_title_rows(dataframe, title):
    rows = []

    for row_index in range(len(dataframe)):
        value = cell_text(dataframe.iat[row_index, 0])

        if value == title or value.startswith(title):
            rows.append(row_index)

    return rows


def parse_dependent_variable(text):
    match = re.search(
        r"Dependent Variable:\s*([A-Za-z0-9_]+)",
        str(text),
    )

    if match:
        return match.group(1)

    return ""


def parse_group_statistics(experiment, dataframe):
    rows = []
    title_rows = find_title_rows(dataframe, "Group Statistics")

    for table_index, title_row in enumerate(title_rows, start=1):
        row_index = title_row + 2
        current_variable = ""

        while row_index < len(dataframe):
            variable = cell_text(dataframe.iat[row_index, 0])
            group = cell_text(dataframe.iat[row_index, 1])

            if is_blank(variable) and is_blank(group):
                break

            if variable:
                current_variable = variable

            if group:
                rows.append(
                    {
                        "experiment": experiment,
                        "table_index": table_index,
                        "analysis_family": VARIABLE_FAMILIES.get(
                            current_variable,
                            "unknown",
                        ),
                        "variable": current_variable,
                        "group": group,
                        "n": to_float(dataframe.iat[row_index, 2]),
                        "mean": to_float(dataframe.iat[row_index, 3]),
                        "std_deviation": to_float(dataframe.iat[row_index, 4]),
                        "std_error_mean": to_float(dataframe.iat[row_index, 5]),
                    }
                )

            row_index += 1

    return rows


def parse_independent_samples_tests(experiment, dataframe):
    rows = []
    title_rows = find_title_rows(dataframe, "Independent Samples Test")

    valid_assumptions = {
        "Equal variances assumed",
        "Equal variances not assumed",
    }

    for table_index, title_row in enumerate(title_rows, start=1):
        row_index = title_row + 4
        current_variable = ""

        while row_index < len(dataframe):
            variable = cell_text(dataframe.iat[row_index, 0])
            assumption = cell_text(dataframe.iat[row_index, 1])

            if is_blank(variable) and is_blank(assumption):
                break

            if variable:
                current_variable = variable

            if assumption in valid_assumptions:
                rows.append(
                    {
                        "experiment": experiment,
                        "table_index": table_index,
                        "analysis_family": VARIABLE_FAMILIES.get(
                            current_variable,
                            "unknown",
                        ),
                        "variable": current_variable,
                        "assumption": assumption,
                        "levene_f": to_float(dataframe.iat[row_index, 2]),
                        "levene_sig": to_float(dataframe.iat[row_index, 3]),
                        "t": to_float(dataframe.iat[row_index, 4]),
                        "df": to_float(dataframe.iat[row_index, 5]),
                        "p_two_tailed": to_float(dataframe.iat[row_index, 6]),
                        "mean_difference": to_float(dataframe.iat[row_index, 7]),
                        "std_error_difference": to_float(dataframe.iat[row_index, 8]),
                        "ci95_lower": to_float(dataframe.iat[row_index, 9]),
                        "ci95_upper": to_float(dataframe.iat[row_index, 10]),
                    }
                )

            row_index += 1

    return rows


def parse_tmt_mixed_fixed_effects(experiment, dataframe):
    rows = []
    title_rows = find_title_rows(dataframe, "Type III Tests of Fixed Effects")

    for table_index, title_row in enumerate(title_rows, start=1):
        row_index = title_row + 2
        data_rows = []
        dependent_variable = ""

        while row_index < len(dataframe):
            source = cell_text(dataframe.iat[row_index, 0])

            if source.startswith("a. Dependent Variable"):
                dependent_variable = parse_dependent_variable(source)
                break

            if is_blank(source):
                break

            if source != "Source":
                data_rows.append(
                    {
                        "row_index": row_index,
                        "effect": source,
                    }
                )

            row_index += 1

        if not dependent_variable:
            for search_index in range(row_index, min(row_index + 8, len(dataframe))):
                text = cell_text(dataframe.iat[search_index, 0])

                if text.startswith("a. Dependent Variable"):
                    dependent_variable = parse_dependent_variable(text)
                    break

        for data_row in data_rows:
            source_row = data_row["row_index"]
            effect = data_row["effect"]

            rows.append(
                {
                    "experiment": experiment,
                    "table_index": table_index,
                    "analysis_family": VARIABLE_FAMILIES.get(
                        dependent_variable,
                        "unknown",
                    ),
                    "variable": dependent_variable,
                    "effect": effect,
                    "numerator_df": to_float(dataframe.iat[source_row, 1]),
                    "denominator_df": to_float(dataframe.iat[source_row, 2]),
                    "f": to_float(dataframe.iat[source_row, 3]),
                    "p": to_float(dataframe.iat[source_row, 4]),
                }
            )

    return rows


def parse_tmt_estimated_marginal_means(experiment, dataframe):
    rows = []
    title_rows = find_title_rows(dataframe, "Estimated Marginal Means")

    for table_index, title_row in enumerate(title_rows, start=1):
        row_index = title_row + 5
        current_group = ""
        data_rows = []
        dependent_variable = ""

        while row_index < len(dataframe):
            group = cell_text(dataframe.iat[row_index, 0])
            tmt_variant = cell_text(dataframe.iat[row_index, 1])

            if group.startswith("a. Dependent Variable"):
                dependent_variable = parse_dependent_variable(group)
                break

            if is_blank(group) and is_blank(tmt_variant):
                break

            if group:
                current_group = group

            if tmt_variant in ["tmt1", "tmt2"]:
                data_rows.append(
                    {
                        "row_index": row_index,
                        "group": current_group,
                        "tmt_variant": tmt_variant,
                    }
                )

            row_index += 1

        if not dependent_variable:
            for search_index in range(row_index, min(row_index + 8, len(dataframe))):
                text = cell_text(dataframe.iat[search_index, 0])

                if text.startswith("a. Dependent Variable"):
                    dependent_variable = parse_dependent_variable(text)
                    break

        for data_row in data_rows:
            source_row = data_row["row_index"]

            rows.append(
                {
                    "experiment": experiment,
                    "table_index": table_index,
                    "analysis_family": VARIABLE_FAMILIES.get(
                        dependent_variable,
                        "unknown",
                    ),
                    "variable": dependent_variable,
                    "group": data_row["group"],
                    "tmt_variant": data_row["tmt_variant"],
                    "mean": to_float(dataframe.iat[source_row, 2]),
                    "std_error": to_float(dataframe.iat[source_row, 3]),
                    "df": to_float(dataframe.iat[source_row, 4]),
                    "ci95_lower": to_float(dataframe.iat[source_row, 5]),
                    "ci95_upper": to_float(dataframe.iat[source_row, 6]),
                }
            )

    return rows


def add_reporting_choice(test_table):
    if test_table.empty:
        return test_table

    output = test_table.copy()
    output["preferred_for_reporting"] = 0
    output["assumption_rule"] = ""

    group_columns = [
        "experiment",
        "analysis_family",
        "variable",
    ]

    for _, group_df in output.groupby(group_columns):
        equal_rows = group_df[
            group_df["assumption"] == "Equal variances assumed"
        ]

        unequal_rows = group_df[
            group_df["assumption"] == "Equal variances not assumed"
        ]

        if equal_rows.empty:
            continue

        equal_index = equal_rows.index[0]
        levene_sig = output.loc[equal_index, "levene_sig"]

        if levene_sig is None or pd.isna(levene_sig):
            chosen_index = equal_index
            rule = "levene_missing_report_equal_variances_assumed"
        elif float(levene_sig) >= 0.05:
            chosen_index = equal_index
            rule = "levene_sig_ge_0_05_report_equal_variances_assumed"
        else:
            if unequal_rows.empty:
                chosen_index = equal_index
                rule = "levene_sig_lt_0_05_but_unequal_row_missing"
            else:
                chosen_index = unequal_rows.index[0]
                rule = "levene_sig_lt_0_05_report_equal_variances_not_assumed"

        output.loc[chosen_index, "preferred_for_reporting"] = 1
        output.loc[chosen_index, "assumption_rule"] = rule

    return output


def add_tmt_reporting_choice(fixed_effects_table):
    if fixed_effects_table.empty:
        return fixed_effects_table

    output = fixed_effects_table.copy()
    output["preferred_for_reporting"] = output["effect"].isin(
        TMT_REPORTABLE_EFFECTS
    ).astype(int)

    return output


def extract_one_experiment(experiment, path):
    if not path.exists():
        return [], [], [], [], {
            "experiment": experiment,
            "status": "skipped_missing_xlsx",
            "xlsx_file": str(path),
            "group_statistics_rows": 0,
            "independent_samples_rows": 0,
            "tmt_mixed_fixed_effects_rows": 0,
            "tmt_emmeans_rows": 0,
            "details": "",
        }

    dataframe = read_spss_export_xlsx(path)

    group_rows = parse_group_statistics(
        experiment=experiment,
        dataframe=dataframe,
    )

    test_rows = parse_independent_samples_tests(
        experiment=experiment,
        dataframe=dataframe,
    )

    if experiment == "tmt":
        mixed_rows = parse_tmt_mixed_fixed_effects(
            experiment=experiment,
            dataframe=dataframe,
        )

        emmeans_rows = parse_tmt_estimated_marginal_means(
            experiment=experiment,
            dataframe=dataframe,
        )
    else:
        mixed_rows = []
        emmeans_rows = []

    status = "created"
    details = ""

    if experiment == "tmt":
        if len(mixed_rows) == 0:
            status = "error_no_tmt_mixed_fixed_effects"
            details = "TMT Type III Tests of Fixed Effects table could not be parsed."

        if len(emmeans_rows) == 0:
            status = "error_no_tmt_emmeans"
            details = (
                details
                + " TMT Estimated Marginal Means table could not be parsed."
            ).strip()

    else:
        if len(group_rows) == 0:
            status = "error_no_group_statistics"
            details = "Group Statistics table was not found or could not be parsed."

        if len(test_rows) == 0:
            status = "error_no_independent_samples_tests"
            details = (
                details
                + " Independent Samples Test table was not found or could not be parsed."
            ).strip()

    manifest_row = {
        "experiment": experiment,
        "status": status,
        "xlsx_file": str(path),
        "group_statistics_rows": len(group_rows),
        "independent_samples_rows": len(test_rows),
        "tmt_mixed_fixed_effects_rows": len(mixed_rows),
        "tmt_emmeans_rows": len(emmeans_rows),
        "details": details,
    }

    return group_rows, test_rows, mixed_rows, emmeans_rows, manifest_row


def main():
    all_group_rows = []
    all_test_rows = []
    all_mixed_rows = []
    all_emmeans_rows = []
    manifest_rows = []

    for experiment, path in SPSS_OUTPUT_FILES.items():
        (
            group_rows,
            test_rows,
            mixed_rows,
            emmeans_rows,
            manifest_row,
        ) = extract_one_experiment(
            experiment=experiment,
            path=path,
        )

        all_group_rows.extend(group_rows)
        all_test_rows.extend(test_rows)
        all_mixed_rows.extend(mixed_rows)
        all_emmeans_rows.extend(emmeans_rows)
        manifest_rows.append(manifest_row)

    group_statistics = pd.DataFrame(all_group_rows)

    independent_tests = pd.DataFrame(all_test_rows)
    independent_tests = add_reporting_choice(independent_tests)

    if independent_tests.empty:
        reporting_tests = pd.DataFrame()
    else:
        reporting_tests = independent_tests[
            independent_tests["preferred_for_reporting"] == 1
        ].copy()

    tmt_mixed_fixed_effects = pd.DataFrame(all_mixed_rows)
    tmt_mixed_fixed_effects = add_tmt_reporting_choice(
        tmt_mixed_fixed_effects
    )

    if tmt_mixed_fixed_effects.empty:
        tmt_reporting = pd.DataFrame()
    else:
        tmt_reporting = tmt_mixed_fixed_effects[
            tmt_mixed_fixed_effects["preferred_for_reporting"] == 1
        ].copy()

    tmt_emmeans = pd.DataFrame(all_emmeans_rows)

    manifest = pd.DataFrame(manifest_rows)

    group_statistics_path = OUTPUT_DIR / "group_statistics.csv"
    independent_tests_path = OUTPUT_DIR / "independent_samples_tests.csv"
    reporting_tests_path = OUTPUT_DIR / "independent_samples_tests_reporting.csv"
    tmt_fixed_path = OUTPUT_DIR / "tmt_mixed_fixed_effects.csv"
    tmt_reporting_path = OUTPUT_DIR / "tmt_mixed_fixed_effects_reporting.csv"
    tmt_emmeans_path = OUTPUT_DIR / "tmt_estimated_marginal_means.csv"
    manifest_path = QC_OUTPUT_DIR / "spss_result_extraction_manifest.csv"

    write_csv(group_statistics, group_statistics_path)
    write_csv(independent_tests, independent_tests_path)
    write_csv(reporting_tests, reporting_tests_path)
    write_csv(tmt_mixed_fixed_effects, tmt_fixed_path)
    write_csv(tmt_reporting, tmt_reporting_path)
    write_csv(tmt_emmeans, tmt_emmeans_path)
    write_csv(manifest, manifest_path)

    print("SPSS sonuç ayıklama tamamlandı.")
    print("")
    print("Manifest:")
    print(manifest.to_string(index=False))
    print("")
    print(f"Group statistics: {group_statistics_path}")
    print(f"Independent samples tests: {independent_tests_path}")
    print(f"Independent reporting rows: {reporting_tests_path}")
    print(f"TMT mixed fixed effects: {tmt_fixed_path}")
    print(f"TMT mixed reporting rows: {tmt_reporting_path}")
    print(f"TMT estimated marginal means: {tmt_emmeans_path}")
    print(f"Manifest file: {manifest_path}")

    errors = manifest[
        manifest["status"].astype(str).str.startswith("error")
    ]

    print("")
    print(f"ERROR sayısı: {len(errors)}")

    if not errors.empty:
        raise SystemExit(1)

    print("")
    print("Sonuç: PASS")


if __name__ == "__main__":
    main()