"""
SPSS output extraction pipeline for NeuroCogEEG.

This script extracts selected SPSS output tables from exported .xlsx files.

It does not run SPSS.
It does not run statistical tests.

Currently supported:
- Group Statistics
- Independent Samples Test

Outputs:
- outputs/statistics/spss_extracted/group_statistics.csv
- outputs/statistics/spss_extracted/independent_samples_tests.csv
- outputs/statistics/spss_extracted/independent_samples_tests_reporting.csv
- outputs/qc/spss_result_extraction_manifest.csv
"""

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
    """
    Read the first sheet of an SPSS Excel export without assuming headers.
    """
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

    try:
        return float(value)
    except Exception:
        return None


def find_title_rows(dataframe, title):
    rows = []

    for row_index in range(len(dataframe)):
        value = cell_text(dataframe.iat[row_index, 0])

        if value == title:
            rows.append(row_index)

    return rows


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


def add_reporting_choice(test_table):
    """
    Choose the row to report based on Levene's test.

    Rule:
    - If Levene Sig. >= .05: report Equal variances assumed.
    - If Levene Sig. < .05: report Equal variances not assumed.
    """
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


def extract_one_experiment(experiment, path):
    if not path.exists():
        return [], [], {
            "experiment": experiment,
            "status": "skipped_missing_xlsx",
            "xlsx_file": str(path),
            "group_statistics_rows": 0,
            "independent_samples_rows": 0,
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

    status = "created"

    details = ""

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
        "details": details,
    }

    return group_rows, test_rows, manifest_row


def main():
    all_group_rows = []
    all_test_rows = []
    manifest_rows = []

    for experiment, path in SPSS_OUTPUT_FILES.items():
        group_rows, test_rows, manifest_row = extract_one_experiment(
            experiment=experiment,
            path=path,
        )

        all_group_rows.extend(group_rows)
        all_test_rows.extend(test_rows)
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

    manifest = pd.DataFrame(manifest_rows)

    group_statistics_path = OUTPUT_DIR / "group_statistics.csv"
    independent_tests_path = OUTPUT_DIR / "independent_samples_tests.csv"
    reporting_tests_path = OUTPUT_DIR / "independent_samples_tests_reporting.csv"
    manifest_path = QC_OUTPUT_DIR / "spss_result_extraction_manifest.csv"

    write_csv(group_statistics, group_statistics_path)
    write_csv(independent_tests, independent_tests_path)
    write_csv(reporting_tests, reporting_tests_path)
    write_csv(manifest, manifest_path)

    print("SPSS sonuç ayıklama tamamlandı.")
    print("")
    print("Manifest:")
    print(manifest.to_string(index=False))
    print("")
    print(f"Group statistics: {group_statistics_path}")
    print(f"Independent samples tests: {independent_tests_path}")
    print(f"Reporting rows: {reporting_tests_path}")
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