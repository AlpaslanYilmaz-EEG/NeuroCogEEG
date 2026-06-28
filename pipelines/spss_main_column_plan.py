"""
SPSS main column selection plan for NeuroCogEEG.

This script creates a column-selection plan for later SPSS main export files.

It does not modify original result CSV files.
It does not create SPSS main result files yet.

Output:
outputs/qc/spss_main_column_plan.csv
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


def read_csv(path):
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )


def get_keep_reason(row):
    column = row["column"]
    family = row["variable_family"]
    spss_include = row["spss_include"]

    if family == "metadata":
        return "metadata"

    if spss_include == "yes":
        return "main_analysis_variable"

    if column == "processing_error":
        return "processing_error_qc"

    if column.endswith("_observed_count"):
        return "minimum_trial_observed_count"

    if column.endswith("_minimum_required"):
        return "minimum_trial_required_count"

    if column.endswith("_minimum_met"):
        return "minimum_trial_pass_fail_flag"

    return ""


def should_keep(row):
    return get_keep_reason(row) != ""


def build_column_plan():
    if not DICTIONARY_PATH.exists():
        raise FileNotFoundError(
            f"Variable dictionary not found: {DICTIONARY_PATH}. "
            "Run pipelines/variable_dictionary_draft.py first."
        )

    dictionary = read_csv(DICTIONARY_PATH)

    plan_rows = []

    for experiment, result_path in RESULT_FILES.items():
        if not result_path.exists():
            print(f"UYARI: Sonuç dosyası bulunamadı: {result_path}")
            continue

        result_df = read_csv(result_path)

        experiment_dictionary = dictionary[
            dictionary["experiment"] == experiment
        ].copy()

        for _, row in experiment_dictionary.iterrows():
            column = row["column"]

            if column not in result_df.columns:
                plan_rows.append(
                    {
                        "experiment": experiment,
                        "column": column,
                        "keep": False,
                        "keep_reason": "column_missing_in_result_file",
                        "variable_family": row["variable_family"],
                        "unit": row["unit"],
                        "analysis_role": row["analysis_role"],
                        "spss_include": row["spss_include"],
                        "missing_allowed": row["missing_allowed"],
                        "missing_count": row["missing_count"],
                        "missing_percent": row["missing_percent"],
                    }
                )
                continue

            keep_reason = get_keep_reason(row)

            plan_rows.append(
                {
                    "experiment": experiment,
                    "column": column,
                    "keep": should_keep(row),
                    "keep_reason": keep_reason,
                    "variable_family": row["variable_family"],
                    "unit": row["unit"],
                    "analysis_role": row["analysis_role"],
                    "spss_include": row["spss_include"],
                    "missing_allowed": row["missing_allowed"],
                    "missing_count": row["missing_count"],
                    "missing_percent": row["missing_percent"],
                }
            )

    return pd.DataFrame(plan_rows)


def main():
    output_dir = PROJECT_ROOT / "outputs" / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = build_column_plan()

    output_path = output_dir / "spss_main_column_plan.csv"

    plan.to_csv(
        output_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )

    kept = plan[plan["keep"] == True]
    not_kept = plan[plan["keep"] == False]

    print("SPSS ana kolon planı oluşturuldu.")
    print(f"Dosya: {output_path}")
    print("")
    print("Deneylere göre tutulacak sütun sayısı:")
    print(kept.groupby("experiment")["column"].count())
    print("")
    print("Tutma gerekçelerine göre dağılım:")
    print(kept.groupby(["experiment", "keep_reason"])["column"].count())
    print("")
    print("Tutulmayan sütun sayısı:")
    print(not_kept.groupby("experiment")["column"].count())
    print("")
    print("Eksik kolon kontrolü:")
    missing = plan[plan["keep_reason"] == "column_missing_in_result_file"]
    if missing.empty:
        print("Eksik kolon yok.")
    else:
        print(missing[["experiment", "column"]].to_string(index=False))


if __name__ == "__main__":
    main()