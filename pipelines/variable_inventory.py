"""
Variable inventory report for NeuroCogEEG outputs.

This script reads generated result CSV files and creates a variable-level
inventory table. It does not run EEG analysis and does not modify result files.
"""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULT_FILES = {
    "flanker": PROJECT_ROOT / "outputs" / "flanker" / "csv" / "flanker_results.csv",
    "gonogo": PROJECT_ROOT / "outputs" / "gonogo" / "csv" / "gonogo_results.csv",
    "readysetgo": PROJECT_ROOT / "outputs" / "readysetgo" / "csv" / "readysetgo_results.csv",
    "tmt": PROJECT_ROOT / "outputs" / "tmt" / "csv" / "tmt_results.csv",
}


def read_result_csv(path):
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )


def get_sample_values(series, max_values=5):
    values = series.dropna().unique()

    if len(values) == 0:
        return ""

    selected = values[:max_values]

    return " | ".join(str(value) for value in selected)


def infer_variable_family(column_name):
    name = column_name.lower()

    event_name_columns = {
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

    if column_name in ["subject_id", "group", "experiment", "tmt_variant"]:
        return "metadata"

    if "processing_error" in name:
        return "processing_error"

    if name in event_name_columns:
        return "event_count"

    if name.startswith("event_count_"):
        return "event_count"

    if name.startswith("raw_"):
        return "quality_control"

    if "_condition_" in name and name.endswith("_kept"):
        return "quality_control"

    if name.startswith("qc_"):
        return "quality_control"

    if "epochs_" in name:
        return "quality_control"

    if "minimum_" in name or "trials_" in name:
        return "quality_control"

    if (
        "rt" in name
        or "reaction" in name
        or "response_time" in name
    ):
        return "behavior_reaction_time"

    if (
        "accuracy" in name
        or "click" in name
        or "task_duration" in name
        or "error_percent" in name
        or "inter_click_interval" in name
    ):
        return "behavior"

    if name.startswith("n2_") or name.startswith("p3_") or name.startswith("ern_"):
        return "erp"

    if name.startswith("cnv_"):
        return "cnv"

    if name.startswith("rp_") or name.startswith("pmp_"):
        return "response_locked"

    if "relative_percent" in name or "absolute_power" in name:
        return "psd"

    if (
        name.startswith("coh_")
        or name.startswith("wpli_")
        or name.startswith("ciplv_")
        or name.startswith("z_coh_")
        or name.startswith("z_wpli_")
        or name.startswith("z_ciplv_")
    ):
        return "connectivity"

    return "other"

def build_inventory():
    rows = []

    for experiment, path in RESULT_FILES.items():
        if not path.exists():
            print(f"UYARI: Sonuç dosyası bulunamadı: {path}")
            continue

        df = read_result_csv(path)
        row_count = len(df)

        for column in df.columns:
            series = df[column]
            missing_count = int(series.isna().sum())
            non_missing_count = int(series.notna().sum())

            if row_count == 0:
                missing_percent = float("nan")
            else:
                missing_percent = float((missing_count / row_count) * 100.0)

            rows.append(
                {
                    "experiment": experiment,
                    "column": column,
                    "variable_family": infer_variable_family(column),
                    "dtype": str(series.dtype),
                    "row_count": row_count,
                    "non_missing_count": non_missing_count,
                    "missing_count": missing_count,
                    "missing_percent": missing_percent,
                    "sample_values": get_sample_values(series),
                }
            )

    return pd.DataFrame(rows)


def main():
    output_dir = PROJECT_ROOT / "outputs" / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory()

    output_path = output_dir / "variable_inventory.csv"

    inventory.to_csv(
        output_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )

    print("Değişken envanteri oluşturuldu.")
    print(f"Dosya: {output_path}")
    print("")
    print("Deneylere göre sütun sayıları:")
    print(inventory.groupby("experiment")["column"].count())
    print("")
    print("Değişken ailelerine göre sayılar:")
    print(inventory.groupby(["experiment", "variable_family"])["column"].count())


if __name__ == "__main__":
    main()