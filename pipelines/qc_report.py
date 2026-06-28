from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULT_FILES = {
    "flanker": PROJECT_ROOT / "outputs" / "flanker" / "csv" / "flanker_results.csv",
    "gonogo": PROJECT_ROOT / "outputs" / "gonogo" / "csv" / "gonogo_results.csv",
    "readysetgo": PROJECT_ROOT
    / "outputs"
    / "readysetgo"
    / "csv"
    / "readysetgo_results.csv",
    "tmt": PROJECT_ROOT / "outputs" / "tmt" / "csv" / "tmt_results.csv",
}


def read_result_csv(path: Path) -> pd.DataFrame:
    """
    Read an SPSS-compatible NeuroCogEEG result CSV.

    Parameters
    ----------
    path:
        CSV file path.

    Returns
    -------
    pd.DataFrame
        Loaded result table.
    """
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )


def value_counts_as_text(
    dataframe: pd.DataFrame,
    column: str,
) -> str:
    """
    Convert value counts of a column to compact text.

    Parameters
    ----------
    dataframe:
        Input DataFrame.

    column:
        Column name.

    Returns
    -------
    str
        Compact value-count summary.
    """
    if column not in dataframe.columns:
        return ""

    counts = dataframe[column].value_counts(dropna=False)

    return "; ".join(
        f"{index}={count}"
        for index, count in counts.items()
    )


def count_processing_errors(dataframe: pd.DataFrame) -> int:
    """
    Count non-empty processing errors.

    Parameters
    ----------
    dataframe:
        Input DataFrame.

    Returns
    -------
    int
        Processing error count.
    """
    if "processing_error" not in dataframe.columns:
        return 0

    errors = dataframe["processing_error"]

    return int(
        errors.notna()
        & (errors.astype(str).str.strip() != "")
    )


def normalize_bool_series(series: pd.Series) -> pd.Series:
    """
    Convert a mixed boolean/string series into nullable booleans.

    Parameters
    ----------
    series:
        Input Series.

    Returns
    -------
    pd.Series
        Nullable boolean Series.
    """
    if series.dtype == bool:
        return series

    text = series.astype(str).str.strip().str.lower()

    return text.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "pass": True,
            "fail": False,
        }
    )


def summarize_dataset(
    experiment: str,
    dataframe: pd.DataFrame,
    source_path: Path,
) -> dict[str, Any]:
    """
    Create dataset-level summary row.

    Parameters
    ----------
    experiment:
        Experiment name.

    dataframe:
        Result DataFrame.

    source_path:
        Source CSV path.

    Returns
    -------
    dict[str, Any]
        Dataset summary row.
    """
    return {
        "experiment": experiment,
        "source_file": str(source_path),
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "group_counts": value_counts_as_text(dataframe, "group"),
        "variant_counts": value_counts_as_text(dataframe, "tmt_variant"),
        "processing_error_count": count_processing_errors(dataframe),
    }


def summarize_missing_values(
    experiment: str,
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Summarize columns that contain missing values.

    Parameters
    ----------
    experiment:
        Experiment name.

    dataframe:
        Result DataFrame.

    Returns
    -------
    list[dict[str, Any]]
        Missing-value summary rows.
    """
    rows: list[dict[str, Any]] = []

    row_count = len(dataframe)

    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())

        if missing_count == 0:
            continue

        missing_percent = (
            float((missing_count / row_count) * 100.0)
            if row_count > 0
            else float("nan")
        )

        rows.append(
            {
                "experiment": experiment,
                "column": column,
                "missing_count": missing_count,
                "missing_percent": missing_percent,
            }
        )

    return rows


def summarize_minimum_trial_checks(
    experiment: str,
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Summarize minimum-trial QC columns.

    Parameters
    ----------
    experiment:
        Experiment name.

    dataframe:
        Result DataFrame.

    Returns
    -------
    list[dict[str, Any]]
        Minimum-trial summary rows.
    """
    rows: list[dict[str, Any]] = []

    minimum_columns = [
        column
        for column in dataframe.columns
        if column.endswith("_minimum_met")
    ]

    for column in minimum_columns:
        values = normalize_bool_series(dataframe[column])

        pass_count = int((values == True).sum())
        fail_count = int((values == False).sum())
        missing_count = int(values.isna().sum())

        rows.append(
            {
                "experiment": experiment,
                "minimum_column": column,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "missing_count": missing_count,
            }
        )

    return rows


def build_text_report(
    dataset_summary: pd.DataFrame,
    missing_summary: pd.DataFrame,
    minimum_summary: pd.DataFrame,
) -> str:
    """
    Build a human-readable QC report.

    Parameters
    ----------
    dataset_summary:
        Dataset-level summary table.

    missing_summary:
        Missing-value summary table.

    minimum_summary:
        Minimum-trial summary table.

    Returns
    -------
    str
        Text report.
    """
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("NeuroCogEEG Genel Kalite Kontrol Raporu")
    lines.append("=" * 80)
    lines.append("")

    lines.append("1) Veri seti özeti")
    lines.append("-" * 80)

    for _, row in dataset_summary.iterrows():
        lines.append(f"Deney: {row['experiment']}")
        lines.append(f"  Satır sayısı: {row['row_count']}")
        lines.append(f"  Sütun sayısı: {row['column_count']}")
        lines.append(f"  Gruplar: {row['group_counts']}")
        if str(row.get("variant_counts", "")).strip():
            lines.append(f"  Varyantlar: {row['variant_counts']}")
        lines.append(f"  Processing error: {row['processing_error_count']}")
        lines.append("")

    lines.append("")
    lines.append("2) Minimum trial kontrolü")
    lines.append("-" * 80)

    if minimum_summary.empty:
        lines.append("Minimum trial kontrol sütunu bulunamadı.")
    else:
        for _, row in minimum_summary.iterrows():
            lines.append(
                f"{row['experiment']} | {row['minimum_column']} | "
                f"PASS={row['pass_count']} | "
                f"FAIL={row['fail_count']} | "
                f"MISSING={row['missing_count']}"
            )

    lines.append("")
    lines.append("")
    lines.append("3) Eksik değer özeti")
    lines.append("-" * 80)

    if missing_summary.empty:
        lines.append("Eksik değer bulunan sütun yok.")
    else:
        for experiment in missing_summary["experiment"].unique():
            experiment_missing = missing_summary[
                missing_summary["experiment"] == experiment
            ]

            lines.append(f"Deney: {experiment}")

            for _, row in experiment_missing.iterrows():
                lines.append(
                    f"  {row['column']}: "
                    f"{row['missing_count']} "
                    f"({row['missing_percent']:.2f}%)"
                )

            lines.append("")

    return "\n".join(lines)


def run_qc_report() -> Path:
    """
    Run QC report generation.

    Returns
    -------
    Path
        Path to the generated text report.
    """
    output_dir = PROJECT_ROOT / "outputs" / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    minimum_rows: list[dict[str, Any]] = []

    for experiment, path in RESULT_FILES.items():
        if not path.exists():
            print(f"⚠️ Sonuç dosyası bulunamadı: {path}")
            continue

        dataframe = read_result_csv(path)

        dataset_rows.append(
            summarize_dataset(
                experiment=experiment,
                dataframe=dataframe,
                source_path=path,
            )
        )

        missing_rows.extend(
            summarize_missing_values(
                experiment=experiment,
                dataframe=dataframe,
            )
        )

        minimum_rows.extend(
            summarize_minimum_trial_checks(
                experiment=experiment,
                dataframe=dataframe,
            )
        )

    dataset_summary = pd.DataFrame(dataset_rows)
    missing_summary = pd.DataFrame(missing_rows)
    minimum_summary = pd.DataFrame(minimum_rows)

    dataset_summary_path = output_dir / "qc_dataset_summary.csv"
    missing_summary_path = output_dir / "qc_missing_summary.csv"
    minimum_summary_path = output_dir / "qc_minimum_trials_summary.csv"
    report_path = output_dir / "qc_report.log"

    dataset_summary.to_csv(
        dataset_summary_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )

    missing_summary.to_csv(
        missing_summary_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )

    minimum_summary.to_csv(
        minimum_summary_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )

    report_text = build_text_report(
        dataset_summary=dataset_summary,
        missing_summary=missing_summary,
        minimum_summary=minimum_summary,
    )

    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print("")
    print(f"QC raporu kaydedildi: {report_path}")
    print(f"Dataset özeti: {dataset_summary_path}")
    print(f"Eksik değer özeti: {missing_summary_path}")
    print(f"Minimum trial özeti: {minimum_summary_path}")

    return report_path


if __name__ == "__main__":
    run_qc_report()