"""
Add effect sizes to NeuroCogEEG post-SPSS result tables.

This script is intentionally post-SPSS:
- It does not rerun EEG preprocessing.
- It does not rerun SPSS tests.
- It does not modify original extracted SPSS CSV files.

Outputs:
- outputs/statistics/effect_sizes/independent_effect_sizes.csv
- outputs/statistics/effect_sizes/independent_samples_tests_reporting_effect_sizes.csv
- outputs/statistics/effect_sizes/tmt_mixed_fixed_effects_reporting_effect_sizes.csv
- outputs/qc/effect_size_manifest.csv
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "statistics" / "effect_sizes"
QC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qc"

PREFERRED_INPUT_DIRS = [
    PROJECT_ROOT / "outputs" / "statistics" / "spss_results",
    PROJECT_ROOT / "outputs" / "statistics" / "spss_outputs",
    PROJECT_ROOT / "outputs" / "statistics",
    PROJECT_ROOT / "outputs" / "spss" / "outputs",
    PROJECT_ROOT / "data" / "spss" / "outputs",
]

CSV_KWARGS = {
    "sep": ";",
    "decimal": ",",
    "encoding": "utf-8-sig",
}


class EffectSizeError(RuntimeError):
    """Raised when expected post-SPSS files or columns are missing."""


def normalize_name(value: str) -> str:
    """Normalize column names from SPSS/Pandas/extracted reports."""
    text = str(value).strip().lower()
    text = text.replace("ı", "i")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, **CSV_KWARGS)


def write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False, **CSV_KWARGS)


def candidate_paths(filename: str) -> list[Path]:
    """Find all matching files, preferring known SPSS output locations."""
    found: list[Path] = []
    seen: set[Path] = set()

    for directory in PREFERRED_INPUT_DIRS:
        path = directory / filename
        if path.exists() and path not in seen:
            found.append(path)
            seen.add(path)

    outputs_dir = PROJECT_ROOT / "outputs"
    if outputs_dir.exists():
        for path in outputs_dir.rglob(filename):
            if path not in seen:
                found.append(path)
                seen.add(path)

    return found


def read_all_matching(filename: str, required: bool = True) -> pd.DataFrame:
    paths = candidate_paths(filename)

    if not paths:
        if required:
            raise EffectSizeError(
                f"Required SPSS result file not found under outputs/: {filename}"
            )
        return pd.DataFrame()

    frames = []
    for path in paths:
        frame = read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True, sort=False)


def find_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
    context: str = "",
) -> str | None:
    normalized_to_original = {
        normalize_name(column): column for column in dataframe.columns
    }

    for candidate in candidates:
        normalized = normalize_name(candidate)
        if normalized in normalized_to_original:
            return normalized_to_original[normalized]

    if required:
        available = ", ".join(str(column) for column in dataframe.columns)
        raise EffectSizeError(
            f"Column not found in {context or 'dataframe'}. "
            f"Tried: {list(candidates)}. Available columns: {available}"
        )

    return None


def to_numeric(series: pd.Series) -> pd.Series:
    """Convert SPSS-like numeric strings robustly."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("\u2212", "-", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9eE+\-.]", "", regex=True)
    )

    cleaned = cleaned.replace({"": np.nan, ".": np.nan, "-": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def group_kind(value) -> str | None:
    """Classify group labels used by the SPSS syntax: 1/control, 2/experimental."""
    text = str(value).strip().lower()
    text = text.replace("ı", "i")

    if text in {"1", "1.0"} or "control" in text or "kontrol" in text:
        return "control"

    if text in {"2", "2.0"} or "experimental" in text or "deney" in text:
        return "experimental"

    return None


def safe_float(value) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def cohen_d_from_groups(
    mean_experimental: float,
    sd_experimental: float,
    n_experimental: float,
    mean_control: float,
    sd_control: float,
    n_control: float,
) -> dict[str, float | str]:
    """Compute independent-groups Cohen's d and Hedges' g.

    Direction is experimental minus control.
    """
    mean_diff = mean_experimental - mean_control
    df = n_experimental + n_control - 2

    if (
        n_experimental < 2
        or n_control < 2
        or not np.isfinite(sd_experimental)
        or not np.isfinite(sd_control)
        or df <= 0
    ):
        return {
            "mean_difference_experimental_minus_control": mean_diff,
            "pooled_sd": np.nan,
            "cohens_d": np.nan,
            "hedges_g": np.nan,
            "glass_delta_control": np.nan,
            "effect_size_status": "not_computed_insufficient_group_sd_or_n",
        }

    pooled_variance = (
        ((n_experimental - 1) * (sd_experimental**2))
        + ((n_control - 1) * (sd_control**2))
    ) / df

    if pooled_variance <= 0 or not np.isfinite(pooled_variance):
        return {
            "mean_difference_experimental_minus_control": mean_diff,
            "pooled_sd": np.nan,
            "cohens_d": np.nan,
            "hedges_g": np.nan,
            "glass_delta_control": np.nan,
            "effect_size_status": "not_computed_invalid_pooled_sd",
        }

    pooled_sd = math.sqrt(pooled_variance)
    d_value = mean_diff / pooled_sd

    # Approximate small-sample correction J(df).
    hedges_correction = 1.0 - (3.0 / ((4.0 * df) - 1.0)) if df > 1 else np.nan
    hedges_g = d_value * hedges_correction if np.isfinite(hedges_correction) else np.nan

    glass_delta = mean_diff / sd_control if sd_control > 0 else np.nan

    return {
        "mean_difference_experimental_minus_control": mean_diff,
        "pooled_sd": pooled_sd,
        "cohens_d": d_value,
        "hedges_g": hedges_g,
        "glass_delta_control": glass_delta,
        "effect_size_status": "computed",
    }


def prepare_standard_keys(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add standardized merge keys when possible."""
    result = dataframe.copy()

    experiment_col = find_column(
        result,
        ["experiment", "task", "deney"],
        required=False,
        context="standard keys",
    )
    analysis_col = find_column(
        result,
        ["analysis_title", "analysis", "title", "table_title", "analysis_family"],
        required=False,
        context="standard keys",
    )
    variable_col = find_column(
        result,
        [
            "variable",
            "variable_name",
            "dependent_variable",
            "test_variable",
            "outcome",
            "measure",
        ],
        required=False,
        context="standard keys",
    )

    result["__experiment"] = (
        result[experiment_col].astype(str).str.strip() if experiment_col else ""
    )
    result["__analysis_title"] = (
        result[analysis_col].astype(str).str.strip() if analysis_col else ""
    )
    result["__variable"] = (
        result[variable_col].astype(str).str.strip() if variable_col else ""
    )

    return result


def compute_independent_effect_sizes(group_statistics: pd.DataFrame) -> pd.DataFrame:
    experiment_col = find_column(
        group_statistics,
        ["experiment", "task", "deney"],
        required=False,
        context="group_statistics.csv",
    )
    analysis_col = find_column(
        group_statistics,
        ["analysis_title", "analysis", "title", "table_title", "analysis_family"],
        required=False,
        context="group_statistics.csv",
    )
    variable_col = find_column(
        group_statistics,
        ["variable", "variable_name", "dependent_variable", "test_variable", "outcome"],
        context="group_statistics.csv",
    )
    group_col = find_column(
        group_statistics,
        ["group", "group_code", "grup", "grup_kodu"],
        context="group_statistics.csv",
    )
    n_col = find_column(
        group_statistics,
        ["n", "valid_n", "count"],
        context="group_statistics.csv",
    )
    mean_col = find_column(
        group_statistics,
        ["mean", "ortalama"],
        context="group_statistics.csv",
    )
    sd_col = find_column(
        group_statistics,
        ["std_deviation", "std. deviation", "standard_deviation", "sd", "ss"],
        context="group_statistics.csv",
    )

    data = group_statistics.copy()
    data["__n"] = to_numeric(data[n_col])
    data["__mean"] = to_numeric(data[mean_col])
    data["__sd"] = to_numeric(data[sd_col])
    data["__group_kind"] = data[group_col].map(group_kind)

    if experiment_col is None:
        data["__experiment_source"] = ""
        experiment_col = "__experiment_source"

    if analysis_col is None:
        data["__analysis_source"] = ""
        analysis_col = "__analysis_source"

    key_columns = [experiment_col, analysis_col, variable_col]
    rows = []

    for key_values, group in data.groupby(key_columns, dropna=False):
        experiment, analysis_title, variable = key_values

        control_rows = group[group["__group_kind"] == "control"]
        experimental_rows = group[group["__group_kind"] == "experimental"]

        if control_rows.empty or experimental_rows.empty:
            rows.append(
                {
                    "experiment": experiment,
                    "analysis_title": analysis_title,
                    "variable": variable,
                    "direction": "experimental_minus_control",
                    "effect_size_status": "not_computed_control_or_experimental_group_missing",
                }
            )
            continue

        control = control_rows.iloc[0]
        experimental = experimental_rows.iloc[0]

        n_control = safe_float(control["__n"])
        mean_control = safe_float(control["__mean"])
        sd_control = safe_float(control["__sd"])
        n_experimental = safe_float(experimental["__n"])
        mean_experimental = safe_float(experimental["__mean"])
        sd_experimental = safe_float(experimental["__sd"])

        effect_values = cohen_d_from_groups(
            mean_experimental=mean_experimental,
            sd_experimental=sd_experimental,
            n_experimental=n_experimental,
            mean_control=mean_control,
            sd_control=sd_control,
            n_control=n_control,
        )

        rows.append(
            {
                "experiment": experiment,
                "analysis_title": analysis_title,
                "variable": variable,
                "direction": "experimental_minus_control",
                "n_control": n_control,
                "mean_control": mean_control,
                "sd_control": sd_control,
                "n_experimental": n_experimental,
                "mean_experimental": mean_experimental,
                "sd_experimental": sd_experimental,
                **effect_values,
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise EffectSizeError("No independent effect-size rows could be created.")

    return result.sort_values(
        ["experiment", "analysis_title", "variable"],
        na_position="last",
    ).reset_index(drop=True)


def merge_independent_reporting(
    reporting: pd.DataFrame,
    effect_sizes: pd.DataFrame,
) -> pd.DataFrame:
    reporting_keys = prepare_standard_keys(reporting)
    effect_keys = prepare_standard_keys(effect_sizes)

    merge_keys = ["__experiment", "__analysis_title", "__variable"]

    value_columns = [
        "direction",
        "n_control",
        "mean_control",
        "sd_control",
        "n_experimental",
        "mean_experimental",
        "sd_experimental",
        "mean_difference_experimental_minus_control",
        "pooled_sd",
        "cohens_d",
        "hedges_g",
        "glass_delta_control",
        "effect_size_status",
    ]

    merged = reporting_keys.merge(
        effect_keys[merge_keys + value_columns],
        on=merge_keys,
        how="left",
    )

    return merged.drop(columns=merge_keys)


def compute_tmt_mixed_effect_sizes(fixed_effects: pd.DataFrame) -> pd.DataFrame:
    effect_col = find_column(
        fixed_effects,
        ["effect", "source", "fixed_effect", "term"],
        context="tmt_mixed_fixed_effects_reporting.csv",
    )
    f_col = find_column(
        fixed_effects,
        ["f", "f_value", "f statistic", "f_statistic"],
        context="tmt_mixed_fixed_effects_reporting.csv",
    )
    numerator_df_col = find_column(
        fixed_effects,
        [
            "numerator_df",
            "numerator df",
            "df1",
            "num_df",
            "df_effect",
        ],
        context="tmt_mixed_fixed_effects_reporting.csv",
    )
    denominator_df_col = find_column(
        fixed_effects,
        [
            "denominator_df",
            "denominator df",
            "df2",
            "den_df",
            "df_error",
        ],
        context="tmt_mixed_fixed_effects_reporting.csv",
    )

    result = fixed_effects.copy()
    f_values = to_numeric(result[f_col])
    numerator_df = to_numeric(result[numerator_df_col])
    denominator_df = to_numeric(result[denominator_df_col])

    denominator = (f_values * numerator_df) + denominator_df
    result["partial_eta_squared"] = np.where(
        denominator > 0,
        (f_values * numerator_df) / denominator,
        np.nan,
    )

    omega_denominator = denominator + 1
    result["partial_omega_squared"] = np.where(
        omega_denominator > 0,
        ((f_values - 1) * numerator_df) / omega_denominator,
        np.nan,
    )

    result["effect_size_status"] = np.where(
        result["partial_eta_squared"].notna(),
        "computed",
        "not_computed_missing_f_or_df",
    )

    # Put effect-size columns after the fixed-effect/source column when possible.
    ordered_columns = []
    for column in result.columns:
        ordered_columns.append(column)
        if column == effect_col:
            ordered_columns.extend(
                [
                    "partial_eta_squared",
                    "partial_omega_squared",
                    "effect_size_status",
                ]
            )

    ordered_columns = list(dict.fromkeys(ordered_columns))
    return result[ordered_columns]


def build_manifest_row(output_name: str, dataframe: pd.DataFrame, status_col: str | None = None) -> dict[str, object]:
    row = {
        "output": output_name,
        "rows": int(len(dataframe)),
        "status": "created" if len(dataframe) > 0 else "empty",
        "computed_rows": "",
        "not_computed_rows": "",
    }

    if status_col and status_col in dataframe.columns:
        status_counts = dataframe[status_col].value_counts(dropna=False).to_dict()
        row["computed_rows"] = int(status_counts.get("computed", 0))
        row["not_computed_rows"] = int(
            len(dataframe) - status_counts.get("computed", 0)
        )

    return row


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    group_statistics = read_all_matching("group_statistics.csv", required=True)
    independent_effect_sizes = compute_independent_effect_sizes(group_statistics)

    independent_path = OUTPUT_DIR / "independent_effect_sizes.csv"
    write_csv(independent_effect_sizes, independent_path)
    manifest_rows.append(
        build_manifest_row(
            "independent_effect_sizes.csv",
            independent_effect_sizes,
            "effect_size_status",
        )
    )

    independent_reporting = read_all_matching(
        "independent_samples_tests_reporting.csv",
        required=False,
    )

    if not independent_reporting.empty:
        independent_reporting_effect_sizes = merge_independent_reporting(
            independent_reporting,
            independent_effect_sizes,
        )
        independent_reporting_path = (
            OUTPUT_DIR / "independent_samples_tests_reporting_effect_sizes.csv"
        )
        write_csv(independent_reporting_effect_sizes, independent_reporting_path)
        manifest_rows.append(
            build_manifest_row(
                "independent_samples_tests_reporting_effect_sizes.csv",
                independent_reporting_effect_sizes,
                "effect_size_status",
            )
        )
    else:
        manifest_rows.append(
            {
                "output": "independent_samples_tests_reporting_effect_sizes.csv",
                "rows": 0,
                "status": "skipped_missing_independent_samples_tests_reporting_csv",
                "computed_rows": 0,
                "not_computed_rows": 0,
            }
        )

    tmt_fixed_effects = read_all_matching(
        "tmt_mixed_fixed_effects_reporting.csv",
        required=False,
    )

    if tmt_fixed_effects.empty:
        tmt_fixed_effects = read_all_matching(
            "tmt_mixed_fixed_effects.csv",
            required=True,
        )

    tmt_effect_sizes = compute_tmt_mixed_effect_sizes(tmt_fixed_effects)
    tmt_path = OUTPUT_DIR / "tmt_mixed_fixed_effects_reporting_effect_sizes.csv"
    write_csv(tmt_effect_sizes, tmt_path)
    manifest_rows.append(
        build_manifest_row(
            "tmt_mixed_fixed_effects_reporting_effect_sizes.csv",
            tmt_effect_sizes,
            "effect_size_status",
        )
    )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = QC_OUTPUT_DIR / "effect_size_manifest.csv"
    write_csv(manifest, manifest_path)

    print("Etki büyüklüğü dosyaları oluşturuldu.")
    print(f"Klasör: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")
    print("")
    print(manifest.to_string(index=False))

    failed = manifest[manifest["status"].isin(["empty"])]
    computed_total = pd.to_numeric(manifest["computed_rows"], errors="coerce").fillna(0).sum()

    if not failed.empty or computed_total == 0:
        print("")
        print("Sonuç: FAIL")
        raise SystemExit(1)

    print("")
    print("Sonuç: PASS")


if __name__ == "__main__":
    main()
