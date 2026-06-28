"""
CSV-based summary figure pipeline for NeuroCogEEG.

This script creates summary bar plots from validated SPSS main CSV files.

It does not read EDF files.
It does not compute EEG features.
It only visualizes already-exported numerical results.

Outputs are saved under:

outputs/<experiment>/figures/summary/
outputs/qc/csv_summary_figure_manifest.csv
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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


EXPERIMENT_LABELS = {
    "flanker": "Flanker",
    "gonogo": "Go/No-Go",
    "readysetgo": "ReadySetGo",
    "tmt": "TMT",
}


METRIC_LABELS = {
    "correct_rt_mean_ms": "Mean correct RT",
    "response_time_mean_ms": "Mean response time",
    "accuracy_percent": "Accuracy",
    "n2_amplitude_uv": "N2 amplitude",
    "p3_amplitude_uv": "P3 amplitude",
    "ern_amplitude_uv": "ERN amplitude",
    "cnv_amplitude_uv": "CNV amplitude",
    "rp_mean_uv": "RP mean amplitude",
    "pmp_peak_uv": "PMP peak amplitude",
    "task_duration_s": "Task duration",
    "correct_click_count": "Correct click count",
    "error_click_count": "Error click count",
    "error_percent": "Error percent",
    "inter_click_interval_mean_s": "Mean inter-click interval",
    "frontal_theta_relative_percent": "Frontal theta relative power",
    "frontal_alpha_relative_percent": "Frontal alpha relative power",
    "frontal_beta_relative_percent": "Frontal beta relative power",
    "parietal_occipital_theta_relative_percent": "Parietal-occipital theta relative power",
    "parietal_occipital_alpha_relative_percent": "Parietal-occipital alpha relative power",
    "parietal_occipital_beta_relative_percent": "Parietal-occipital beta relative power",
}


METRIC_UNITS = {
    "correct_rt_mean_ms": "ms",
    "response_time_mean_ms": "ms",
    "accuracy_percent": "%",
    "n2_amplitude_uv": "µV",
    "p3_amplitude_uv": "µV",
    "ern_amplitude_uv": "µV",
    "cnv_amplitude_uv": "µV",
    "rp_mean_uv": "µV",
    "pmp_peak_uv": "µV",
    "task_duration_s": "s",
    "correct_click_count": "count",
    "error_click_count": "count",
    "error_percent": "%",
    "inter_click_interval_mean_s": "s",
    "frontal_theta_relative_percent": "%",
    "frontal_alpha_relative_percent": "%",
    "frontal_beta_relative_percent": "%",
    "parietal_occipital_theta_relative_percent": "%",
    "parietal_occipital_alpha_relative_percent": "%",
    "parietal_occipital_beta_relative_percent": "%",
}


PLOT_CONFIG = {
    "flanker": {
        "group_columns": ["group"],
        "metrics": [
            "correct_rt_mean_ms",
            "accuracy_percent",
            "n2_amplitude_uv",
            "p3_amplitude_uv",
            "ern_amplitude_uv",
            "rp_mean_uv",
            "pmp_peak_uv",
        ],
    },
    "gonogo": {
        "group_columns": ["group"],
        "metrics": [
            "correct_rt_mean_ms",
            "accuracy_percent",
            "n2_amplitude_uv",
            "p3_amplitude_uv",
            "rp_mean_uv",
            "pmp_peak_uv",
        ],
    },
    "readysetgo": {
        "group_columns": ["group"],
        "metrics": [
            "response_time_mean_ms",
            "accuracy_percent",
            "cnv_amplitude_uv",
            "rp_mean_uv",
            "pmp_peak_uv",
        ],
    },
    "tmt": {
        "group_columns": ["group", "tmt_variant"],
        "metrics": [
            "task_duration_s",
            "correct_click_count",
            "error_click_count",
            "error_percent",
            "inter_click_interval_mean_s",
            "frontal_theta_relative_percent",
            "frontal_alpha_relative_percent",
            "frontal_beta_relative_percent",
            "parietal_occipital_theta_relative_percent",
            "parietal_occipital_alpha_relative_percent",
            "parietal_occipital_beta_relative_percent",
            "rp_mean_uv",
            "pmp_peak_uv",
        ],
    },
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


def safe_filename(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def get_metric_label(metric):
    return METRIC_LABELS.get(metric, metric)


def get_metric_unit(metric):
    return METRIC_UNITS.get(metric, "")


def make_group_label(row, group_columns):
    parts = [str(row[column]) for column in group_columns]
    return " / ".join(parts)


def summarize_metric(dataframe, metric, group_columns):
    work = dataframe[group_columns + [metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")

    summary = (
        work.groupby(group_columns, dropna=False)[metric]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["sem"] = summary["sem"].fillna(0.0)

    summary["group_label"] = summary.apply(
        lambda row: make_group_label(row, group_columns),
        axis=1,
    )

    summary["group_label_with_n"] = summary.apply(
        lambda row: f"{row['group_label']}\n(n={int(row['count'])})",
        axis=1,
    )

    return summary


def save_bar_plot(summary, experiment, metric, output_path):
    metric_label = get_metric_label(metric)
    unit = get_metric_unit(metric)

    x_labels = summary["group_label_with_n"].tolist()
    means = summary["mean"].to_numpy(dtype=float)
    sems = summary["sem"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    positions = np.arange(len(x_labels))

    ax.bar(
        positions,
        means,
        yerr=sems,
        capsize=4,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)

    title = f"{EXPERIMENT_LABELS[experiment]} - {metric_label}"
    ax.set_title(title)

    if unit:
        ax.set_ylabel(f"{metric_label} ({unit})")
    else:
        ax.set_ylabel(metric_label)

    ax.set_xlabel("Group")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_metric(dataframe, experiment, metric, group_columns):
    output_dir = PROJECT_ROOT / "outputs" / experiment / "figures" / "summary"

    output_path = (
        output_dir
        / f"{safe_filename(experiment)}_{safe_filename(metric)}_summary.png"
    )

    if metric not in dataframe.columns:
        return {
            "experiment": experiment,
            "metric": metric,
            "status": "skipped_missing_column",
            "rows_total": int(len(dataframe)),
            "rows_used": 0,
            "missing_count": None,
            "output_path": "",
        }

    numeric_values = pd.to_numeric(dataframe[metric], errors="coerce")
    missing_count = int(numeric_values.isna().sum())
    rows_used = int(numeric_values.notna().sum())

    if rows_used == 0:
        return {
            "experiment": experiment,
            "metric": metric,
            "status": "skipped_no_numeric_data",
            "rows_total": int(len(dataframe)),
            "rows_used": 0,
            "missing_count": missing_count,
            "output_path": "",
        }

    summary = summarize_metric(
        dataframe=dataframe,
        metric=metric,
        group_columns=group_columns,
    )

    save_bar_plot(
        summary=summary,
        experiment=experiment,
        metric=metric,
        output_path=output_path,
    )

    return {
        "experiment": experiment,
        "metric": metric,
        "status": "created",
        "rows_total": int(len(dataframe)),
        "rows_used": rows_used,
        "missing_count": missing_count,
        "output_path": str(output_path),
    }


def run_csv_summary_plots():
    manifest_rows = []

    for experiment, path in SPSS_MAIN_FILES.items():
        if not path.exists():
            manifest_rows.append(
                {
                    "experiment": experiment,
                    "metric": "",
                    "status": "skipped_missing_spss_main_file",
                    "rows_total": 0,
                    "rows_used": 0,
                    "missing_count": None,
                    "output_path": "",
                }
            )
            continue

        dataframe = read_csv(path)
        config = PLOT_CONFIG[experiment]

        group_columns = config["group_columns"]
        metrics = config["metrics"]

        missing_group_columns = [
            column for column in group_columns if column not in dataframe.columns
        ]

        if missing_group_columns:
            raise ValueError(
                f"{experiment}: missing group columns: {missing_group_columns}"
            )

        for metric in metrics:
            manifest_rows.append(
                plot_metric(
                    dataframe=dataframe,
                    experiment=experiment,
                    metric=metric,
                    group_columns=group_columns,
                )
            )

    manifest = pd.DataFrame(manifest_rows)

    manifest_path = PROJECT_ROOT / "outputs" / "qc" / "csv_summary_figure_manifest.csv"

    write_csv(manifest, manifest_path)

    return manifest_path, manifest


def main():
    manifest_path, manifest = run_csv_summary_plots()

    print("CSV tabanlı özet grafikler oluşturuldu.")
    print(f"Manifest dosyası: {manifest_path}")
    print("")
    print("Durum özeti:")
    print(manifest["status"].value_counts(dropna=False))
    print("")
    print("Deneylere göre oluşturulan grafik sayısı:")
    created = manifest[manifest["status"] == "created"]
    if created.empty:
        print("Oluşturulan grafik yok.")
    else:
        print(created.groupby("experiment")["metric"].count())
    print("")
    print("Oluşturulamayan grafikler:")
    failed = manifest[manifest["status"] != "created"]
    if failed.empty:
        print("Yok.")
    else:
        print(failed.to_string(index=False))


if __name__ == "__main__":
    main()