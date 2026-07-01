"""
Full NeuroCogEEG analysis runner.

This script runs the entire validated NeuroCogEEG workflow in the correct order:

1. Syntax check
2. Main EEG analysis pipelines
3. QC reports
4. Variable inventory and dictionary
5. SPSS main export and validation
6. Summary figures
7. Waveform and topomap figures
8. Figure validation

It does not implement analysis logic itself. It orchestrates existing scripts.

Outputs:
- outputs/qc/full_analysis_run_<timestamp>.log
- outputs/qc/full_analysis_summary_<timestamp>.csv
- outputs/qc/full_analysis_summary_latest.csv

Usage:
    python pipelines/run_full_analysis.py
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qc"


STEPS = [
    {
        "name": "syntax_check",
        "description": "Python syntax check for neurocogeeg and pipelines",
        "command": [
            sys.executable,
            "-m",
            "compileall",
            "neurocogeeg",
            "pipelines",
        ],
    },
    {
        "name": "flanker_pipeline",
        "description": "Run Flanker EEG analysis pipeline",
        "command": [sys.executable, "pipelines/flanker.py"],
    },
    {
        "name": "gonogo_pipeline",
        "description": "Run Go/No-Go EEG analysis pipeline",
        "command": [sys.executable, "pipelines/gonogo.py"],
    },
    {
        "name": "readysetgo_pipeline",
        "description": "Run ReadySetGo EEG analysis pipeline",
        "command": [sys.executable, "pipelines/readysetgo.py"],
    },
    {
        "name": "tmt_pipeline",
        "description": "Run TMT EEG analysis pipeline",
        "command": [sys.executable, "pipelines/tmt.py"],
    },
    {
        "name": "qc_report",
        "description": "Create general QC report",
        "command": [sys.executable, "pipelines/qc_report.py"],
    },
    {
        "name": "variable_inventory",
        "description": "Create variable inventory",
        "command": [sys.executable, "pipelines/variable_inventory.py"],
    },
    {
        "name": "variable_dictionary_draft",
        "description": "Create variable dictionary draft",
        "command": [sys.executable, "pipelines/variable_dictionary_draft.py"],
    },
    {
        "name": "spss_main_column_plan",
        "description": "Create SPSS main column-selection plan",
        "command": [sys.executable, "pipelines/spss_main_column_plan.py"],
    },
    {
        "name": "spss_main_export",
        "description": "Export SPSS main analysis CSV files",
        "command": [sys.executable, "pipelines/spss_main_export.py"],
    },
    {
        "name": "validate_spss_main",
        "description": "Validate SPSS main analysis CSV files",
        "command": [sys.executable, "pipelines/validate_spss_main.py"],
    },
    {
    "name": "prepare_statistics_data",
    "description": "Prepare statistics-ready CSV files and validate participant IDs",
    "command": [sys.executable, "pipelines/prepare_statistics_data.py"],
    },
    {
    "name": "statistics_descriptives",
    "description": "Create descriptive statistics tables from statistics-ready files",
    "command": [sys.executable, "pipelines/statistics_descriptives.py"],
    },
    {
        "name": "plot_csv_summaries",
        "description": "Create CSV-based summary figures",
        "command": [sys.executable, "pipelines/plot_csv_summaries.py"],
    },
    {
        "name": "plot_flanker_waveforms",
        "description": "Create Flanker waveform figures",
        "command": [sys.executable, "pipelines/plot_flanker_waveforms.py"],
    },
    {
        "name": "plot_gonogo_waveforms",
        "description": "Create Go/No-Go waveform figures",
        "command": [sys.executable, "pipelines/plot_gonogo_waveforms.py"],
    },
    {
        "name": "plot_readysetgo_waveforms",
        "description": "Create ReadySetGo waveform figures",
        "command": [sys.executable, "pipelines/plot_readysetgo_waveforms.py"],
    },
    {
        "name": "plot_tmt_waveforms",
        "description": "Create TMT waveform figures",
        "command": [sys.executable, "pipelines/plot_tmt_waveforms.py"],
    },
    {
        "name": "plot_tmt_topomaps",
        "description": "Create TMT topomap figures",
        "command": [sys.executable, "pipelines/plot_tmt_topomaps.py"],
    },
    {
        "name": "validate_figures",
        "description": "Validate all generated figures",
        "command": [sys.executable, "pipelines/validate_figures.py"],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full NeuroCogEEG analysis workflow."
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue running later steps even if one step fails. "
            "Default behavior is to stop at the first error."
        ),
    )

    parser.add_argument(
        "--start-at",
        default=None,
        help=(
            "Optional step name to start from. "
            "Example: --start-at qc_report"
        ),
    )

    parser.add_argument(
        "--stop-after",
        default=None,
        help=(
            "Optional step name to stop after. "
            "Example: --stop-after validate_spss_main"
        ),
    )

    return parser.parse_args()


def get_selected_steps(start_at, stop_after):
    step_names = [step["name"] for step in STEPS]

    if start_at is not None and start_at not in step_names:
        valid = ", ".join(step_names)
        raise ValueError(
            f"Unknown --start-at step: {start_at}. Valid steps are: {valid}"
        )

    if stop_after is not None and stop_after not in step_names:
        valid = ", ".join(step_names)
        raise ValueError(
            f"Unknown --stop-after step: {stop_after}. Valid steps are: {valid}"
        )

    start_index = 0
    stop_index = len(STEPS) - 1

    if start_at is not None:
        start_index = step_names.index(start_at)

    if stop_after is not None:
        stop_index = step_names.index(stop_after)

    if start_index > stop_index:
        raise ValueError(
            "--start-at cannot come after --stop-after in the workflow order."
        )

    return STEPS[start_index : stop_index + 1]


def make_run_paths():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    QC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = QC_OUTPUT_DIR / f"full_analysis_run_{timestamp}.log"
    summary_path = QC_OUTPUT_DIR / f"full_analysis_summary_{timestamp}.csv"
    latest_summary_path = QC_OUTPUT_DIR / "full_analysis_summary_latest.csv"

    return log_path, summary_path, latest_summary_path


def write_log_line(log_file, text):
    print(text)
    log_file.write(text + "\n")
    log_file.flush()


def format_command(command):
    return " ".join(str(part) for part in command)


def run_step(step, log_file):
    start_time = time.time()

    write_log_line(log_file, "")
    write_log_line(log_file, "=" * 88)
    write_log_line(log_file, f"STEP: {step['name']}")
    write_log_line(log_file, f"DESCRIPTION: {step['description']}")
    write_log_line(log_file, f"COMMAND: {format_command(step['command'])}")
    write_log_line(log_file, "=" * 88)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        step["command"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    assert process.stdout is not None

    for line in process.stdout:
        write_log_line(log_file, line.rstrip())

    return_code = process.wait()
    elapsed_seconds = time.time() - start_time

    if return_code == 0:
        status = "PASS"
    else:
        status = "FAIL"

    write_log_line(log_file, "")
    write_log_line(
        log_file,
        (
            f"STEP RESULT: {step['name']} | "
            f"{status} | "
            f"return_code={return_code} | "
            f"elapsed_seconds={elapsed_seconds:.2f}"
        ),
    )

    return {
        "step": step["name"],
        "description": step["description"],
        "command": format_command(step["command"]),
        "status": status,
        "return_code": int(return_code),
        "elapsed_seconds": float(elapsed_seconds),
    }


def write_summary(summary_rows, summary_path, latest_summary_path):
    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        summary_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )

    summary.to_csv(
        latest_summary_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )

    return summary


def print_final_summary(log_file, summary, log_path, summary_path):
    failed = summary[summary["status"] != "PASS"]

    write_log_line(log_file, "")
    write_log_line(log_file, "=" * 88)
    write_log_line(log_file, "FULL ANALYSIS SUMMARY")
    write_log_line(log_file, "=" * 88)
    write_log_line(log_file, summary[["step", "status", "elapsed_seconds"]].to_string(index=False))
    write_log_line(log_file, "")
    write_log_line(log_file, f"Log file: {log_path}")
    write_log_line(log_file, f"Summary file: {summary_path}")

    if failed.empty:
        write_log_line(log_file, "")
        write_log_line(log_file, "FINAL RESULT: PASS")
    else:
        write_log_line(log_file, "")
        write_log_line(log_file, "FINAL RESULT: FAIL")
        write_log_line(log_file, "")
        write_log_line(log_file, "Failed steps:")
        write_log_line(log_file, failed[["step", "return_code"]].to_string(index=False))


def main():
    args = parse_args()

    selected_steps = get_selected_steps(
        start_at=args.start_at,
        stop_after=args.stop_after,
    )

    log_path, summary_path, latest_summary_path = make_run_paths()

    summary_rows = []

    with log_path.open("w", encoding="utf-8") as log_file:
        write_log_line(log_file, "NeuroCogEEG full analysis runner")
        write_log_line(log_file, f"Project root: {PROJECT_ROOT}")
        write_log_line(log_file, f"Started at: {datetime.now().isoformat(timespec='seconds')}")
        write_log_line(log_file, f"Continue on error: {args.continue_on_error}")
        write_log_line(log_file, "")
        write_log_line(log_file, "Selected steps:")
        for step in selected_steps:
            write_log_line(log_file, f"- {step['name']}")

        for step in selected_steps:
            result = run_step(
                step=step,
                log_file=log_file,
            )

            summary_rows.append(result)

            if result["status"] != "PASS" and not args.continue_on_error:
                write_log_line(log_file, "")
                write_log_line(
                    log_file,
                    (
                        "Stopping because a step failed. "
                        "Use --continue-on-error only if you explicitly want "
                        "later steps to run after a failure."
                    ),
                )
                break

        summary = write_summary(
            summary_rows=summary_rows,
            summary_path=summary_path,
            latest_summary_path=latest_summary_path,
        )

        print_final_summary(
            log_file=log_file,
            summary=summary,
            log_path=log_path,
            summary_path=summary_path,
        )

    failed = summary[summary["status"] != "PASS"]

    if failed.empty:
        raise SystemExit(0)

    raise SystemExit(1)


if __name__ == "__main__":
    main()
