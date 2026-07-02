"""
Post-SPSS result extraction runner for NeuroCogEEG.

This script runs the post-SPSS workflow after SPSS output files have been
exported to Excel.

It does not run SPSS.

Required input files:
- outputs/statistics/spss_output/flanker_analysis_output.xlsx
- outputs/statistics/spss_output/gonogo_analysis_output.xlsx
- outputs/statistics/spss_output/readysetgo_analysis_output.xlsx
- outputs/statistics/spss_output/tmt_analysis_output.xlsx

Workflow:
1. Validate exported SPSS Excel files.
2. Run extract_spss_results.py.
3. Run generate_report_tables.py.
4. Validate expected report table files.

Outputs:
- outputs/qc/post_spss_results_summary_latest.csv
- outputs/qc/post_spss_results_run_<timestamp>.log
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qc"


REQUIRED_SPSS_EXCEL_FILES = {
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


EXPECTED_OUTPUT_FILES = {
    "group_statistics": PROJECT_ROOT
    / "outputs"
    / "statistics"
    / "spss_extracted"
    / "group_statistics.csv",
    "independent_samples_tests_reporting": PROJECT_ROOT
    / "outputs"
    / "statistics"
    / "spss_extracted"
    / "independent_samples_tests_reporting.csv",
    "tmt_mixed_fixed_effects_reporting": PROJECT_ROOT
    / "outputs"
    / "statistics"
    / "spss_extracted"
    / "tmt_mixed_fixed_effects_reporting.csv",
    "tmt_estimated_marginal_means": PROJECT_ROOT
    / "outputs"
    / "statistics"
    / "spss_extracted"
    / "tmt_estimated_marginal_means.csv",
    "between_subject_results_table": PROJECT_ROOT
    / "outputs"
    / "statistics"
    / "report_tables"
    / "between_subject_results_table.csv",
    "tmt_mixed_fixed_effects_table": PROJECT_ROOT
    / "outputs"
    / "statistics"
    / "report_tables"
    / "tmt_mixed_fixed_effects_table.csv",
    "tmt_estimated_marginal_means_table": PROJECT_ROOT
    / "outputs"
    / "statistics"
    / "report_tables"
    / "tmt_estimated_marginal_means_table.csv",
}


STEPS = [
    {
        "name": "extract_spss_results",
        "description": "Extract SPSS result tables from exported Excel files",
        "command": [sys.executable, "pipelines/extract_spss_results.py"],
    },
    {
        "name": "generate_report_tables",
        "description": "Generate report-ready statistical result tables",
        "command": [sys.executable, "pipelines/generate_report_tables.py"],
    },
]


def make_run_paths():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    QC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = QC_OUTPUT_DIR / f"post_spss_results_run_{timestamp}.log"
    summary_path = QC_OUTPUT_DIR / f"post_spss_results_summary_{timestamp}.csv"
    latest_summary_path = QC_OUTPUT_DIR / "post_spss_results_summary_latest.csv"

    return log_path, summary_path, latest_summary_path


def write_log_line(log_file, text):
    print(text)
    log_file.write(text + "\n")
    log_file.flush()


def validate_input_excel_files():
    rows = []

    for experiment, path in REQUIRED_SPSS_EXCEL_FILES.items():
        if not path.exists():
            rows.append(
                {
                    "step": f"validate_input_{experiment}",
                    "description": "Validate required SPSS Excel input file",
                    "command": "",
                    "status": "FAIL",
                    "return_code": 1,
                    "elapsed_seconds": 0.0,
                    "details": f"missing: {path}",
                }
            )
            continue

        if path.stat().st_size == 0:
            rows.append(
                {
                    "step": f"validate_input_{experiment}",
                    "description": "Validate required SPSS Excel input file",
                    "command": "",
                    "status": "FAIL",
                    "return_code": 1,
                    "elapsed_seconds": 0.0,
                    "details": f"empty: {path}",
                }
            )
            continue

        rows.append(
            {
                "step": f"validate_input_{experiment}",
                "description": "Validate required SPSS Excel input file",
                "command": "",
                "status": "PASS",
                "return_code": 0,
                "elapsed_seconds": 0.0,
                "details": f"size_bytes={path.stat().st_size}",
            }
        )

    return rows


def validate_output_files():
    rows = []

    for file_key, path in EXPECTED_OUTPUT_FILES.items():
        if not path.exists():
            rows.append(
                {
                    "step": f"validate_output_{file_key}",
                    "description": "Validate expected post-SPSS output file",
                    "command": "",
                    "status": "FAIL",
                    "return_code": 1,
                    "elapsed_seconds": 0.0,
                    "details": f"missing: {path}",
                }
            )
            continue

        if path.stat().st_size == 0:
            rows.append(
                {
                    "step": f"validate_output_{file_key}",
                    "description": "Validate expected post-SPSS output file",
                    "command": "",
                    "status": "FAIL",
                    "return_code": 1,
                    "elapsed_seconds": 0.0,
                    "details": f"empty: {path}",
                }
            )
            continue

        rows.append(
            {
                "step": f"validate_output_{file_key}",
                "description": "Validate expected post-SPSS output file",
                "command": "",
                "status": "PASS",
                "return_code": 0,
                "elapsed_seconds": 0.0,
                "details": f"size_bytes={path.stat().st_size}",
            }
        )

    return rows


def run_step(step, log_file):
    start_time = time.time()

    write_log_line(log_file, "")
    write_log_line(log_file, "=" * 88)
    write_log_line(log_file, f"STEP: {step['name']}")
    write_log_line(log_file, f"DESCRIPTION: {step['description']}")
    write_log_line(log_file, f"COMMAND: {' '.join(step['command'])}")
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

    status = "PASS" if return_code == 0 else "FAIL"

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
        "command": " ".join(step["command"]),
        "status": status,
        "return_code": int(return_code),
        "elapsed_seconds": float(elapsed_seconds),
        "details": "",
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


def main():
    log_path, summary_path, latest_summary_path = make_run_paths()
    summary_rows = []

    with log_path.open("w", encoding="utf-8") as log_file:
        write_log_line(log_file, "NeuroCogEEG post-SPSS result runner")
        write_log_line(log_file, f"Project root: {PROJECT_ROOT}")
        write_log_line(log_file, f"Started at: {datetime.now().isoformat(timespec='seconds')}")

        input_rows = validate_input_excel_files()
        summary_rows.extend(input_rows)

        if any(row["status"] != "PASS" for row in input_rows):
            write_log_line(log_file, "")
            write_log_line(log_file, "Input validation failed. Stopping.")
            summary = write_summary(
                summary_rows,
                summary_path,
                latest_summary_path,
            )
            write_log_line(log_file, summary.to_string(index=False))
            raise SystemExit(1)

        for step in STEPS:
            result = run_step(step, log_file)
            summary_rows.append(result)

            if result["status"] != "PASS":
                write_log_line(log_file, "")
                write_log_line(log_file, "Stopping because a post-SPSS step failed.")
                summary = write_summary(
                    summary_rows,
                    summary_path,
                    latest_summary_path,
                )
                write_log_line(log_file, summary.to_string(index=False))
                raise SystemExit(1)

        output_rows = validate_output_files()
        summary_rows.extend(output_rows)

        summary = write_summary(
            summary_rows,
            summary_path,
            latest_summary_path,
        )

        failed = summary[summary["status"] != "PASS"]

        write_log_line(log_file, "")
        write_log_line(log_file, "=" * 88)
        write_log_line(log_file, "POST-SPSS RESULTS SUMMARY")
        write_log_line(log_file, "=" * 88)
        write_log_line(log_file, summary[["step", "status", "details"]].to_string(index=False))
        write_log_line(log_file, "")
        write_log_line(log_file, f"Log file: {log_path}")
        write_log_line(log_file, f"Summary file: {summary_path}")

        if failed.empty:
            write_log_line(log_file, "")
            write_log_line(log_file, "FINAL RESULT: PASS")
            raise SystemExit(0)

        write_log_line(log_file, "")
        write_log_line(log_file, "FINAL RESULT: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()