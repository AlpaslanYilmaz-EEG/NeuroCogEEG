"""
SPSS syntax validation script for NeuroCogEEG.

This script validates generated SPSS syntax files.

It does not run SPSS.
It does not execute statistical tests.

Checks:
- Expected .sps files exist.
- Syntax files are non-empty.
- Each syntax file contains expected SPSS command blocks.
- Master run_all_analyses.sps contains INSERT commands for all experiment files.

Outputs:
- outputs/qc/spss_syntax_validation_report.csv
"""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SYNTAX_DIR = PROJECT_ROOT / "outputs" / "statistics" / "spss_syntax"
QC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qc"


EXPECTED_SYNTAX_FILES = {
    "flanker": {
        "path": SYNTAX_DIR / "flanker_analysis.sps",
        "required_text": [
            "GET DATA",
            "DATASET NAME flanker_data",
            "T-TEST GROUPS=group_code(1 2)",
            "EXAMINE VARIABLES=",
        ],
        "forbidden_text": [
            "MIXED ",
        ],
    },
    "gonogo": {
        "path": SYNTAX_DIR / "gonogo_analysis.sps",
        "required_text": [
            "GET DATA",
            "DATASET NAME gonogo_data",
            "T-TEST GROUPS=group_code(1 2)",
            "EXAMINE VARIABLES=",
        ],
        "forbidden_text": [
            "MIXED ",
        ],
    },
    "readysetgo": {
        "path": SYNTAX_DIR / "readysetgo_analysis.sps",
        "required_text": [
            "GET DATA",
            "DATASET NAME readysetgo_data",
            "T-TEST GROUPS=group_code(1 2)",
            "EXAMINE VARIABLES=",
        ],
        "forbidden_text": [
            "MIXED ",
        ],
    },
    "tmt": {
        "path": SYNTAX_DIR / "tmt_analysis.sps",
        "required_text": [
            "GET DATA",
            "DATASET NAME tmt_data",
            "AUTORECODE VARIABLES=participant_id",
            "MIXED ",
            "group_code*tmt_variant_code",
            "SUBJECT(participant_code)",
            "EXAMINE VARIABLES=",
        ],
        "forbidden_text": [
            "T-TEST GROUPS=group_code(1 2)",
        ],
    },
    "run_all": {
        "path": SYNTAX_DIR / "run_all_analyses.sps",
        "required_text": [
            "INSERT FILE=",
            "flanker_analysis.sps",
            "gonogo_analysis.sps",
            "readysetgo_analysis.sps",
            "tmt_analysis.sps",
        ],
        "forbidden_text": [],
    },
}


def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def write_csv(dataframe, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )


def validate_one_syntax_file(name, spec):
    rows = []
    path = spec["path"]

    if not path.exists():
        rows.append(
            {
                "syntax_name": name,
                "severity": "ERROR",
                "issue": "syntax_file_missing",
                "syntax_file": str(path),
                "details": "",
            }
        )
        return rows

    size_bytes = path.stat().st_size

    if size_bytes == 0:
        rows.append(
            {
                "syntax_name": name,
                "severity": "ERROR",
                "issue": "syntax_file_empty",
                "syntax_file": str(path),
                "details": "",
            }
        )
        return rows

    text = read_text(path)

    for required in spec["required_text"]:
        if required not in text:
            rows.append(
                {
                    "syntax_name": name,
                    "severity": "ERROR",
                    "issue": "required_text_missing",
                    "syntax_file": str(path),
                    "details": required,
                }
            )

    for forbidden in spec["forbidden_text"]:
        if forbidden in text:
            rows.append(
                {
                    "syntax_name": name,
                    "severity": "ERROR",
                    "issue": "forbidden_text_found",
                    "syntax_file": str(path),
                    "details": forbidden,
                }
            )

    rows.append(
        {
            "syntax_name": name,
            "severity": "INFO",
            "issue": "syntax_file_summary",
            "syntax_file": str(path),
            "details": f"size_bytes={size_bytes}",
        }
    )

    return rows


def main():
    validation_rows = []

    for name, spec in EXPECTED_SYNTAX_FILES.items():
        validation_rows.extend(
            validate_one_syntax_file(
                name=name,
                spec=spec,
            )
        )

    validation = pd.DataFrame(validation_rows)

    output_path = QC_OUTPUT_DIR / "spss_syntax_validation_report.csv"
    write_csv(validation, output_path)

    errors = validation[validation["severity"] == "ERROR"]
    info = validation[validation["severity"] == "INFO"]

    print("SPSS syntax doğrulama raporu")
    print("")
    print("Dosya özetleri:")
    print(info[["syntax_name", "details"]].to_string(index=False))
    print("")
    print(f"ERROR sayısı: {len(errors)}")

    if not errors.empty:
        print("")
        print(
            errors[
                [
                    "syntax_name",
                    "issue",
                    "syntax_file",
                    "details",
                ]
            ].to_string(index=False)
        )
        raise SystemExit(1)

    print("")
    print("Sonuç: PASS")
    print(f"Rapor dosyası: {output_path}")


if __name__ == "__main__":
    main()