"""
Figure validation script for NeuroCogEEG.

This script validates all generated figure manifests and checks whether the
expected PNG files exist and are non-empty.

It does not create figures.
It does not modify analysis results.

Outputs:
outputs/qc/figure_validation_report.csv
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MANIFESTS = {
    "csv_summary": {
        "path": PROJECT_ROOT / "outputs" / "qc" / "csv_summary_figure_manifest.csv",
        "expected_created": 31,
    },
    "flanker_waveform": {
        "path": PROJECT_ROOT / "outputs" / "qc" / "flanker_waveform_manifest.csv",
        "expected_created": 4,
    },
    "gonogo_waveform": {
        "path": PROJECT_ROOT / "outputs" / "qc" / "gonogo_waveform_manifest.csv",
        "expected_created": 4,
    },
    "readysetgo_waveform": {
        "path": PROJECT_ROOT / "outputs" / "qc" / "readysetgo_waveform_manifest.csv",
        "expected_created": 3,
    },
    "tmt_waveform": {
        "path": PROJECT_ROOT / "outputs" / "qc" / "tmt_waveform_manifest.csv",
        "expected_created": 2,
    },
    "tmt_topomap": {
        "path": PROJECT_ROOT / "outputs" / "qc" / "tmt_topomap_manifest.csv",
        "expected_created": 12,
    },
}


def read_manifest(path):
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )


def validate_manifest(manifest_name, manifest_info):
    rows = []

    manifest_path = manifest_info["path"]
    expected_created = manifest_info["expected_created"]

    if not manifest_path.exists():
        rows.append(
            {
                "manifest": manifest_name,
                "severity": "ERROR",
                "issue": "manifest_missing",
                "expected_created": expected_created,
                "observed_created": 0,
                "output_path": str(manifest_path),
                "details": "",
            }
        )
        return rows

    manifest = read_manifest(manifest_path)

    if "status" not in manifest.columns:
        rows.append(
            {
                "manifest": manifest_name,
                "severity": "ERROR",
                "issue": "status_column_missing",
                "expected_created": expected_created,
                "observed_created": 0,
                "output_path": str(manifest_path),
                "details": "",
            }
        )
        return rows

    if "output_path" not in manifest.columns:
        rows.append(
            {
                "manifest": manifest_name,
                "severity": "ERROR",
                "issue": "output_path_column_missing",
                "expected_created": expected_created,
                "observed_created": 0,
                "output_path": str(manifest_path),
                "details": "",
            }
        )
        return rows

    created = manifest[manifest["status"] == "created"].copy()
    observed_created = int(len(created))

    if observed_created != expected_created:
        rows.append(
            {
                "manifest": manifest_name,
                "severity": "ERROR",
                "issue": "created_count_mismatch",
                "expected_created": expected_created,
                "observed_created": observed_created,
                "output_path": str(manifest_path),
                "details": "",
            }
        )

    for _, row in created.iterrows():
        output_path_text = str(row["output_path"])

        if not output_path_text.strip():
            rows.append(
                {
                    "manifest": manifest_name,
                    "severity": "ERROR",
                    "issue": "created_row_has_empty_output_path",
                    "expected_created": expected_created,
                    "observed_created": observed_created,
                    "output_path": "",
                    "details": "",
                }
            )
            continue

        output_path = Path(output_path_text)

        if not output_path.exists():
            rows.append(
                {
                    "manifest": manifest_name,
                    "severity": "ERROR",
                    "issue": "figure_file_missing",
                    "expected_created": expected_created,
                    "observed_created": observed_created,
                    "output_path": str(output_path),
                    "details": "",
                }
            )
            continue

        if output_path.stat().st_size == 0:
            rows.append(
                {
                    "manifest": manifest_name,
                    "severity": "ERROR",
                    "issue": "figure_file_empty",
                    "expected_created": expected_created,
                    "observed_created": observed_created,
                    "output_path": str(output_path),
                    "details": "",
                }
            )

    rows.append(
        {
            "manifest": manifest_name,
            "severity": "INFO",
            "issue": "manifest_summary",
            "expected_created": expected_created,
            "observed_created": observed_created,
            "output_path": str(manifest_path),
            "details": f"rows={len(manifest)}",
        }
    )

    return rows


def main():
    validation_rows = []

    for manifest_name, manifest_info in MANIFESTS.items():
        validation_rows.extend(
            validate_manifest(
                manifest_name=manifest_name,
                manifest_info=manifest_info,
            )
        )

    validation = pd.DataFrame(validation_rows)

    output_path = PROJECT_ROOT / "outputs" / "qc" / "figure_validation_report.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    validation.to_csv(
        output_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )

    errors = validation[validation["severity"] == "ERROR"]
    info = validation[validation["severity"] == "INFO"]

    print("NeuroCogEEG grafik doğrulama raporu")
    print("")
    print("Manifest özetleri:")
    print(
        info[
            [
                "manifest",
                "expected_created",
                "observed_created",
                "details",
            ]
        ].to_string(index=False)
    )
    print("")
    print(f"ERROR sayısı: {len(errors)}")

    if not errors.empty:
        print("")
        print(
            errors[
                [
                    "manifest",
                    "issue",
                    "output_path",
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