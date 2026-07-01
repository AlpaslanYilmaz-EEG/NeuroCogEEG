"""
SPSS syntax generator for NeuroCogEEG.

This script generates SPSS .sps syntax files from statistics-ready CSV files.

It does not run statistical tests in Python.

Generated syntax files:
- outputs/statistics/spss_syntax/flanker_analysis.sps
- outputs/statistics/spss_syntax/gonogo_analysis.sps
- outputs/statistics/spss_syntax/readysetgo_analysis.sps
- outputs/statistics/spss_syntax/tmt_analysis.sps
- outputs/statistics/spss_syntax/run_all_analyses.sps
- outputs/qc/spss_syntax_manifest.csv
"""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STATISTICS_READY_FILES = {
    "flanker": PROJECT_ROOT / "outputs" / "statistics" / "flanker_statistics_ready.csv",
    "gonogo": PROJECT_ROOT / "outputs" / "statistics" / "gonogo_statistics_ready.csv",
    "readysetgo": PROJECT_ROOT / "outputs" / "statistics" / "readysetgo_statistics_ready.csv",
    "tmt": PROJECT_ROOT / "outputs" / "statistics" / "tmt_statistics_ready.csv",
}


SYNTAX_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "statistics" / "spss_syntax"
QC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qc"


STRING_COLUMNS = {
    "subject_id",
    "participant_id",
    "group",
    "experiment",
    "tmt_variant",
}


BETWEEN_SUBJECT_ANALYSES = {
    "flanker": [
        {
            "title": "Flanker behavior",
            "include_flag": "include_behavior",
            "variables": [
                "correct_rt_mean_ms",
                "accuracy_percent",
            ],
        },
        {
            "title": "Flanker N2/P3",
            "include_flag": "include_n2_p3",
            "variables": [
                "n2_amplitude_uv",
                "p3_amplitude_uv",
            ],
        },
        {
            "title": "Flanker ERN",
            "include_flag": "include_ern",
            "variables": [
                "ern_amplitude_uv",
            ],
        },
        {
            "title": "Flanker response-locked RP/PMP",
            "include_flag": "include_rp_pmp",
            "variables": [
                "rp_mean_uv",
                "pmp_peak_uv",
            ],
        },
    ],
    "gonogo": [
        {
            "title": "Go/No-Go behavior",
            "include_flag": "include_behavior",
            "variables": [
                "correct_rt_mean_ms",
                "accuracy_percent",
            ],
        },
        {
            "title": "Go/No-Go N2/P3",
            "include_flag": "include_n2_p3",
            "variables": [
                "n2_amplitude_uv",
                "p3_amplitude_uv",
            ],
        },
        {
            "title": "Go/No-Go response-locked RP/PMP",
            "include_flag": "include_rp_pmp",
            "variables": [
                "rp_mean_uv",
                "pmp_peak_uv",
            ],
        },
    ],
    "readysetgo": [
        {
            "title": "ReadySetGo behavior",
            "include_flag": "include_behavior",
            "variables": [
                "response_time_mean_ms",
                "accuracy_percent",
            ],
        },
        {
            "title": "ReadySetGo CNV",
            "include_flag": "include_cnv",
            "variables": [
                "cnv_amplitude_uv",
            ],
        },
        {
            "title": "ReadySetGo response-locked RP/PMP",
            "include_flag": "include_rp_pmp",
            "variables": [
                "rp_mean_uv",
                "pmp_peak_uv",
            ],
        },
    ],
}


TMT_MIXED_ANALYSES = [
    {
        "title": "TMT behavior",
        "include_flags": [
            "include_behavior",
            "include_tmt_complete_pair",
        ],
        "variables": [
            "task_duration_s",
            "error_percent",
        ],
    },
    {
        "title": "TMT ROI PSD",
        "include_flags": [
            "include_psd",
            "include_tmt_complete_pair",
        ],
        "variables": [
            "frontal_theta_relative_percent",
            "frontal_alpha_relative_percent",
            "frontal_beta_relative_percent",
            "parietal_occipital_theta_relative_percent",
            "parietal_occipital_alpha_relative_percent",
            "parietal_occipital_beta_relative_percent",
        ],
    },
    {
        "title": "TMT response-locked RP/PMP",
        "include_flags": [
            "include_rp_pmp",
            "include_tmt_complete_pair",
        ],
        "variables": [
            "rp_mean_uv",
            "pmp_peak_uv",
        ],
    },
]


def read_csv(path):
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(dataframe, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )


def spss_path(path):
    return str(path).replace("\\", "/")


def spss_comment(text):
    return f"* {text}."


def variable_format(column):
    if column in STRING_COLUMNS:
        return f"{column} A128"

    return f"{column} F12.6"


def make_variable_format_block(columns):
    lines = []

    for column in columns:
        lines.append(f"    {variable_format(column)}")

    return "\n".join(lines)


def make_get_data_syntax(csv_path, dataframe, dataset_name):
    variable_block = make_variable_format_block(dataframe.columns)

    return f"""
SET UNICODE=ON.

{spss_comment("If numeric decimals are imported incorrectly, check SPSS locale/decimal separator settings")}

GET DATA
  /TYPE=TXT
  /FILE='{spss_path(csv_path)}'
  /ENCODING='UTF8'
  /DELCASE=LINE
  /DELIMITERS=";"
  /QUALIFIER='"'
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /IMPORTCASE=ALL
  /VARIABLES=
{variable_block}
  .
CACHE.
EXECUTE.

DATASET NAME {dataset_name}.
DATASET ACTIVATE {dataset_name}.
"""


def make_group_code_syntax():
    return """
NUMERIC group_code (F1.0).
IF (group = 'control') group_code = 1.
IF (group = 'experimental') group_code = 2.
VALUE LABELS group_code
  1 'control'
  2 'experimental'.
VARIABLE LEVEL group_code (NOMINAL).
EXECUTE.
"""


def make_tmt_codes_syntax():
    return """
NUMERIC group_code (F1.0).
IF (group = 'control') group_code = 1.
IF (group = 'experimental') group_code = 2.
VALUE LABELS group_code
  1 'control'
  2 'experimental'.
VARIABLE LEVEL group_code (NOMINAL).

NUMERIC tmt_variant_code (F1.0).
IF (tmt_variant = 'tmt1') tmt_variant_code = 1.
IF (tmt_variant = 'tmt2') tmt_variant_code = 2.
VALUE LABELS tmt_variant_code
  1 'tmt1'
  2 'tmt2'.
VARIABLE LEVEL tmt_variant_code (NOMINAL).

AUTORECODE VARIABLES=participant_id
  /INTO participant_code
  /PRINT.
VARIABLE LEVEL participant_code (NOMINAL).

EXECUTE.
"""


def make_examine_block(title, include_condition, variables, by_variable):
    variable_text = " ".join(variables)

    return f"""
{spss_comment(title + " - descriptives and assumption checks")}

TEMPORARY.
SELECT IF ({include_condition}).
EXAMINE VARIABLES={variable_text} BY {by_variable}
  /PLOT BOXPLOT HISTOGRAM NPPLOT
  /COMPARE GROUPS
  /STATISTICS DESCRIPTIVES
  /CINTERVAL 95
  /MISSING LISTWISE
  /NOTOTAL.
"""


def make_ttest_block(title, include_condition, variables):
    variable_text = " ".join(variables)

    return f"""
{spss_comment(title + " - independent samples t-test")}

TEMPORARY.
SELECT IF ({include_condition}).
T-TEST GROUPS=group_code(1 2)
  /MISSING=ANALYSIS
  /VARIABLES={variable_text}
  /CRITERIA=CI(.95).
"""


def make_mixed_block(title, include_condition, variable):
    return f"""
{spss_comment(title + " - mixed model for " + variable)}

TEMPORARY.
SELECT IF ({include_condition}).
MIXED {variable} BY group_code tmt_variant_code
  /CRITERIA=CIN(95) MXITER(100) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE)
    LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=group_code tmt_variant_code group_code*tmt_variant_code | SSTYPE(3)
  /RANDOM=INTERCEPT | SUBJECT(participant_code) COVTYPE(VC)
  /EMMEANS=TABLES(group_code*tmt_variant_code)
  /PRINT=SOLUTION TESTCOV.
"""


def validate_variables(experiment, dataframe, variables):
    missing = [variable for variable in variables if variable not in dataframe.columns]

    if missing:
        raise ValueError(
            f"{experiment}: variables missing from statistics-ready file: {missing}"
        )


def make_between_subject_syntax(experiment, dataframe, csv_path):
    dataset_name = f"{experiment}_data"

    parts = [
        spss_comment(f"NeuroCogEEG {experiment} SPSS analysis syntax"),
        spss_comment("Generated automatically by pipelines/generate_spss_syntax.py"),
        make_get_data_syntax(
            csv_path=csv_path,
            dataframe=dataframe,
            dataset_name=dataset_name,
        ),
        make_group_code_syntax(),
    ]

    for analysis in BETWEEN_SUBJECT_ANALYSES[experiment]:
        variables = analysis["variables"]
        include_flag = analysis["include_flag"]

        validate_variables(
            experiment=experiment,
            dataframe=dataframe,
            variables=[include_flag] + variables,
        )

        include_condition = f"{include_flag} = 1"

        parts.append(
            make_examine_block(
                title=analysis["title"],
                include_condition=include_condition,
                variables=variables,
                by_variable="group_code",
            )
        )

        parts.append(
            make_ttest_block(
                title=analysis["title"],
                include_condition=include_condition,
                variables=variables,
            )
        )

    return "\n".join(parts)


def make_tmt_syntax(dataframe, csv_path):
    dataset_name = "tmt_data"

    parts = [
        spss_comment("NeuroCogEEG TMT SPSS mixed-model analysis syntax"),
        spss_comment("Generated automatically by pipelines/generate_spss_syntax.py"),
        spss_comment("TMT uses participant_id as repeated-participant identifier"),
        make_get_data_syntax(
            csv_path=csv_path,
            dataframe=dataframe,
            dataset_name=dataset_name,
        ),
        make_tmt_codes_syntax(),
    ]

    for analysis in TMT_MIXED_ANALYSES:
        variables = analysis["variables"]
        include_flags = analysis["include_flags"]

        validate_variables(
            experiment="tmt",
            dataframe=dataframe,
            variables=include_flags + variables,
        )

        include_condition = " AND ".join(
            f"{include_flag} = 1"
            for include_flag in include_flags
        )

        parts.append(
            make_examine_block(
                title=analysis["title"],
                include_condition=include_condition,
                variables=variables,
                by_variable="group_code BY tmt_variant_code",
            )
        )

        for variable in variables:
            parts.append(
                make_mixed_block(
                    title=analysis["title"],
                    include_condition=include_condition,
                    variable=variable,
                )
            )

    return "\n".join(parts)


def make_run_all_syntax(syntax_paths):
    lines = [
        spss_comment("Run all NeuroCogEEG SPSS analysis syntax files"),
        spss_comment("Open this file in SPSS and run it to execute all generated analysis syntax files"),
    ]

    for path in syntax_paths:
        lines.append(
            f"INSERT FILE='{spss_path(path)}'."
        )

    return "\n\n".join(lines) + "\n"


def generate_syntax_files():
    manifest_rows = []
    syntax_paths = []

    for experiment, csv_path in STATISTICS_READY_FILES.items():
        if not csv_path.exists():
            manifest_rows.append(
                {
                    "experiment": experiment,
                    "syntax_file": "",
                    "status": "skipped_missing_statistics_ready_file",
                    "details": str(csv_path),
                }
            )
            continue

        dataframe = read_csv(csv_path)

        if experiment in BETWEEN_SUBJECT_ANALYSES:
            syntax_text = make_between_subject_syntax(
                experiment=experiment,
                dataframe=dataframe,
                csv_path=csv_path,
            )
        elif experiment == "tmt":
            syntax_text = make_tmt_syntax(
                dataframe=dataframe,
                csv_path=csv_path,
            )
        else:
            raise ValueError(f"Unknown experiment: {experiment}")

        syntax_path = SYNTAX_OUTPUT_DIR / f"{experiment}_analysis.sps"

        write_text(
            path=syntax_path,
            text=syntax_text,
        )

        syntax_paths.append(syntax_path)

        manifest_rows.append(
            {
                "experiment": experiment,
                "syntax_file": str(syntax_path),
                "status": "created",
                "details": "",
            }
        )

    run_all_path = SYNTAX_OUTPUT_DIR / "run_all_analyses.sps"

    write_text(
        path=run_all_path,
        text=make_run_all_syntax(syntax_paths),
    )

    manifest_rows.append(
        {
            "experiment": "all",
            "syntax_file": str(run_all_path),
            "status": "created",
            "details": "master syntax file",
        }
    )

    manifest = pd.DataFrame(manifest_rows)

    manifest_path = QC_OUTPUT_DIR / "spss_syntax_manifest.csv"

    write_csv(
        dataframe=manifest,
        path=manifest_path,
    )

    return manifest_path, manifest


def main():
    manifest_path, manifest = generate_syntax_files()

    print("SPSS syntax dosyaları oluşturuldu.")
    print(f"Manifest dosyası: {manifest_path}")
    print("")
    print("Durum özeti:")
    print(manifest["status"].value_counts(dropna=False))
    print("")
    print(manifest.to_string(index=False))

    failed = manifest[manifest["status"] != "created"]

    if not failed.empty:
        raise SystemExit(1)

    print("")
    print("Sonuç: PASS")


if __name__ == "__main__":
    main()