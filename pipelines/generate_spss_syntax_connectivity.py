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


RESULT_FILES = {
    "flanker": PROJECT_ROOT / "outputs" / "flanker" / "csv" / "flanker_results.csv",
    "gonogo": PROJECT_ROOT / "outputs" / "gonogo" / "csv" / "gonogo_results.csv",
    "readysetgo": PROJECT_ROOT / "outputs" / "readysetgo" / "csv" / "readysetgo_results.csv",
}

CONNECTIVITY_METHODS = ("coh", "wpli", "ciplv")
CONNECTIVITY_BANDS = ("theta", "alpha", "beta")
CONNECTIVITY_PAIRS = (
    "interhemispheric_F3_F4",
    "intrahemispheric_left_F3_P7",
    "intrahemispheric_right_F4_P8",
)


def connectivity_variables(method):
    return [
        f"{method}_{band}_{pair}"
        for band in CONNECTIVITY_BANDS
        for pair in CONNECTIVITY_PAIRS
    ]


def augment_statistics_ready_with_connectivity(experiment, dataframe):
    """Create an SPSS input copy containing already-computed connectivity columns."""
    if experiment not in RESULT_FILES:
        return dataframe, STATISTICS_READY_FILES[experiment]

    result_path = RESULT_FILES[experiment]
    if not result_path.exists():
        raise FileNotFoundError(f"Connectivity result file not found: {result_path}")

    result_df = read_csv(result_path)
    connectivity_columns = [
        column
        for column in result_df.columns
        if column.startswith(("coh_", "wpli_", "ciplv_"))
    ]

    expected = [
        variable
        for method in CONNECTIVITY_METHODS
        for variable in connectivity_variables(method)
    ]
    missing_expected = [c for c in expected if c not in connectivity_columns]
    if missing_expected:
        raise ValueError(
            f"{experiment}: expected connectivity columns missing from result CSV: "
            f"{missing_expected}"
        )

    merge_keys = [
        key for key in ("subject_id", "group", "experiment")
        if key in dataframe.columns and key in result_df.columns
    ]
    if "subject_id" not in merge_keys:
        raise ValueError(f"{experiment}: subject_id is required to merge connectivity columns")

    columns_to_add = [c for c in expected if c not in dataframe.columns]
    augmented = dataframe.copy()
    if columns_to_add:
        source = result_df[merge_keys + columns_to_add].copy()
        if source.duplicated(merge_keys).any():
            raise ValueError(f"{experiment}: duplicate merge keys in result CSV")
        augmented = augmented.merge(source, on=merge_keys, how="left", validate="one_to_one")

    output_path = (
        PROJECT_ROOT / "outputs" / "statistics" /
        f"{experiment}_statistics_ready_with_connectivity.csv"
    )
    write_csv(augmented, output_path)
    return augmented, output_path


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
        {
            "title": "Flanker left frontocentral RP/PMP sensitivity",
            "include_flag": "include_rp_pmp",
            "variables": [
            "left_frontocentral_rp_mean_uv",
            "left_frontocentral_pmp_peak_uv",
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
        {
            "title": "Go/No-Go left frontocentral RP/PMP sensitivity",
            "include_flag": "include_rp_pmp",
            "variables": [
                "left_frontocentral_rp_mean_uv",
                "left_frontocentral_pmp_peak_uv",
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
        {
            "title": "ReadySetGo left frontocentral CNV sensitivity",
            "include_flag": "include_cnv",
            "variables": [
                "left_frontocentral_cnv_amplitude_uv",
            ],
        },
        {
            "title": "ReadySetGo left frontocentral RP/PMP sensitivity",
            "include_flag": "include_rp_pmp",
            "variables": [
                "left_frontocentral_rp_mean_uv",
                "left_frontocentral_pmp_peak_uv",
            ],
        },
    ],
}


for _experiment in ("flanker", "gonogo", "readysetgo"):
    for _method in CONNECTIVITY_METHODS:
        BETWEEN_SUBJECT_ANALYSES[_experiment].append(
            {
                "title": f"{_experiment} {_method.upper()} connectivity",
                "variables": connectivity_variables(_method),
                "include_condition": "1 = 1",
            }
        )


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
    {
        "title": "TMT left frontocentral RP/PMP sensitivity",
        "include_flags": [
            "include_rp_pmp",
            "include_tmt_complete_pair",
        ],
        "variables": [
            "left_frontocentral_rp_mean_uv",
            "left_frontocentral_pmp_peak_uv",
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
        include_flag = analysis.get("include_flag")

        variables_to_validate = list(variables)
        if include_flag is not None:
            variables_to_validate = [include_flag] + variables_to_validate

        validate_variables(
            experiment=experiment,
            dataframe=dataframe,
            variables=variables_to_validate,
        )

        include_condition = analysis.get(
            "include_condition",
            f"{include_flag} = 1" if include_flag is not None else "1 = 1",
        )

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
            dataframe, analysis_csv_path = augment_statistics_ready_with_connectivity(
                experiment=experiment,
                dataframe=dataframe,
            )
            syntax_text = make_between_subject_syntax(
                experiment=experiment,
                dataframe=dataframe,
                csv_path=analysis_csv_path,
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