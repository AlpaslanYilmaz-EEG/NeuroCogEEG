# NeuroCogEEG Changelog

## Initial scaffold

- Created project package structure.
- Added core module placeholders.
- Added pipeline placeholders.
- Added YAML configuration placeholders.
- Added permanent `data/` and `outputs/` directory trees with `.gitkeep` files.
- Added project documentation files.

## Field Test Checkpoint

All four experiment pipelines were tested with real EDF files.

Completed:
- Flanker real EDF pipeline test
- Go/No-Go real EDF pipeline test
- ReadySetGo real EDF pipeline test
- TMT real EDF pipeline test
- General QC report generation

Current status:
- Processing error count is zero for all experiments.
- Flanker has missing ERN values for participants without error responses.
- ReadySetGo has one participant failing set-locked minimum trial criteria.
- TMT has 18 rows failing response-locked RP/PMP minimum trial criteria.
- TMT PSD outputs are complete for all rows.
```markdown
---

## Field Test and SPSS Export Checkpoint

All four experiment pipelines were tested with real EDF files.

Completed real-data pipeline tests:

- Flanker
- Go/No-Go
- ReadySetGo
- TMT

Current real-data status:

- All four pipelines completed without processing errors.
- Flanker produced 46 rows.
- Go/No-Go produced 46 rows.
- ReadySetGo produced 46 rows.
- TMT produced 92 rows, with 46 rows for TMT1 and 46 rows for TMT2.
- TMT variants are represented using the `tmt_variant` column.

Known and expected missing-value situations:

- Flanker ERN can be missing when a participant has no usable error-response epochs.
- ReadySetGo CNV and set-locked connectivity can be missing when set-locked minimum trial criteria are not met.
- TMT RP/PMP values can be missing when response-locked minimum trial criteria are not met.
- Missing `qc_ica_excluded_components` values usually indicate that no ICA component was excluded.

QC and export tools added:

- `pipelines/qc_report.py`
- `pipelines/variable_inventory.py`
- `pipelines/variable_dictionary_draft.py`
- `pipelines/spss_main_column_plan.py`
- `pipelines/spss_main_export.py`
- `pipelines/validate_spss_main.py`

Generated SPSS main analysis files:

- `outputs/flanker/csv/flanker_spss_main.csv`
- `outputs/gonogo/csv/gonogo_spss_main.csv`
- `outputs/readysetgo/csv/readysetgo_spss_main.csv`
- `outputs/tmt/csv/tmt_spss_main.csv`

SPSS main validation result:

- Flanker: 46 rows, 54 columns
- Go/No-Go: 46 rows, 53 columns
- ReadySetGo: 46 rows, 52 columns
- TMT: 92 rows, 50 columns
- Validation error count: 0
- Result: PASS

Current project status:

The pipeline can now run from real EDF files to cleaned SPSS-ready main analysis CSV files with automated QC, variable inventory, variable dictionary draft, column selection plan, export, and validation checks.
```
---

## Figure Generation Checkpoint

Figure generation pipelines were added and validated.

Completed figure pipelines:

- `pipelines/plot_csv_summaries.py`
- `pipelines/plot_flanker_waveforms.py`
- `pipelines/plot_gonogo_waveforms.py`
- `pipelines/plot_readysetgo_waveforms.py`
- `pipelines/plot_tmt_waveforms.py`
- `pipelines/plot_tmt_topomaps.py`
- `pipelines/validate_figures.py`

Generated figure classes:

- CSV-based summary figures
- Flanker stimulus-locked and response-locked waveform figures
- Go/No-Go stimulus-locked and response-locked waveform figures
- ReadySetGo set-locked CNV and response-locked waveform figures
- TMT response-locked waveform figures
- TMT descriptive scalp topomap figures

Expected figure counts:

- CSV summary figures: 31
- Flanker waveform figures: 4
- Go/No-Go waveform figures: 4
- ReadySetGo waveform figures: 3
- TMT waveform figures: 2
- TMT topomap figures: 12
- Total expected figures: 56

Validation status:

- `pipelines/validate_figures.py` passed.
- Figure validation error count: 0.
- Generated figure files are treated as reproducible outputs and are ignored by Git.

---

## Full Analysis Runner Checkpoint

A full analysis orchestration pipeline was added.

Added script:

- `pipelines/run_full_analysis.py`

Purpose:

The script runs the complete NeuroCogEEG workflow in the validated order:

1. Syntax check
2. Main EEG analysis pipelines
3. QC report generation
4. Variable inventory generation
5. Variable dictionary draft generation
6. SPSS main column planning
7. SPSS main export
8. SPSS main validation
9. CSV-based summary figures
10. Waveform figures
11. TMT topomap figures
12. Figure validation

Validation status:

- Partial workflow test passed.
- Full workflow test passed.
- Final result: PASS.

Current status:

The project can now be reproduced from raw EDF files to validated SPSS-ready outputs and validated figure outputs using a single command:

```bash
python pipelines/run_full_analysis.py
