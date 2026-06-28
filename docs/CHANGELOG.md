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