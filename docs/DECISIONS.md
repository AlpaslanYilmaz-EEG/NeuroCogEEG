# NeuroCogEEG Decision Log

---

## Decision 001

Experiment-specific parameters will never be hard-coded.

Reason:
Improves reproducibility and makes experiments configurable.

---

## Decision 002

All experiment parameters will be stored in YAML files.

Reason:
Separates scientific configuration from program logic.

---

## Decision 003

constants.py will contain only immutable project-wide constants.

Reason:
Keeps the source code independent from experimental protocols.

---

## Decision 004

ROI definitions are experiment-specific and therefore belong in YAML files.

Reason:
Different paradigms require different ROI selections.

---

## Decision 005

Event markers are experiment-specific and therefore belong in YAML files.

Reason:
Different acquisition systems and paradigms use different marker definitions.

---

## Decision 006

ERP component windows are experiment-specific and therefore belong in YAML files.

Reason:
Component latency depends on the experimental paradigm.

---

## Decision 007

The project will avoid unnecessary abstraction.

Reason:
The code should remain easy to understand and maintain.

---

## Decision 008

Enum classes will not be used unless a clear future need arises.

Reason:
The current project size does not justify the additional complexity.

---

## Decision 009

Data paths will not be stored in constants.py or YAML configuration files.

Reason:
File-system paths are computer-specific and are not scientific parameters.

---

## Decision 010

The official GitHub repository structure is fixed as the initial NeuroCogEEG project structure.

Reason:
A stable repository layout reduces confusion and keeps development consistent.

---

## Decision 011

YAML configuration files will live only in the repository-level `configs/` directory.

Reason:
Workspace folders should contain raw data, behavior files, and outputs only.
