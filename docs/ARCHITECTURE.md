# NeuroCogEEG Architecture

## Project Vision

NeuroCogEEG is a transparent, reproducible and modular EEG analysis framework
designed for executive function research using low-cost EEG systems.

The primary goal is scientific reliability rather than software complexity.

---

## Design Principles

1. Scientific correctness comes before programming convenience.
2. Reproducibility is a core requirement.
3. Experiment-specific parameters are never hard-coded.
4. Every function must have a single responsibility.
5. Readability is preferred over writing fewer lines of code.
6. Every methodological decision should be traceable to the scientific literature whenever possible.
7. The software should remain understandable years after it is written.

---

## Repository Data Policy

The repository contains source code, reusable configuration files,
documentation, tests, and the permanent project folder structure.

The `data/` and `outputs/` directory trees are part of the repository structure,
but real EEG data and generated analysis outputs must not be committed.

Empty folders are preserved with `.gitkeep` files.

---

## Configuration Philosophy

Only immutable, universally valid information belongs in `constants.py`.

Examples include:

- EEG frequency bands
- Emotiv EPOC X channel order
- Unit conversion constants
- Supported file extensions

All experiment-specific information must be stored in YAML configuration files.

Examples include:

- Event markers
- ERP time windows
- ROI definitions
- Filter settings
- Epoch durations
- Baseline intervals
- Connectivity parameters
- Statistical settings

---

## Data and Output Layout

Each experiment has a fixed relative folder structure under `data/`:

- `data/flanker/raw/control/`
- `data/flanker/raw/experimental/`
- `data/gonogo/raw/control/`
- `data/gonogo/raw/experimental/`
- `data/readysetgo/raw/control/`
- `data/readysetgo/raw/experimental/`
- `data/tmt/raw/control/`
- `data/tmt/raw/experimental/`

Each experiment has a corresponding output structure under `outputs/`:

- `csv/`
- `figures/`
- `reports/`
- `logs/`

---

## Coding Principles

- No magic numbers.
- Type hints for all public functions.
- Google-style docstrings.
- PEP8 compliant code.
- Modular architecture.
- Functions should be independently testable.

---

## Project Structure

The project is organized into independent modules.

Each module performs one specific task and can be maintained separately.

The pipelines (Flanker, Go/No-Go, Ready-Set-Go, TMT) only orchestrate the workflow by calling reusable modules.

No analysis logic should be duplicated between pipelines.

## Event Semantics

Numeric event codes do not have global meaning in NeuroCogEEG.

The same numeric marker value may represent different events in different
experiments. Therefore, event semantics must always be defined in the
experiment-specific YAML file.

Analysis code must never directly compare events to hard-coded numeric values
such as 48, 49 or 50. Event codes must always be accessed through the loaded
experiment configuration.
---

## Event Semantics

Numeric event codes do not have global meaning in NeuroCogEEG.

The same numeric marker value may represent different events in different
experiments. Therefore, event semantics must always be defined in the
experiment-specific YAML file.

Analysis code must never directly compare events to hard-coded numeric values
such as 48, 49 or 50.

Instead, event codes must always be accessed through the loaded experiment
configuration.

Correct usage:

```python
stimulus_code = config["events"]["stimulus"]
correct_code = config["events"]["correct_response"]

