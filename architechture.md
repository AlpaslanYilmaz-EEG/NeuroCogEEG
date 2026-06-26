# NeuroCogEEG Architecture

## Project Vision

NeuroCogEEG is a transparent, reproducible and modular EEG analysis framework
designed for executive function research using low-cost EEG systems.

The primary goal is scientific reliability rather than software complexity.

---

# Design Principles

1. Scientific correctness comes before programming convenience.

2. Reproducibility is a core requirement.

3. Experiment-specific parameters are never hard-coded.

4. Every function must have a single responsibility.

5. Readability is preferred over writing fewer lines of code.

6. Every methodological decision should be traceable to the scientific literature whenever possible.

7. The software should remain understandable years after it is written.

---

# Configuration Philosophy

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

# Coding Principles

- No magic numbers.
- Type hints for all public functions.
- Google-style docstrings.
- PEP8 compliant code.
- Modular architecture.
- Functions should be independently testable.

---

# Project Structure

The project is organized into independent modules.

Each module performs one specific task and can be maintained separately.

The pipelines (Flanker, Go/No-Go, Ready-Set-Go, TMT) only orchestrate the workflow by calling reusable modules.

No analysis logic should be duplicated between pipelines.
