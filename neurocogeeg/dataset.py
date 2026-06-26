"""
Dataset and data-path management utilities for NeuroCogEEG.

This module defines the fixed relative folder structure used by the project.
It does not load EEG data and does not perform any signal processing.

The purpose of this module is to keep all file-system logic in one place so
that pipelines do not contain hard-coded paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neurocogeeg.constants import SUPPORTED_EEG_EXTENSIONS


VALID_EXPERIMENTS = (
    "flanker",
    "gonogo",
    "readysetgo",
    "tmt",
)

VALID_GROUPS = (
    "control",
    "experimental",
)

OUTPUT_SUBDIRS = (
    "csv",
    "figures",
    "reports",
    "logs",
)


@dataclass(frozen=True)
class SubjectRecord:
    """
    Representation of a single participant EEG file.

    Attributes:
        subject_id:
            Participant identifier derived from the EDF filename stem.

        experiment:
            Experiment name, such as ``flanker`` or ``tmt``.

        group:
            Group name, either ``control`` or ``experimental``.

        edf_path:
            Full path to the participant's EDF file.
    """

    subject_id: str
    experiment: str
    group: str
    edf_path: Path


@dataclass(frozen=True)
class ExperimentDataset:
    """
    Accessor for one experiment's data and output folders.

    Parameters:
        project_root:
            Root directory of the NeuroCogEEG repository.

        experiment:
            Experiment name. Must be one of:
            ``flanker``, ``gonogo``, ``readysetgo`` or ``tmt``.
    """

    project_root: Path
    experiment: str

    def __post_init__(self) -> None:
        """Validate dataset initialization."""
        if self.experiment not in VALID_EXPERIMENTS:
            valid = ", ".join(VALID_EXPERIMENTS)
            raise ValueError(
                f"Unknown experiment: {self.experiment!r}. "
                f"Valid experiments are: {valid}"
            )

    @property
    def data_dir(self) -> Path:
        """Return the experiment-level data directory."""
        return self.project_root / "data" / self.experiment

    @property
    def raw_dir(self) -> Path:
        """Return the experiment-level raw data directory."""
        return self.data_dir / "raw"

    @property
    def behavior_dir(self) -> Path:
        """Return the experiment-level behavior directory."""
        return self.data_dir / "behavior"

    @property
    def output_dir(self) -> Path:
        """Return the experiment-level output directory."""
        return self.project_root / "outputs" / self.experiment

    @property
    def csv_dir(self) -> Path:
        """Return the CSV output directory."""
        return self.output_dir / "csv"

    @property
    def figures_dir(self) -> Path:
        """Return the figures output directory."""
        return self.output_dir / "figures"

    @property
    def reports_dir(self) -> Path:
        """Return the reports output directory."""
        return self.output_dir / "reports"

    @property
    def logs_dir(self) -> Path:
        """Return the logs output directory."""
        return self.output_dir / "logs"

    def group_raw_dir(self, group: str) -> Path:
        """
        Return the raw EDF directory for a group.

        Parameters:
            group:
                Group name. Must be either ``control`` or ``experimental``.

        Returns:
            Path to the raw EEG folder for the selected group.
        """
        self._validate_group(group)
        return self.raw_dir / group

    def list_edf_files(self, group: str) -> list[Path]:
        """
        List EDF files for a given group.

        Parameters:
            group:
                Group name. Must be either ``control`` or ``experimental``.

        Returns:
            Sorted list of EDF file paths.
        """
        raw_group_dir = self.group_raw_dir(group)

        if not raw_group_dir.exists():
            return []

        edf_files: list[Path] = []

        for file_path in raw_group_dir.iterdir():
            if (
                file_path.is_file()
                and file_path.suffix.lower() in SUPPORTED_EEG_EXTENSIONS
            ):
                edf_files.append(file_path)

        return sorted(edf_files)

    def list_subjects(self) -> list[SubjectRecord]:
        """
        List all participant EDF files for the experiment.

        Returns:
            List of ``SubjectRecord`` objects for both control and experimental
            groups.
        """
        subjects: list[SubjectRecord] = []

        for group in VALID_GROUPS:
            for edf_path in self.list_edf_files(group):
                subjects.append(
                    SubjectRecord(
                        subject_id=edf_path.stem,
                        experiment=self.experiment,
                        group=group,
                        edf_path=edf_path,
                    )
                )

        return subjects

    def ensure_output_dirs(self) -> None:
        """
        Create output directories if they do not already exist.

        This does not create raw data folders. Raw data folders are expected to
        exist as part of the project structure.
        """
        for output_subdir in OUTPUT_SUBDIRS:
            path = self.output_dir / output_subdir
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_group(group: str) -> None:
        """
        Validate group name.

        Parameters:
            group:
                Group name to validate.
        """
        if group not in VALID_GROUPS:
            valid = ", ".join(VALID_GROUPS)
            raise ValueError(
                f"Unknown group: {group!r}. Valid groups are: {valid}"
            )


def get_project_root() -> Path:
    """
    Return the NeuroCogEEG repository root directory.

    The function assumes this file is located at:

    ``NeuroCogEEG/neurocogeeg/dataset.py``

    Returns:
        Path to the repository root.
    """
    return Path(__file__).resolve().parents[1]


def get_dataset(experiment: str) -> ExperimentDataset:
    """
    Create an ``ExperimentDataset`` object for a given experiment.

    Parameters:
        experiment:
            Experiment name, such as ``flanker``, ``gonogo``,
            ``readysetgo`` or ``tmt``.

    Returns:
        Initialized ``ExperimentDataset``.
    """
    return ExperimentDataset(
        project_root=get_project_root(),
        experiment=experiment,
    )
