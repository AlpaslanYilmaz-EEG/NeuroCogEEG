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
from typing import Iterable

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

EXPERIMENT_VARIANTS = {
    "tmt": (
        "tmt1",
        "tmt2",
    ),
}


@dataclass(frozen=True)
class SubjectRecord:
    """
    Representation of a single participant EEG file.

    Attributes
    ----------
    subject_id:
        Participant identifier derived from the EDF filename stem.

    experiment:
        Experiment name, such as ``flanker`` or ``tmt``.

    group:
        Group name, either ``control`` or ``experimental``.

    edf_path:
        Full path to the participant's EDF file.

    variant:
        Optional experiment variant. For example, TMT uses ``tmt1`` and
        ``tmt2``. Experiments without variants use ``None``.
    """

    subject_id: str
    experiment: str
    group: str
    edf_path: Path
    variant: str | None = None


@dataclass(frozen=True)
class ExperimentDataset:
    """
    Accessor for one experiment's data and output folders.

    Parameters
    ----------
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

    @property
    def variants(self) -> tuple[str, ...]:
        """
        Return variants for the current experiment.

        Experiments without variants return an empty tuple.
        """
        return EXPERIMENT_VARIANTS.get(self.experiment, ())

    @property
    def has_variants(self) -> bool:
        """
        Return whether this experiment has variants.

        Example
        -------
        TMT has variants: ``tmt1`` and ``tmt2``.
        """
        return len(self.variants) > 0

    def group_raw_dir(
        self,
        group: str,
        variant: str | None = None,
    ) -> Path:
        """
        Return the raw EDF directory for a group.

        Parameters
        ----------
        group:
            Group name. Must be either ``control`` or ``experimental``.

        variant:
            Optional experiment variant. Required for variant-based
            experiments such as TMT.

        Returns
        -------
        Path
            Path to the raw EEG folder for the selected group.
        """
        self._validate_group(group)

        if self.has_variants:
            self._validate_variant(variant)
            return self.raw_dir / str(variant) / group

        return self.raw_dir / group

    def list_edf_files(
        self,
        group: str,
        variant: str | None = None,
    ) -> list[Path]:
        """
        List EDF files for a given group.

        Parameters
        ----------
        group:
            Group name. Must be either ``control`` or ``experimental``.

        variant:
            Optional experiment variant. Required for variant-based
            experiments such as TMT.

        Returns
        -------
        list[Path]
            Sorted list of EDF file paths.
        """
        raw_group_dir = self.group_raw_dir(
            group=group,
            variant=variant,
        )

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

    def list_subjects(
        self,
        variants: Iterable[str] | None = None,
    ) -> list[SubjectRecord]:
        """
        List all participant EDF files for the experiment.

        Parameters
        ----------
        variants:
            Optional variant names to include. Only used for experiments with
            variants.

        Returns
        -------
        list[SubjectRecord]
            List of ``SubjectRecord`` objects.
        """
        subjects: list[SubjectRecord] = []

        if self.has_variants:
            selected_variants = (
                tuple(variants)
                if variants is not None
                else self.variants
            )

            for variant in selected_variants:
                self._validate_variant(variant)

                for group in VALID_GROUPS:
                    for edf_path in self.list_edf_files(
                        group=group,
                        variant=variant,
                    ):
                        subjects.append(
                            SubjectRecord(
                                subject_id=edf_path.stem,
                                experiment=self.experiment,
                                group=group,
                                edf_path=edf_path,
                                variant=variant,
                            )
                        )

            return subjects

        for group in VALID_GROUPS:
            for edf_path in self.list_edf_files(group=group):
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

        Parameters
        ----------
        group:
            Group name to validate.
        """
        if group not in VALID_GROUPS:
            valid = ", ".join(VALID_GROUPS)
            raise ValueError(
                f"Unknown group: {group!r}. Valid groups are: {valid}"
            )

    def _validate_variant(self, variant: str | None) -> None:
        """
        Validate experiment variant.

        Parameters
        ----------
        variant:
            Variant name to validate.
        """
        if not self.has_variants:
            if variant is not None:
                raise ValueError(
                    f"Experiment {self.experiment!r} does not use variants."
                )
            return

        if variant is None:
            valid = ", ".join(self.variants)
            raise ValueError(
                f"Experiment {self.experiment!r} requires a variant. "
                f"Valid variants are: {valid}"
            )

        if variant not in self.variants:
            valid = ", ".join(self.variants)
            raise ValueError(
                f"Unknown variant: {variant!r}. "
                f"Valid variants for {self.experiment!r} are: {valid}"
            )


def get_project_root() -> Path:
    """
    Return the NeuroCogEEG repository root directory.

    The function assumes this file is located at:

    ``NeuroCogEEG/neurocogeeg/dataset.py``

    Returns
    -------
    Path
        Path to the repository root.
    """
    return Path(__file__).resolve().parents[1]


def get_dataset(experiment: str) -> ExperimentDataset:
    """
    Create an ``ExperimentDataset`` object for a given experiment.

    Parameters
    ----------
    experiment:
        Experiment name, such as ``flanker``, ``gonogo``,
        ``readysetgo`` or ``tmt``.

    Returns
    -------
    ExperimentDataset
        Initialized ``ExperimentDataset``.
    """
    return ExperimentDataset(
        project_root=get_project_root(),
        experiment=experiment,
    )