"""
NeuroCogEEG

Project-wide immutable constants.

Only values that are universally valid for the project should be
defined here.

Experiment-specific parameters (event markers, ERP windows,
ROI definitions, filters, epoch durations, preprocessing settings,
etc.) MUST be stored in the corresponding YAML configuration files.

Author
------
NeuroCogEEG Project

License
-------
MIT
"""

from __future__ import annotations

# =============================================================================
# UNIT CONVERSIONS
# =============================================================================

V_TO_UV = 1e6
UV_TO_V = 1e-6

S_TO_MS = 1000
MS_TO_S = 1e-3

# =============================================================================
# EEG FREQUENCY BANDS (Hz)
# =============================================================================

FREQUENCY_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# =============================================================================
# EMOTIV EPOC X CHANNEL ORDER
# =============================================================================

EMOTIV_EPOCX_CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]

# =============================================================================
# SUPPORTED FILE TYPES
# =============================================================================

SUPPORTED_EEG_EXTENSIONS = (
    ".edf",
)

SUPPORTED_EXPORT_EXTENSIONS = (
    ".csv",
    ".xlsx",
)

# =============================================================================
# DEFAULT NUMERIC DATA TYPES
# =============================================================================

FLOAT_DTYPE = "float64"
INTEGER_DTYPE = "int64"
