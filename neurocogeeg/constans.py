"""
NeuroCogEEG - constants.py

Project-wide constants.

This module contains values that are not expected to change
between experiments.

Author:
    NeuroCogEEG Project

License:
    MIT
"""

from __future__ import annotations

# =============================================================================
# EEG FREQUENCY BANDS (Hz)
# =============================================================================

DELTA = (1.0, 4.0)
THETA = (4.0, 8.0)
ALPHA = (8.0, 13.0)
BETA = (13.0, 30.0)
GAMMA = (30.0, 45.0)

FREQUENCY_BANDS = {
    "delta": DELTA,
    "theta": THETA,
    "alpha": ALPHA,
    "beta": BETA,
    "gamma": GAMMA,
}
# =============================================================================
# ERP COMPONENT WINDOWS (seconds)
# =============================================================================

# Readiness Potential
RP_WINDOW = (-0.50, 0.00)

# Post-Movement Positivity
PMP_WINDOW = (0.15, 0.35)

# Error-Related Negativity
ERN_WINDOW = (0.00, 0.10)

# N2
N2_WINDOW = (0.20, 0.35)

# P3
P3_WINDOW = (0.30, 0.60)

ERP_WINDOWS = {
    "RP": RP_WINDOW,
    "PMP": PMP_WINDOW,
    "ERN": ERN_WINDOW,
    "N2": N2_WINDOW,
    "P3": P3_WINDOW,
}
# =============================================================================
# EMOTIV EPOCX CHANNELS
# =============================================================================

EMOTIV_CHANNELS = [
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
# REGIONS OF INTEREST (ROI)
# =============================================================================

ROI = {

    "frontal": [
        "F3",
        "F4",
        "FC5",
        "FC6",
    ],

    "left": [
        "F3",
        "FC5",
        "T7",
        "P7",
        "O1",
    ],

    "right": [
        "F4",
        "FC6",
        "T8",
        "P8",
        "O2",
    ],

    "parietal": [
        "P7",
        "P8",
        "O1",
        "O2",
    ]
}
# =============================================================================
# UNIT CONVERSIONS
# =============================================================================

VOLT_TO_UV = 1e6

UV_TO_VOLT = 1e-6
