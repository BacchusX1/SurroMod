"""
Data Format Validators
======================
Reusable validation helpers for point-cloud and temporal field data
formats (e.g. GRaM challenge).

All validators raise ``ValueError`` with a descriptive message on failure.
"""

from __future__ import annotations

import numpy as np


def validate_pos(pos: np.ndarray) -> None:
    """Validate that *pos* is a 2-D array with shape (N, 3)."""
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(
            f"pos must be 2-D with shape (N, 3), got shape {pos.shape}"
        )


def validate_velocity(vel: np.ndarray, *, name: str = "velocity") -> None:
    """Validate that a velocity array is 3-D with shape (T, N, 3)."""
    if vel.ndim != 3 or vel.shape[2] != 3:
        raise ValueError(
            f"{name} must be 3-D with shape (T, N, 3), got shape {vel.shape}"
        )


def validate_velocity_pos_consistency(
    vel: np.ndarray, pos: np.ndarray, *, name: str = "velocity"
) -> None:
    """Check that the spatial dimension of *vel* matches *pos*."""
    if vel.shape[1] != pos.shape[0]:
        raise ValueError(
            f"{name} spatial dim ({vel.shape[1]}) does not match "
            f"pos point count ({pos.shape[0]})"
        )


def validate_idcs_airfoil(idcs: np.ndarray, num_points: int) -> None:
    """
    Validate an airfoil index array.

    Checks:
    - 1-D integer array
    - All values in [0, num_points)
    """
    if idcs.ndim != 1:
        raise ValueError(
            f"idcs_airfoil must be 1-D, got ndim={idcs.ndim}"
        )
    if not np.issubdtype(idcs.dtype, np.integer):
        raise ValueError(
            f"idcs_airfoil must be integer dtype, got {idcs.dtype}"
        )
    if idcs.size > 0:
        if idcs.min() < 0 or idcs.max() >= num_points:
            raise ValueError(
                f"idcs_airfoil values must be in [0, {num_points}), "
                f"got range [{idcs.min()}, {idcs.max()}]"
            )


def validate_temporal_point_cloud_sample(data: dict) -> None:
    """
    Full validation of a Temporal Point Cloud Field sample dict.

    Expected keys: ``pos``, ``velocity_in``, ``velocity_out``, optionally ``t``.
    """
    if "pos" not in data:
        raise ValueError("Sample is missing required key 'pos'")
    pos = np.asarray(data["pos"])
    validate_pos(pos)

    for key in ("velocity_in", "velocity_out"):
        if key not in data:
            raise ValueError(f"Sample is missing required key '{key}'")
        vel = np.asarray(data[key])
        validate_velocity(vel, name=key)
        validate_velocity_pos_consistency(vel, pos, name=key)

    if "t" in data:
        t = np.asarray(data["t"])
        if t.ndim != 1:
            raise ValueError(f"t must be 1-D, got ndim={t.ndim}")

