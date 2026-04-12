"""
Point Feature Fusion
====================
Feature engineering node that concatenates multiple per-point features
along the last dimension into a single feature tensor.

Concatenates any subset of:
    - pos                       : (N, 3)
    - velocity_history_features : (N, T_in * 3)
    - geometry_mask             : (N, 1)
    - dist_to_surface           : (N, 1)
    - nearest_surface_vec       : (N, 3)  (optional)
    - pressure_features         : (N, T_in)  – flattened pressure (optional)

Output: ``point_features`` with shape (N, D) where D is the sum of
enabled feature dimensions.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PointFeatureFusion:
    """
    Concatenate enabled per-point features into a single feature matrix.

    Expected ``hyperparams``:
        - include_pos                  : bool (default True)
        - include_velocity_history     : bool (default True)
        - include_geometry_mask        : bool (default True)
        - include_dist_to_surface      : bool (default True)
        - include_nearest_surface_vec  : bool (default False)
        - include_pressure             : bool (default False)
        - include_low_freq             : bool (default False) – include vel_low_freq_features
    - include_high_freq            : bool (default False) – include vel_high_freq_features
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = hyperparams or {}
        self._include_pos: bool = bool(hp.get("include_pos", True))
        self._include_velocity_history: bool = bool(hp.get("include_velocity_history", True))
        self._include_geometry_mask: bool = bool(hp.get("include_geometry_mask", True))
        self._include_dist_to_surface: bool = bool(hp.get("include_dist_to_surface", True))
        self._include_nearest_surface_vec: bool = bool(hp.get("include_nearest_surface_vec", False))
        self._include_pressure: bool = bool(hp.get("include_pressure", False))
        self._include_low_freq: bool = bool(hp.get("include_low_freq", False))
        self._include_high_freq: bool = bool(hp.get("include_high_freq", False))
        self._seed = seed

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Concatenate selected per-point features into ``point_features``.

        In multi-sample mode, fuses features for each sample.
        """
        samples = inputs.get("samples")
        if samples is not None and isinstance(samples, list) and len(samples) > 0:
            return self._execute_multi(inputs)
        return self._execute_single(inputs)

    def _execute_single(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Fuse features for a single sample (legacy mode)."""
        point_features, feature_dims = self._fuse(inputs)
        N, D = point_features.shape

        result: dict[str, Any] = {**inputs}
        result["point_features"] = point_features
        result["feature_meta"] = {
            "num_points": int(N),
            "total_dim": int(D),
            "feature_dims": feature_dims,
        }

        logger.info(
            "PointFeatureFusion: N=%d  D=%d  components=%s",
            N, D, list(feature_dims.keys()),
        )
        return result

    def _execute_multi(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Fuse features for each sample."""
        samples = inputs["samples"]

        # Detect lazy mode
        lazy = "filepath" in samples[0] and "pos" not in samples[0]

        if lazy:
            if "fe_pipeline" not in inputs:
                inputs["fe_pipeline"] = []
            inputs["fe_pipeline"].append(self)
            logger.info(
                "PointFeatureFusion[lazy]: registered for %d samples",
                len(samples),
            )
            return {**inputs}

        # Eager mode
        for sample in samples:
            pf, _ = self._fuse(sample)
            sample["point_features"] = pf
        D = samples[0]["point_features"].shape[1] if samples else 0
        logger.info(
            "PointFeatureFusion[multi]: %d samples, D=%d", len(samples), D,
        )
        return {**inputs}

    # ── lazy per-sample processing (called by GFF) ───────────────────────

    def process_sample(self, sample_data: dict[str, Any]) -> None:
        """Fuse per-point features in-place."""
        pf, _ = self._fuse(sample_data)
        sample_data["point_features"] = pf

    def _fuse(self, data: dict[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
        """Concatenate enabled features from *data* dict.  Returns (point_features, dims)."""
        parts: list[np.ndarray] = []
        feature_dims: dict[str, int] = {}
        N = self._infer_num_points(data)

        if self._include_pos and "pos" in data:
            arr = np.asarray(data["pos"], dtype=np.float32)
            self._validate_shape(arr, N, "pos")
            parts.append(arr)
            feature_dims["pos"] = arr.shape[1]

        if self._include_velocity_history and "velocity_history_features" in data:
            arr = np.asarray(data["velocity_history_features"], dtype=np.float32)
            self._validate_shape(arr, N, "velocity_history_features")
            parts.append(arr)
            feature_dims["velocity_history_features"] = arr.shape[1]

        if self._include_geometry_mask and "geometry_mask" in data:
            arr = np.asarray(data["geometry_mask"], dtype=np.float32)
            self._validate_shape(arr, N, "geometry_mask")
            parts.append(arr)
            feature_dims["geometry_mask"] = arr.shape[1]

        if self._include_dist_to_surface and "dist_to_surface" in data:
            arr = np.asarray(data["dist_to_surface"], dtype=np.float32)
            self._validate_shape(arr, N, "dist_to_surface")
            parts.append(arr)
            feature_dims["dist_to_surface"] = arr.shape[1]

        if self._include_nearest_surface_vec and "nearest_surface_vec" in data:
            arr = np.asarray(data["nearest_surface_vec"], dtype=np.float32)
            self._validate_shape(arr, N, "nearest_surface_vec")
            parts.append(arr)
            feature_dims["nearest_surface_vec"] = arr.shape[1]

        if self._include_pressure and "pressure" in data:
            # pressure shape: (T, N) → flatten to (N, T)
            p = np.asarray(data["pressure"], dtype=np.float32)
            if p.ndim == 2 and p.shape[1] == N:
                arr = p.T  # (N, T)
            elif p.ndim == 2 and p.shape[0] == N:
                arr = p    # already (N, T)
            else:
                raise ValueError(
                    f"pressure must have shape (T, N) or (N, T), got {p.shape}"
                )
            parts.append(arr)
            feature_dims["pressure"] = arr.shape[1]

        if self._include_low_freq and "vel_low_freq_features" in data:
            arr = np.asarray(data["vel_low_freq_features"], dtype=np.float32)
            self._validate_shape(arr, N, "vel_low_freq_features")
            parts.append(arr)
            feature_dims["vel_low_freq_features"] = arr.shape[1]

        if self._include_high_freq and "vel_high_freq_features" in data:
            arr = np.asarray(data["vel_high_freq_features"], dtype=np.float32)
            self._validate_shape(arr, N, "vel_high_freq_features")
            parts.append(arr)
            feature_dims["vel_high_freq_features"] = arr.shape[1]

        if not parts:
            raise ValueError(
                "PointFeatureFusion: no features selected or available."
            )
        return np.concatenate(parts, axis=1).astype(np.float32), feature_dims

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _infer_num_points(inputs: dict[str, Any]) -> int:
        """Infer N from the first available per-point array."""
        for key in ("pos", "velocity_history_features", "geometry_mask",
                     "dist_to_surface", "nearest_surface_vec", "pressure_features"):
            if key in inputs:
                arr = np.asarray(inputs[key])
                if arr.ndim >= 2:
                    return arr.shape[0]
        raise ValueError("Cannot infer number of points — no per-point arrays found.")

    @staticmethod
    def _validate_shape(arr: np.ndarray, N: int, name: str) -> None:
        """Ensure the array is 2-D with N rows."""
        if arr.ndim == 1:
            raise ValueError(
                f"{name} must be 2-D (N, D), got 1-D shape {arr.shape}. "
                f"Reshape with .reshape(-1, 1) if it's a scalar feature."
            )
        if arr.ndim != 2 or arr.shape[0] != N:
            raise ValueError(
                f"{name} must have shape (N={N}, D), got shape {arr.shape}"
            )
