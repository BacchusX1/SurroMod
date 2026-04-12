"""
Surface Distance Feature
========================
Feature engineering node that computes the distance from each point
in a point cloud to the nearest surface point.

Inputs:
    - pos             : (N, 3)                – all point positions
    - surface_points  : (#surface_points, 3)  – surface point positions

Outputs:
    - dist_to_surface      : (N, 1)  – scalar distance to nearest surface point
    - nearest_surface_vec  : (N, 3)  – vector to nearest surface point (optional)
    - surface_distance_meta : dict   – metadata
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SurfaceDistanceFeature:
    """
    Compute distance-to-nearest-surface features for each point.

    Expected ``hyperparams``:
        - return_vector       : bool – include nearest_surface_vec (default False)
        - normalize_distance  : bool – normalise distances to [0, 1] (default False)
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = hyperparams or {}
        self._return_vector: bool = bool(hp.get("return_vector", False))
        self._normalize_distance: bool = bool(hp.get("normalize_distance", False))
        self._seed = seed

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Compute distance from each point to the nearest surface point.

        In **multi-sample** mode, extracts surface points from each
        sample's ``idcs_airfoil`` and computes ``geometry_mask`` and
        ``dist_to_surface`` per sample.
        """
        samples = inputs.get("samples")
        if samples is not None and isinstance(samples, list) and len(samples) > 0:
            return self._execute_multi(inputs)
        return self._execute_single(inputs)

    def _execute_single(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Compute surface distance for a single pos array."""
        pos = np.asarray(inputs["pos"], dtype=np.float32)
        surface_points = np.asarray(inputs["surface_points"], dtype=np.float32)

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"pos must have shape (N, 3), got {pos.shape}")
        if surface_points.ndim != 2 or surface_points.shape[1] != 3:
            raise ValueError(
                f"surface_points must have shape (M, 3), got {surface_points.shape}"
            )

        dist, vec = self._compute(pos, surface_points)

        result: dict[str, Any] = {**inputs}
        result["dist_to_surface"] = dist
        if vec is not None:
            result["nearest_surface_vec"] = vec

        result["surface_distance_meta"] = {
            "num_points": int(pos.shape[0]),
            "num_surface_points": int(surface_points.shape[0]),
            "return_vector": self._return_vector,
            "normalize_distance": self._normalize_distance,
        }

        logger.info(
            "SurfaceDistanceFeature: N=%d  surface=%d  return_vec=%s",
            pos.shape[0], surface_points.shape[0], self._return_vector,
        )
        return result

    def _execute_multi(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Compute surface distance and geometry mask for each sample."""
        samples = inputs["samples"]

        # Detect lazy mode
        lazy = "filepath" in samples[0] and "pos" not in samples[0]

        if lazy:
            if "fe_pipeline" not in inputs:
                inputs["fe_pipeline"] = []
            inputs["fe_pipeline"].append(self)
            logger.info(
                "SurfaceDistanceFeature[lazy]: registered for %d samples",
                len(samples),
            )
            return {**inputs}

        # Eager mode
        for sample in samples:
            self.process_sample(sample)

        logger.info(
            "SurfaceDistanceFeature[multi]: %d samples", len(samples),
        )
        return {**inputs}

    # ── lazy per-sample processing (called by GFF) ───────────────────────

    def process_sample(self, sample_data: dict[str, Any]) -> None:
        """Compute dist_to_surface + geometry_mask in-place."""
        pos = np.asarray(sample_data["pos"], dtype=np.float32)
        N = pos.shape[0]

        idcs_airfoil = sample_data.get("idcs_airfoil")
        if idcs_airfoil is not None and len(idcs_airfoil) > 0:
            surface_points = pos[idcs_airfoil]
            dist, vec = self._compute(pos, surface_points)
            geo_mask = np.zeros((N, 1), dtype=np.float32)
            geo_mask[idcs_airfoil, 0] = 1.0
        else:
            dist = np.zeros((N, 1), dtype=np.float32)
            vec = np.zeros((N, 3), dtype=np.float32) if self._return_vector else None
            geo_mask = np.zeros((N, 1), dtype=np.float32)

        sample_data["dist_to_surface"] = dist
        sample_data["geometry_mask"] = geo_mask
        if vec is not None:
            sample_data["nearest_surface_vec"] = vec

    def _compute(
        self, pos: np.ndarray, surface_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Core computation: distance (N,1) and optional vector (N,3)."""
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(surface_points)
        distances, indices = nn.kneighbors(pos)

        dist = distances.astype(np.float32)  # (N, 1)
        if self._normalize_distance and dist.max() > 0:
            dist = dist / dist.max()

        vec = None
        if self._return_vector:
            nearest_pts = surface_points[indices.ravel()]
            vec = (nearest_pts - pos).astype(np.float32)

        return dist, vec
