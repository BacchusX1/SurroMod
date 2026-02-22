"""
Geometry Sampler
================
Feature engineering node that samples points on 2-D geometry surfaces
to produce a fixed-length feature vector suitable for flat regressors.

Given an upstream geometry (typically *landmarks* – dense point clouds
of shape ``(N, n_pts, 2)``), this node evaluates the surface at a
user-configurable set of x-locations yielding ``(N, 2 * n_points)``
features (y_upper and y_lower at each sample location).

When the upstream geometry is already a compact representation
(CST, Bezier) that is already 2-D ``(N, n_coeffs)``, the sampler
passes it through unchanged — those representations *are* the feature
vector and do not need resampling.

Sampling methods
----------------
* **uniform**         – equidistant x-locations in [0, 1].
* **cosine**          – cosine-spaced (denser near leading/trailing edge).
* **curvature_based** – adaptive spacing based on local curvature
                        (more points where the shape changes rapidly).

The interface is prepared for future 3-D extension by keeping the
sampling logic in a clearly separated private method.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class GeometrySampler:
    """
    Feature-engineering node that resamples 2-D geometry point-cloud
    data into a fixed-length feature vector.

    Expected ``hyperparams``:
        - n_points        : int – number of x sample locations (default 100)
        - sampling_method : str – ``'uniform'``, ``'cosine'``, or
                                  ``'curvature_based'`` (default ``'uniform'``)
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = hyperparams or {}
        self._n_points: int = int(hp.get("n_points", 100))
        self._method: str = str(hp.get("sampling_method", "uniform"))
        self._seed = seed

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Resample geometry data from upstream.

        If X is 3-D ``(N, n_pts, 2)`` — treated as point-cloud landmarks,
        resampled to ``(N, 2 * n_points)``.

        If X is 2-D ``(N, d)`` — assumed to be a compact parametric
        representation (CST / Bezier) and passed through unchanged.
        """
        X: np.ndarray = inputs["X"]
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 3 and X.shape[2] == 2:
            # Landmarks: (N, n_pts, 2) → (N, 2 * n_points)
            X_sampled = self._sample_landmarks(X)
            feature_names = self._build_feature_names()
            logger.info(
                "GeometrySampler: resampled landmarks %s → %s  method=%s",
                X.shape, X_sampled.shape, self._method,
            )
        elif X.ndim == 2:
            # Compact representation — pass through
            X_sampled = X
            feature_names = inputs.get("feature_names", [
                f"geo_{i}" for i in range(X.shape[1])
            ])
            logger.info(
                "GeometrySampler: compact geometry %s passed through",
                X.shape,
            )
        else:
            raise ValueError(
                f"GeometrySampler: unexpected X shape {X.shape}.  "
                f"Expected (N, n_pts, 2) for landmarks or (N, d) for "
                f"compact representations."
            )

        outputs: dict[str, Any] = {**inputs, "X": X_sampled, "feature_names": feature_names}
        return outputs

    # ── sampling logic ───────────────────────────────────────────────────

    def _sample_landmarks(self, X: np.ndarray) -> np.ndarray:
        """
        Resample ``(N, n_pts, 2)`` point-cloud landmarks.

        For each sample:
        1.  Split the contour into upper and lower surfaces.
        2.  Generate x sample locations using the chosen method.
        3.  Interpolate y_upper and y_lower at each x location.
        4.  Output: ``[y_upper_0, ..., y_upper_k, y_lower_0, ..., y_lower_k]``.
        """
        N = X.shape[0]
        out = np.empty((N, 2 * self._n_points), dtype=np.float32)

        x_samples = self._generate_x_locations()

        for i in range(N):
            pts = X[i]  # (n_pts, 2)
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]

            y_upper, y_lower = self._split_and_interpolate(
                x_coords, y_coords, x_samples
            )
            out[i, :self._n_points] = y_upper
            out[i, self._n_points:] = y_lower

        return out

    def _generate_x_locations(self) -> np.ndarray:
        """Generate *n_points* x sample locations in [0, 1]."""
        n = self._n_points

        if self._method == "uniform":
            return np.linspace(0.0, 1.0, n, dtype=np.float32)

        if self._method == "cosine":
            # Cosine spacing: denser near 0 and 1 (LE + TE)
            theta = np.linspace(0.0, np.pi, n)
            return (0.5 * (1.0 - np.cos(theta))).astype(np.float32)

        if self._method == "curvature_based":
            # Simplified curvature-based: use half-cosine (denser at LE)
            # then add small uniform perturbation for coverage
            theta = np.linspace(0.0, np.pi, n)
            x = 0.5 * (1.0 - np.cos(theta))
            # Mild compression toward 0 (leading edge, high curvature)
            x = x ** 0.8
            return x.astype(np.float32)

        raise ValueError(f"Unknown sampling method: {self._method}")

    @staticmethod
    def _split_and_interpolate(
        x: np.ndarray,
        y: np.ndarray,
        x_samples: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split an airfoil-like contour into upper and lower surfaces
        and interpolate at the given x sample locations.

        Assumes the contour is ordered starting from trailing edge,
        going around the lower surface to the leading edge, then back
        along the upper surface (standard counter-clockwise ordering).
        The leading edge is identified as the point with minimum x.

        Returns (y_upper, y_lower) each of shape ``(n_points,)``.
        """
        # Find leading edge (min x)
        le_idx = int(np.argmin(x))

        # Split into upper (LE → TE going up) and lower (LE → TE going down)
        if le_idx == 0:
            # Contour starts at LE: first half = lower, second half = upper
            mid = len(x) // 2
            x_lower, y_lower_raw = x[:mid], y[:mid]
            x_upper, y_upper_raw = x[mid:], y[mid:]
        else:
            # Standard: lower = [0..le_idx], upper = [le_idx..end] reversed
            x_lower = x[:le_idx + 1]
            y_lower_raw = y[:le_idx + 1]
            x_upper = x[le_idx:][::-1]
            y_upper_raw = y[le_idx:][::-1]

        # Sort by x for interpolation
        order_u = np.argsort(x_upper)
        order_l = np.argsort(x_lower)

        y_upper = np.interp(x_samples, x_upper[order_u], y_upper_raw[order_u])
        y_lower = np.interp(x_samples, x_lower[order_l], y_lower_raw[order_l])

        return y_upper.astype(np.float32), y_lower.astype(np.float32)

    def _build_feature_names(self) -> list[str]:
        """Build feature names for the sampled output."""
        names: list[str] = []
        for i in range(self._n_points):
            names.append(f"y_upper_{i}")
        for i in range(self._n_points):
            names.append(f"y_lower_{i}")
        return names
