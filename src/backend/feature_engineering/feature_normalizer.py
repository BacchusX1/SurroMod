"""
Feature Normalizer
==================
Per-feature standardisation or normalisation for point-cloud feature matrices.

Fits normalisation statistics on the **training** samples and applies the
same transform at inference / test time, so the GFF always receives
zero-mean, unit-variance (or min-max scaled) inputs.

Supports three modes
--------------------
standard   – subtract mean, divide by std  (zero-mean, unit-variance)
minmax     – scale to [0, 1] per feature
robust     – subtract median, divide by IQR  (outlier-resistant)

Hyperparameters
---------------
mode            : str   – "standard" | "minmax" | "robust"  (default "standard")
per_component   : bool  – True = fit one scaler per XYZ component of velocity
                          features; False = treat all dims identically
                          (default False)
epsilon         : float – small constant added to denominator to avoid /0
                          (default 1e-8)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """
    Pipeline-executable per-feature normaliser.

    In **lazy multi-sample** mode (batch / GRaM pipeline) the node
    registers itself into ``fe_pipeline`` so the GFF applies it on every
    mini-batch.  It fits statistics lazily on the first call.

    Expected ``hyperparams``
    ------------------------
    mode          : "standard" | "minmax" | "robust"
    per_component : bool   (default False)
    epsilon       : float  (default 1e-8)
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = hyperparams or {}
        # Accept both the unique frontend key and the short alias
        self._mode: str = str(hp.get("normalizer_mode", hp.get("mode", "standard")))
        self._per_component: bool = bool(hp.get("per_component", False))
        self._epsilon: float = float(hp.get("epsilon", 1e-8))

        # Fitted statistics (None until first fit)
        self._center: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._is_fitted: bool = False

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        samples = inputs.get("samples")
        if samples is not None and isinstance(samples, list) and len(samples) > 0:
            return self._execute_multi(inputs)
        return self._execute_single(inputs)

    def _execute_single(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Fit + transform a single point_features matrix."""
        pf = inputs.get("point_features")
        if pf is None:
            logger.warning("FeatureNormalizer: no point_features found – pass-through.")
            return {**inputs}
        pf = np.asarray(pf, dtype=np.float32)
        if not self._is_fitted:
            self._fit(pf)
        result = {**inputs}
        result["point_features"] = self._transform(pf)
        return result

    def _execute_multi(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Register in fe_pipeline for lazy execution by the GFF."""
        samples = inputs["samples"]
        lazy = "filepath" in samples[0] and "point_features" not in samples[0]

        if lazy:
            if "fe_pipeline" not in inputs:
                inputs["fe_pipeline"] = []
            inputs["fe_pipeline"].append(self)
            logger.info(
                "FeatureNormalizer[lazy]: registered for %d samples (will fit on first batch).",
                len(samples),
            )
            return {**inputs}

        # Eager: fit on ALL training samples' point_features
        all_pf = [
            np.asarray(s["point_features"], dtype=np.float32)
            for s in samples
            if "point_features" in s
        ]
        if all_pf:
            combined = np.concatenate(all_pf, axis=0)
            self._fit(combined)
            for s in samples:
                if "point_features" in s:
                    s["point_features"] = self._transform(
                        np.asarray(s["point_features"], dtype=np.float32)
                    )
        return {**inputs}

    # ── lazy per-sample processing (called by GFF) ───────────────────────

    def fit_on_samples(self, samples_data: list[dict[str, Any]]) -> None:
        """
        Fit normalisation statistics on a collection of already-processed
        sample dicts (each must have ``point_features`` populated).

        Called by the GFF before the training loop so that statistics are
        derived from the full training set rather than the first sample seen.

        Uses an **incremental (Welford-style) pass** so that only one sample's
        point_features array is live at a time, keeping peak RAM at O(1 sample)
        instead of O(N_samples * N_points * D).  For standard mode this is
        exact; for minmax/robust we collect per-sample min/max/quantiles and
        aggregate them.
        """
        valid = [
            s for s in samples_data if "point_features" in s
        ]
        if not valid:
            return

        n_samples = len(valid)
        D = np.asarray(valid[0]["point_features"], dtype=np.float32).shape[1]

        if self._mode == "standard":
            # Welford online mean + M2 (for variance)
            n_pts = 0
            mean = np.zeros(D, dtype=np.float64)
            M2   = np.zeros(D, dtype=np.float64)
            for s in valid:
                X = np.asarray(s["point_features"], dtype=np.float64)  # (N_i, D)
                m = X.shape[0]
                b_mean = X.mean(axis=0)
                b_M2   = ((X - b_mean) ** 2).sum(axis=0)
                delta  = b_mean - mean
                n_new  = n_pts + m
                mean  += delta * (m / n_new)
                M2    += b_M2 + delta ** 2 * (n_pts * m / n_new)
                n_pts  = n_new
            self._center = mean.astype(np.float32)
            self._scale  = np.sqrt(M2 / max(n_pts, 1)).astype(np.float32)

        elif self._mode == "minmax":
            # Track running per-feature min and max
            global_min = np.full(D, np.inf,  dtype=np.float64)
            global_max = np.full(D, -np.inf, dtype=np.float64)
            for s in valid:
                X = np.asarray(s["point_features"], dtype=np.float64)
                global_min = np.minimum(global_min, X.min(axis=0))
                global_max = np.maximum(global_max, X.max(axis=0))
            self._center = global_min.astype(np.float32)
            self._scale  = (global_max - global_min).astype(np.float32)

        else:
            # robust (median / IQR): reservoir-sample up to 200 K points to
            # keep memory bounded, then use numpy percentile.
            rng = np.random.RandomState(42)
            reservoir: list[np.ndarray] = []
            reservoir_n = 0
            cap = 200_000          # max points kept in RAM
            for s in valid:
                X = np.asarray(s["point_features"], dtype=np.float32)
                m = X.shape[0]
                if reservoir_n + m <= cap:
                    reservoir.append(X)
                    reservoir_n += m
                else:
                    # Randomly replace existing rows to keep total ≤ cap
                    keep = max(0, cap - reservoir_n)
                    if keep > 0:
                        reservoir.append(X[:keep])
                        reservoir_n += keep
                    # reservoir is now full; stop collecting
                    break
            combined = np.concatenate(reservoir, axis=0)
            self._center = np.median(combined, axis=0).astype(np.float32)
            q75 = np.percentile(combined, 75, axis=0).astype(np.float32)
            q25 = np.percentile(combined, 25, axis=0).astype(np.float32)
            self._scale = (q75 - q25).astype(np.float32)

        self._scale = np.where(self._scale < self._epsilon, 1.0, self._scale).astype(np.float32)
        self._is_fitted = True

        logger.info(
            "FeatureNormalizer.fit_on_samples: fitted incrementally on %d samples "
            "(mode=%s, D=%d, center=[%.4f…%.4f], scale=[%.4f…%.4f])",
            n_samples, self._mode, D,
            float(self._center.min()), float(self._center.max()),
            float(self._scale.min()),  float(self._scale.max()),
        )

    # ── incremental streaming fit API (called by GFF to avoid OOM) ──────────

    def begin_incremental_fit(self) -> None:
        """Start a streaming fit pass.  Resets all accumulators."""
        self._inc_n: int = 0
        self._inc_D: int | None = None
        # standard
        self._inc_mean: np.ndarray | None = None
        self._inc_M2:   np.ndarray | None = None
        # minmax
        self._inc_min: np.ndarray | None = None
        self._inc_max: np.ndarray | None = None
        # robust: reservoir
        self._inc_reservoir: list[np.ndarray] = []
        self._inc_reservoir_n: int = 0
        self._inc_reservoir_cap: int = 200_000

    def update_incremental_fit(self, X: np.ndarray) -> None:
        """Feed one sample's point_features (N_i, D) into the accumulator."""
        X = np.asarray(X, dtype=np.float64)
        m = X.shape[0]
        D = X.shape[1] if X.ndim > 1 else 1

        if self._inc_D is None:
            self._inc_D = D
            if self._mode == "standard":
                self._inc_mean = np.zeros(D, dtype=np.float64)
                self._inc_M2   = np.zeros(D, dtype=np.float64)
            elif self._mode == "minmax":
                self._inc_min = np.full(D, np.inf,  dtype=np.float64)
                self._inc_max = np.full(D, -np.inf, dtype=np.float64)

        if self._mode == "standard":
            b_mean = X.mean(axis=0)
            b_M2   = ((X - b_mean) ** 2).sum(axis=0)
            delta  = b_mean - self._inc_mean
            n_new  = self._inc_n + m
            self._inc_mean += delta * (m / n_new)
            self._inc_M2   += b_M2 + delta ** 2 * (self._inc_n * m / n_new)
        elif self._mode == "minmax":
            self._inc_min = np.minimum(self._inc_min, X.min(axis=0))
            self._inc_max = np.maximum(self._inc_max, X.max(axis=0))
        else:  # robust: reservoir sampling
            if self._inc_reservoir_n < self._inc_reservoir_cap:
                keep = min(m, self._inc_reservoir_cap - self._inc_reservoir_n)
                self._inc_reservoir.append(X[:keep].astype(np.float32))
                self._inc_reservoir_n += keep

        self._inc_n += m

    def finalize_incremental_fit(self) -> None:
        """Compute final statistics from the accumulator and mark as fitted."""
        if self._inc_n == 0:
            return

        if self._mode == "standard":
            self._center = self._inc_mean.astype(np.float32)
            self._scale  = np.sqrt(self._inc_M2 / max(self._inc_n, 1)).astype(np.float32)
        elif self._mode == "minmax":
            self._center = self._inc_min.astype(np.float32)
            self._scale  = (self._inc_max - self._inc_min).astype(np.float32)
        else:  # robust
            combined = np.concatenate(self._inc_reservoir, axis=0)
            self._center = np.median(combined, axis=0).astype(np.float32)
            q75 = np.percentile(combined, 75, axis=0).astype(np.float32)
            q25 = np.percentile(combined, 25, axis=0).astype(np.float32)
            self._scale = (q75 - q25).astype(np.float32)
            del combined, self._inc_reservoir

        self._scale = np.where(
            self._scale < self._epsilon, 1.0, self._scale,
        ).astype(np.float32)
        self._is_fitted = True

        D = self._inc_D or 0
        logger.info(
            "FeatureNormalizer.finalize_incremental_fit: mode=%s  D=%d  n_pts=%d  "
            "center=[%.4f…%.4f]  scale=[%.4f…%.4f]",
            self._mode, D, self._inc_n,
            float(self._center.min()), float(self._center.max()),
            float(self._scale.min()),  float(self._scale.max()),
        )

    # ── lazy per-sample processing (called by GFF) ───────────────────────

    def process_sample(self, sample_data: dict[str, Any]) -> None:
        """Normalise point_features in-place; fit statistics if not yet fitted."""
        pf = sample_data.get("point_features")
        if pf is None:
            return
        pf = np.asarray(pf, dtype=np.float32)
        if not self._is_fitted:
            self._fit(pf)
        sample_data["point_features"] = self._transform(pf)

    # ── core fit / transform ─────────────────────────────────────────────

    def _fit(self, X: np.ndarray) -> None:
        """Compute and store normalisation statistics from X (N, D)."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        axis = 0  # fit across the N (point) dimension

        if self._mode == "standard":
            self._center = X.mean(axis=axis)
            self._scale = X.std(axis=axis)
        elif self._mode == "minmax":
            self._center = X.min(axis=axis)
            self._scale = X.max(axis=axis) - X.min(axis=axis)
        elif self._mode == "robust":
            self._center = np.median(X, axis=axis)
            q75 = np.percentile(X, 75, axis=axis)
            q25 = np.percentile(X, 25, axis=axis)
            self._scale = q75 - q25
        else:
            raise ValueError(f"FeatureNormalizer: unknown mode '{self._mode}'.")

        # Protect against zero-scale features
        self._scale = np.where(self._scale < self._epsilon, 1.0, self._scale)
        self._is_fitted = True

        logger.info(
            "FeatureNormalizer: fitted  mode=%s  D=%d  "
            "center=[%.4f…%.4f]  scale=[%.4f…%.4f]",
            self._mode, X.shape[1] if X.ndim > 1 else 1,
            float(self._center.min()), float(self._center.max()),
            float(self._scale.min()),  float(self._scale.max()),
        )

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Apply stored statistics to X (N, D) → (N, D) float32."""
        return ((X - self._center) / self._scale).astype(np.float32)
