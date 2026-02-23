"""
Scaler
======
Data scaling / normalisation for the SurroMod pipeline.

Supports MinMax, Standard (z-score), and Log Transform.
Stores the fitted scaler so inverse_transform can be called later
(e.g. for validator plots).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Scaler:
    """
    Data scaler for feature engineering in surrogate modelling pipelines.

    Expected ``node_data`` hyperparams:
        - method : str – one of 'MinMax', 'Standard', 'LogTransform'
    """

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        hp = hyperparams or {}
        self._method: str = str(hp.get("method", "MinMax"))

        # Fitted parameters (set in fit)
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._fitted = False

    # ── core API ─────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "Scaler":
        """Compute scaling parameters from *X* (n_samples, n_features)."""
        X = np.asarray(X, dtype=np.float32)

        if self._method == "MinMax":
            self._min = X.min(axis=0)
            self._max = X.max(axis=0)
        elif self._method == "Standard":
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0)
            self._std[self._std == 0] = 1.0  # avoid /0
        elif self._method == "LogTransform":
            self._min = X.min(axis=0)  # for shift if negatives
        else:
            raise ValueError(f"Unknown scaler method: {self._method}")

        self._fitted = True
        logger.info("Scaler: fitted method=%s on shape=%s", self._method, X.shape)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted scaling to *X*."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted – call fit() first.")
        X = np.asarray(X, dtype=np.float32)

        if self._method == "MinMax":
            denom = self._max - self._min
            denom[denom == 0] = 1.0
            return (X - self._min) / denom

        if self._method == "Standard":
            return (X - self._mean) / self._std

        if self._method == "LogTransform":
            shift = np.where(self._min < 0, -self._min + 1.0, 0.0)
            return np.log1p(X + shift)

        raise ValueError(f"Unknown method: {self._method}")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Reverse the scaling (for true-vs-predicted plots)."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted.")
        X_s = np.asarray(X_scaled, dtype=np.float32)

        if self._method == "MinMax":
            denom = self._max - self._min
            denom[denom == 0] = 1.0
            return X_s * denom + self._min

        if self._method == "Standard":
            return X_s * self._std + self._mean

        if self._method == "LogTransform":
            shift = np.where(self._min < 0, -self._min + 1.0, 0.0)
            return np.expm1(X_s) - shift

        raise ValueError(f"Unknown method: {self._method}")

    # ── pipeline interface ───────────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Pipeline node entry point.

        Receives ``X`` (and optionally ``y``) from upstream, fits on X,
        transforms both, and passes everything downstream.
        """
        X: np.ndarray = inputs["X"]
        y = inputs.get("y")

        X_scaled = self.fit_transform(X)

        outputs: dict[str, Any] = {**inputs, "X": X_scaled, "scaler": self}

        # Scale holdout features with the same fitted transform (no re-fitting)
        if "X_holdout" in inputs:
            X_ho = np.asarray(inputs["X_holdout"], dtype=np.float32)
            outputs["X_holdout"] = self.transform(X_ho)

        # Scale labels too (needed for inverse_transform in validator)
        if y is not None:
            y_arr = np.asarray(y, dtype=np.float32)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            y_scaler = Scaler({"method": self._method})
            y_scaled = y_scaler.fit_transform(y_arr)
            if y_scaled.shape[1] == 1:
                y_scaled = y_scaled.ravel()
            outputs["y"] = y_scaled
            outputs["y_scaler"] = y_scaler

            # Scale holdout labels with the same y_scaler
            if "y_holdout" in inputs:
                y_ho = np.asarray(inputs["y_holdout"], dtype=np.float32)
                if y_ho.ndim == 1:
                    y_ho = y_ho.reshape(-1, 1)
                y_ho_scaled = y_scaler.transform(y_ho)
                if y_ho_scaled.shape[1] == 1:
                    y_ho_scaled = y_ho_scaled.ravel()
                outputs["y_holdout"] = y_ho_scaled

        logger.info("Scaler: transformed X=%s method=%s", X_scaled.shape, self._method)
        return outputs

