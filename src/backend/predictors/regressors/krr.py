"""
KRR (Kernel Ridge Regression)
=============================
Scikit-learn Kernel Ridge Regression surrogate model.

Hyperparameters:
    kernel, alpha, gamma, degree, coef0
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.backend.predictors.model_base import ModelBase

logger = logging.getLogger(__name__)


class KRRRegressor(ModelBase):
    """
    Kernel Ridge Regression for surrogate modelling.

    Wraps ``sklearn.kernel_ridge.KernelRidge`` behind the unified
    :class:`ModelBase` lifecycle.
    """

    model_name = "KRR"
    model_category = "regressor"
    model_type = "KRR"

    # ── Hyperparameter schema ────────────────────────────────────────────

    @classmethod
    def default_hyperparams(cls) -> dict[str, Any]:
        return {
            "kernel": "rbf",
            "alpha": 1.0,
            "gamma": 0.1,
            "degree": 3,
            "coef0": 1.0,
        }

    # ── Lifecycle: build ─────────────────────────────────────────────────

    def build(self) -> None:
        """Instantiate the ``KernelRidge`` estimator from current hyperparams."""
        try:
            from sklearn.kernel_ridge import KernelRidge
        except ImportError as exc:
            raise RuntimeError(
                "KRR requires scikit-learn.  Install with `pip install scikit-learn`."
            ) from exc

        kernel = str(self.get_hyperparam("kernel", "rbf"))
        alpha = float(self.get_hyperparam("alpha", 1.0))
        gamma = float(self.get_hyperparam("gamma", 0.1))
        degree = int(self.get_hyperparam("degree", 3))
        coef0 = float(self.get_hyperparam("coef0", 1.0))

        self._model = KernelRidge(
            kernel=kernel,
            alpha=alpha,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
        )

        logger.info(
            "%s: built  kernel=%s  alpha=%.4g  gamma=%.4g  degree=%d",
            self.model_name, kernel, alpha, gamma, degree,
        )

    # ── Lifecycle: train ─────────────────────────────────────────────────

    def train(self, X: Any, y: Any, **kwargs: Any) -> None:
        """Fit the KernelRidge estimator."""
        if self._model is None:
            self.build()

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        self._model.fit(X_arr, y_arr)
        self._is_trained = True

        logger.info("%s: training complete  X=%s", self.model_name, X_arr.shape)

    # ── Lifecycle: predict ───────────────────────────────────────────────

    def predict(self, X: Any) -> np.ndarray:
        """Return predictions as a numpy array."""
        self._require_trained("predict")
        X_arr = np.asarray(X, dtype=np.float64)
        preds = self._model.predict(X_arr)
        return np.asarray(preds, dtype=np.float32)

    # ── Lifecycle: score ─────────────────────────────────────────────────

    def score(self, X: Any, y: Any) -> dict[str, float]:
        """Compute R², RMSE, MAE."""
        self._require_trained("score")
        preds = self.predict(X).ravel()
        y_arr = np.asarray(y, dtype=np.float32).ravel()

        ss_res = float(np.sum((y_arr - preds) ** 2))
        ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rmse = float(np.sqrt(np.mean((y_arr - preds) ** 2)))
        mae = float(np.mean(np.abs(y_arr - preds)))

        return {"r2": round(r2, 6), "rmse": round(rmse, 6), "mae": round(mae, 6)}

    # ── Serialisation ────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save trained KRR to a directory (joblib + meta)."""
        self._require_trained("save")
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._model, path / "model.joblib")

        meta = {
            "class": self.__class__.__qualname__,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "hyperparams": self._hyperparams,
            "metadata": self._metadata,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        logger.info("%s: saved → %s", self.model_name, path)

    @classmethod
    def load(cls, path: str | Path) -> "KRRRegressor":
        """Load a previously saved KRR from *path*."""
        import joblib

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())

        instance = cls(hyperparams=meta.get("hyperparams"))
        instance._metadata = meta.get("metadata", {})
        instance._model = joblib.load(path / "model.joblib")
        instance._is_trained = True

        logger.info("%s: loaded ← %s", cls.model_name, path)
        return instance
