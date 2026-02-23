"""
KRR (Kernel Ridge Regression)
=============================
PyTorch-based differentiable Kernel Ridge Regression surrogate model.

The kernel system ``(K + αI) w = y`` is solved via ``torch.linalg.solve``
every forward pass, making the entire operation differentiable so that
upstream feature extractors (e.g. MLP) can receive gradients through KRR.

Hyperparameters:
    kernel, alpha, gamma, degree, coef0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.backend.predictors.model_base import ModelBase

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# PyTorch module
# ═════════════════════════════════════════════════════════════════════════════

class _KRRNetwork(nn.Module):
    """
    Differentiable Kernel Ridge Regression.

    *   ``forward(X, y)`` — fit the kernel system on ``(X, y)`` and return
        in-sample predictions.  Gradients flow through ``torch.linalg.solve``.
    *   ``forward(X)`` — predict using the stored training data and
        coefficients (inference mode).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 0.1,
        kernel: str = "rbf",
        degree: int = 3,
        coef0: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha_reg = alpha
        self.gamma = gamma
        self.kernel_type = kernel
        self.degree = degree
        self.coef0 = coef0

        # Stored after the last fit-forward (detached, for inference)
        self._X_train: torch.Tensor | None = None
        self._weights: torch.Tensor | None = None

    # ── Kernel functions ─────────────────────────────────────────────────

    def _kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == "rbf":
            dist_sq = torch.cdist(X1, X2).pow(2)
            return torch.exp(-self.gamma * dist_sq)
        if self.kernel_type == "linear":
            return X1 @ X2.T
        if self.kernel_type == "poly":
            return (self.gamma * (X1 @ X2.T) + self.coef0).pow(self.degree)
        if self.kernel_type == "sigmoid":
            return torch.tanh(self.gamma * (X1 @ X2.T) + self.coef0)
        raise ValueError(f"Unsupported kernel: {self.kernel_type}")

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        X : (N, D) input features.
        y : (N, T) targets.  When provided, solve the kernel system
            (training / differentiable forward).  When ``None``, predict
            with stored data (inference).
        """
        if y is not None:
            # ── Training: solve and predict ──────────────────────────────
            K = self._kernel(X, X)
            n = K.size(0)
            K_reg = K + self.alpha_reg * torch.eye(n, device=K.device, dtype=K.dtype)

            if y.ndim == 1:
                y = y.unsqueeze(-1)

            weights = torch.linalg.solve(K_reg, y)
            preds = K @ weights

            # Store detached copies for inference
            self._X_train = X.detach().clone()
            self._weights = weights.detach().clone()

            return preds

        # ── Inference: use stored data ───────────────────────────────────
        if self._X_train is None or self._weights is None:
            raise RuntimeError(
                "KRR has not been fitted.  Call forward(X, y) first."
            )
        K = self._kernel(X, self._X_train)
        return K @ self._weights


# ═════════════════════════════════════════════════════════════════════════════
# ModelBase wrapper
# ═════════════════════════════════════════════════════════════════════════════

class KRRRegressor(ModelBase):
    """
    Differentiable Kernel Ridge Regression surrogate model.

    Wraps :class:`_KRRNetwork` behind the unified :class:`ModelBase`
    lifecycle.  Fully differentiable — gradients flow through the kernel
    solve so KRR can participate in end-to-end branch training.

    Automatically uses CUDA when a GPU is available.
    """

    model_name = "KRR"
    model_category = "regressor"
    model_type = "KRR"
    is_differentiable = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        """Instantiate the ``_KRRNetwork`` from current hyperparams."""
        kernel = str(self.get_hyperparam("kernel", "rbf"))
        alpha = float(self.get_hyperparam("alpha", 1.0))
        gamma = float(self.get_hyperparam("gamma", 0.1))
        degree = int(self.get_hyperparam("degree", 3))
        coef0 = float(self.get_hyperparam("coef0", 1.0))

        self._model = _KRRNetwork(
            alpha=alpha,
            gamma=gamma,
            kernel=kernel,
            degree=degree,
            coef0=coef0,
        ).to(self._device)

        logger.info(
            "%s: built  kernel=%s  alpha=%.4g  gamma=%.4g  degree=%d  device=%s",
            self.model_name, kernel, alpha, gamma, degree, self._device,
        )

    # ── Lifecycle: train ─────────────────────────────────────────────────

    def train(self, X: Any, y: Any, **kwargs: Any) -> None:
        """
        Fit the KRR by solving the kernel system (single forward pass).

        KRR does not require iterative optimisation — the kernel system is
        solved exactly.  Calling ``train`` performs one ``forward(X, y)``
        which stores the training data and coefficients.
        """
        if self._model is None:
            self.build()

        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self._device)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float32), device=self._device)

        self._model.eval()
        with torch.no_grad():
            self._model(X_t, y_t)
        self._is_trained = True

        logger.info("%s: training complete  X=%s  device=%s", self.model_name, X_t.shape, self._device)

    # ── Lifecycle: predict ───────────────────────────────────────────────

    def predict(self, X: Any) -> np.ndarray:
        """Return predictions as a numpy array."""
        self._require_trained("predict")
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self._device)

        self._model.eval()
        with torch.no_grad():
            preds = self._model(X_t)
        return preds.cpu().numpy().astype(np.float32)

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

    # ── Torch module access ──────────────────────────────────────────────

    def get_torch_module(self) -> nn.Module:
        """Return the ``_KRRNetwork`` for use in composed pipelines."""
        if self._model is None:
            raise RuntimeError("Call build() before get_torch_module().")
        return self._model

    # ── Serialisation ────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save trained KRR to a directory (state-dict + meta)."""
        self._require_trained("save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self._model.state_dict(), path / "model.pt")
        # Also save stored training data for inference
        extra = {
            "X_train": self._model._X_train.cpu().numpy().tolist()
            if self._model._X_train is not None else None,
            "weights": self._model._weights.cpu().numpy().tolist()
            if self._model._weights is not None else None,
        }
        meta = {
            "class": self.__class__.__qualname__,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "hyperparams": self._hyperparams,
            "metadata": self._metadata,
            "extra": extra,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        logger.info("%s: saved → %s", self.model_name, path)

    @classmethod
    def load(cls, path: str | Path) -> "KRRRegressor":
        """Load a previously saved KRR from *path*."""
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())

        instance = cls(hyperparams=meta.get("hyperparams"))
        instance._metadata = meta.get("metadata", {})
        instance.build()

        state = torch.load(path / "model.pt", map_location=instance._device, weights_only=True)
        instance._model.load_state_dict(state)

        extra = meta.get("extra", {})
        if extra.get("X_train") is not None:
            instance._model._X_train = torch.tensor(
                extra["X_train"], dtype=torch.float32, device=instance._device,
            )
        if extra.get("weights") is not None:
            instance._model._weights = torch.tensor(
                extra["weights"], dtype=torch.float32, device=instance._device,
            )

        instance._is_trained = True
        logger.info("%s: loaded ← %s  device=%s", cls.model_name, path, instance._device)
        return instance
