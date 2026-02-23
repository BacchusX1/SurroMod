"""
Train-Test Splitter
===================
Splits data into a **training** set and a **holdout** set that passes
through the pipeline untouched until a validator node consumes it.

The holdout set is stored under ``X_holdout`` / ``y_holdout`` keys and
is **never** used for model fitting, scaling, or feature engineering.
Downstream FE nodes (Scaler, PCA, Autoencoder, …) must:

  1. Fit only on ``X`` (train).
  2. Apply the same fitted transform to ``X_holdout`` if present.

This guarantees that the validator evaluates on truly unseen data with
no information leakage from preprocessing.

Hyperparameters:
    holdout_ratio  – fraction of data reserved for validation  (default 0.2)
    shuffle        – whether to shuffle before splitting       (default True)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TrainTestSplitter:
    """
    Feature-engineering node that splits data into train and holdout sets.

    Expected ``hyperparams``:
        - holdout_ratio : float – fraction to reserve (0, 1)
        - shuffle       : bool  – shuffle before splitting
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = hyperparams or {}
        self._holdout_ratio: float = float(hp.get("holdout_ratio", 0.2))
        self._shuffle: bool = bool(hp.get("shuffle", True))
        self._seed = seed

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Split upstream ``X`` and ``y`` into train and holdout sets.

        Outputs
        -------
        X, y             – training portion  (used for model fitting)
        X_holdout, y_holdout – holdout portion (passed through to validator)
        """
        X = np.asarray(inputs["X"], dtype=np.float32)
        y = np.asarray(inputs.get("y"), dtype=np.float32) if inputs.get("y") is not None else None

        n = X.shape[0]
        n_holdout = max(1, int(round(n * self._holdout_ratio)))
        n_train = n - n_holdout

        indices = np.arange(n)
        if self._shuffle:
            rng = np.random.RandomState(self._seed)
            rng.shuffle(indices)

        train_idx = indices[:n_train]
        holdout_idx = indices[n_train:]

        X_train = X[train_idx]
        X_holdout = X[holdout_idx]

        outputs: dict[str, Any] = {
            **inputs,
            "X": X_train,
            "X_holdout": X_holdout,
        }

        if y is not None:
            y_train = y[train_idx]
            y_holdout = y[holdout_idx]
            outputs["y"] = y_train
            outputs["y_holdout"] = y_holdout

        # Split feature_names (they describe columns, not rows — pass through)
        # Split label_names likewise — pass through unchanged.

        logger.info(
            "TrainTestSplitter: %d samples → %d train + %d holdout (%.0f%%)",
            n, n_train, n_holdout, self._holdout_ratio * 100,
        )

        return outputs
