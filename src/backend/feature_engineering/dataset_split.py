"""
Dataset Split
=============
Unified feature-engineering node that partitions data into
train / validation / test subsets.

Supports two ``data_kind`` modes:

*  **scalar** (default) — splits ``X`` and ``y`` arrays row-wise,
   producing ``X_holdout`` / ``y_holdout`` for a held-out test set.
   Downstream FE nodes must fit only on ``X`` (train) then apply
   the same transform to ``X_holdout`` (no information leakage).

*  **3d_field** — splits by sample IDs (``sample_ids``, ``num_samples``,
   or inferred from ``pos``) and emits ``train_ids``, ``val_ids``,
   ``test_ids`` index arrays for use by graph-based forecasters.

Both modes support random (optionally shuffled) and sequential splits.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DatasetSplit:
    """
    Unified dataset splitting node for scalar and 3D-field pipelines.

    Hyperparams
    -----------
    data_kind    : ``"scalar"`` | ``"3d_field"``  (default ``"scalar"``)
    split_mode   : ``"random"`` | ``"sequential"``
    train_ratio  : float (default 0.8)
    val_ratio    : float (default 0.0 for scalar, 0.0 for 3d_field)
    test_ratio   : float (default 0.2)
    random_seed  : int   (default 42)
    shuffle      : bool  (default True)
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = hyperparams or {}
        self._data_kind: str = str(hp.get("data_kind", "scalar"))
        self._split_mode: str = str(hp.get("split_mode", "random"))
        self._train_ratio: float = float(hp.get("train_ratio", 0.8))
        self._val_ratio: float = float(hp.get("val_ratio", 0.0))
        self._test_ratio: float = float(hp.get("test_ratio", 0.2))
        self._random_seed: int = int(hp.get("random_seed", seed or 42))
        self._shuffle: bool = bool(hp.get("shuffle", True))
        self._seed = seed

    # ── public API ───────────────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._data_kind == "3d_field":
            return self._split_3d_field(inputs)
        return self._split_scalar(inputs)

    # ── scalar mode (replaces TrainTestSplitter) ─────────────────────────

    def _split_scalar(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Split ``X`` / ``y`` row-wise into train and holdout sets.

        Outputs
        -------
        X, y                 – training portion (used for model fitting)
        X_holdout, y_holdout – holdout portion (passed through to validator)
        """
        X = np.asarray(inputs["X"], dtype=np.float32)
        y = (
            np.asarray(inputs["y"], dtype=np.float32)
            if inputs.get("y") is not None
            else None
        )

        n = X.shape[0]
        train_idx, val_idx, test_idx = self._compute_split_indices(n)

        # For scalar mode: train = train+val, holdout = test
        if len(val_idx) > 0:
            train_idx = np.concatenate([train_idx, val_idx])

        X_train = X[train_idx]
        X_holdout = X[test_idx]

        result: dict[str, Any] = {
            **inputs,
            "X": X_train,
            "X_holdout": X_holdout,
        }

        if y is not None:
            result["y"] = y[train_idx]
            result["y_holdout"] = y[test_idx]

        logger.info(
            "DatasetSplit[scalar]: %d samples → %d train + %d holdout (%.0f%%)",
            n, len(train_idx), len(test_idx), self._test_ratio * 100,
        )
        return result

    # ── 3d_field mode (replaces old DatasetSplit) ────────────────────────

    def _split_3d_field(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Split by **simulation ID**, not by spatial points.

        When the upstream data contains a ``samples`` list (batch mode),
        unique ``sim_id`` values are extracted, shuffled, and split into
        train / val / test groups.  All temporal windows belonging to the
        same simulation go into the same split.

        Emits ``train_indices`` and ``test_indices`` — arrays of indices
        into the ``samples`` list.
        """
        samples = inputs.get("samples")
        if samples is not None and isinstance(samples, list) and len(samples) > 0:
            return self._split_by_simulation(inputs, samples)

        # Legacy fallback: split by point indices
        return self._split_by_points(inputs)

    def _split_by_simulation(
        self, inputs: dict[str, Any], samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Split sample windows grouped by sim_id."""
        from collections import defaultdict

        # Group window indices by sim_id
        sim_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            sim_to_indices[s["sim_id"]].append(i)

        unique_sims = sorted(sim_to_indices.keys())
        n_sims = len(unique_sims)
        if n_sims < 2:
            raise ValueError(
                f"DatasetSplit[3d_field]: need ≥2 simulations for splitting, "
                f"got {n_sims}."
            )

        # Split simulation IDs
        train_idx, val_idx, test_idx = self._compute_split_indices(n_sims)

        train_sims = [unique_sims[i] for i in train_idx]
        val_sims = [unique_sims[i] for i in val_idx]
        test_sims = [unique_sims[i] for i in test_idx]

        # Map back to window indices
        train_indices = sorted(
            j for sid in train_sims for j in sim_to_indices[sid]
        )
        val_indices = sorted(
            j for sid in val_sims for j in sim_to_indices[sid]
        )
        test_indices = sorted(
            j for sid in test_sims for j in sim_to_indices[sid]
        )

        result: dict[str, Any] = {**inputs}
        result["train_indices"] = np.array(train_indices, dtype=np.int64)
        result["val_indices"] = np.array(val_indices, dtype=np.int64)
        result["test_indices"] = np.array(test_indices, dtype=np.int64)
        result["train_sim_ids"] = train_sims
        result["test_sim_ids"] = test_sims
        result["split_meta"] = {
            "num_simulations": n_sims,
            "num_train_sims": len(train_sims),
            "num_val_sims": len(val_sims),
            "num_test_sims": len(test_sims),
            "num_train_windows": len(train_indices),
            "num_test_windows": len(test_indices),
            "split_mode": self._split_mode,
            "random_seed": self._random_seed,
        }

        logger.info(
            "DatasetSplit[3d_field]: %d sims → train=%d (%d win) "
            "val=%d test=%d (%d win)",
            n_sims,
            len(train_sims), len(train_indices),
            len(val_sims),
            len(test_sims), len(test_indices),
        )
        return result

    def _split_by_points(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Legacy fallback: split point indices (single-sample mode)."""
        if "sample_ids" in inputs:
            ids = np.asarray(inputs["sample_ids"])
        elif "num_samples" in inputs:
            ids = np.arange(int(inputs["num_samples"]))
        elif "pos" in inputs:
            ids = np.arange(inputs["pos"].shape[0])
        else:
            raise ValueError(
                "DatasetSplit[3d_field]: provide 'samples', 'sample_ids', "
                "'num_samples', or 'pos'."
            )

        n = len(ids)
        if n == 0:
            raise ValueError("DatasetSplit[3d_field]: empty sample set.")

        train_idx, val_idx, test_idx = self._compute_split_indices(n)

        result: dict[str, Any] = {**inputs}
        result["train_ids"] = ids[train_idx]
        result["val_ids"] = ids[val_idx]
        result["test_ids"] = ids[test_idx]
        result["split_meta"] = {
            "num_samples": int(n),
            "num_train": len(train_idx),
            "num_val": len(val_idx),
            "num_test": len(test_idx),
            "split_mode": self._split_mode,
            "random_seed": self._random_seed,
        }

        logger.info(
            "DatasetSplit[3d_field]: N=%d → train=%d val=%d test=%d (mode=%s)",
            n, len(train_idx), len(val_idx), len(test_idx), self._split_mode,
        )
        return result

    # ── shared helpers ───────────────────────────────────────────────────

    def _compute_split_indices(
        self, n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (train_idx, val_idx, test_idx) index arrays."""
        total = self._train_ratio + self._val_ratio + self._test_ratio
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "DatasetSplit: ratios sum to %.3f — normalising.", total,
            )
        tr = self._train_ratio / total
        vr = self._val_ratio / total

        n_train = max(1, int(round(n * tr)))
        n_val = max(0, int(round(n * vr)))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_val += n_test
            n_test = 0

        if self._split_mode == "random":
            rng = np.random.RandomState(self._random_seed)
            order = np.arange(n)
            if self._shuffle:
                rng.shuffle(order)
        elif self._split_mode == "sequential":
            order = np.arange(n)
        else:
            raise ValueError(f"Unknown split_mode: {self._split_mode}")

        train_idx = order[:n_train]
        val_idx = order[n_train : n_train + n_val]
        test_idx = order[n_train + n_val :]

        return train_idx, val_idx, test_idx
