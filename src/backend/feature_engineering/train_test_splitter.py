"""
Train-Test Splitter  (backward-compatible wrapper)
===================================================
Delegates to :class:`DatasetSplit` with ``data_kind="scalar"``.

Existing workflows that reference ``TrainTestSplitter`` will continue to
work without modification.  New workflows should use ``DatasetSplit``
directly and set ``data_kind`` in the hyperparams.
"""

from __future__ import annotations

from typing import Any

from src.backend.feature_engineering.dataset_split import DatasetSplit


class TrainTestSplitter:
    """Thin wrapper around DatasetSplit for backward compatibility."""

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = dict(hyperparams or {})
        # Map legacy hyperparam names to unified names
        hp.setdefault("data_kind", "scalar")
        if "holdout_ratio" in hp:
            hp.setdefault("test_ratio", hp.pop("holdout_ratio"))
            hp.setdefault("train_ratio", 1.0 - hp["test_ratio"])
        hp.setdefault("shuffle", True)
        self._delegate = DatasetSplit(hyperparams=hp, seed=seed)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return self._delegate.execute(inputs)
