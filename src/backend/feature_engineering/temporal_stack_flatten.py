"""
Temporal Stack Flatten
======================
Feature engineering node that reshapes a temporal velocity field
from (T_in, N, 3) to (N, T_in * 3) by concatenating the time steps
along the feature dimension.

This produces a flat per-point feature vector encoding the full
velocity history, suitable for point-wise predictors or as input
to graph neural networks.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TemporalStackFlatten:
    """
    Flatten a temporal field into a per-point feature vector.

    Expected ``hyperparams``:
        - flatten_order : str – ``"time_major"`` (only supported mode)
        - source_field  : str – input key to read from (default: ``"velocity_in"``)
        - output_field  : str – output key to write to (default: ``"velocity_history_features"``)
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = hyperparams or {}
        self._flatten_order: str = str(hp.get("flatten_order", "time_major"))
        self._source_field: str = str(hp.get("source_field", "velocity_in"))
        self._output_field: str = str(hp.get("output_field", "velocity_history_features"))
        self._seed = seed

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Flatten ``velocity_in`` from (T_in, N, 3) to (N, T_in * 3).

        In multi-sample mode, processes each sample's velocity_in.
        """
        samples = inputs.get("samples")
        if samples is not None and isinstance(samples, list) and len(samples) > 0:
            return self._execute_multi(inputs)
        return self._execute_single(inputs)

    def _execute_single(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Flatten a single temporal field array."""
        field_in = np.asarray(inputs[self._source_field], dtype=np.float32)

        if field_in.ndim != 3 or field_in.shape[2] != 3:
            raise ValueError(
                f"{self._source_field} must have shape (T_in, N, 3), got {field_in.shape}"
            )

        T_in, N, _ = field_in.shape
        flat = self._flatten(field_in)

        result: dict[str, Any] = {**inputs}
        result[self._output_field] = flat

        logger.info(
            "TemporalStackFlatten[%s→%s]: (T_in=%d, N=%d, 3) → (N=%d, %d)",
            self._source_field, self._output_field, T_in, N, N, T_in * 3,
        )
        return result

    def _execute_multi(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Flatten source_field for each sample."""
        samples = inputs["samples"]

        # Detect lazy mode: samples carry filepath but not the source field
        lazy = "filepath" in samples[0] and self._source_field not in samples[0]

        if lazy:
            # Pass-through: register self as an FE pipeline step
            if "fe_pipeline" not in inputs:
                inputs["fe_pipeline"] = []
            inputs["fe_pipeline"].append(self)
            logger.info(
                "TemporalStackFlatten[lazy, %s→%s]: registered for %d samples",
                self._source_field, self._output_field, len(samples),
            )
            return {**inputs}

        # Eager mode (small datasets / unit tests)
        for sample in samples:
            field_in = np.asarray(sample[self._source_field], dtype=np.float32)
            sample[self._output_field] = self._flatten(field_in)
        T_in = samples[0][self._source_field].shape[0] if samples else 0
        logger.info(
            "TemporalStackFlatten[multi, %s→%s]: %d samples, T_in=%d",
            self._source_field, self._output_field, len(samples), T_in,
        )
        return {**inputs}

    # ── lazy per-sample processing (called by GFF) ───────────────────────

    def process_sample(self, sample_data: dict[str, Any]) -> None:
        """Flatten source_field → output_field in-place."""
        field_in = np.asarray(sample_data[self._source_field], dtype=np.float32)
        sample_data[self._output_field] = self._flatten(field_in)

    def _flatten(self, velocity_in: np.ndarray) -> np.ndarray:
        """(T_in, N, 3) → (N, T_in*3)."""
        if self._flatten_order != "time_major":
            raise ValueError(f"Unknown flatten_order: {self._flatten_order}")
        T_in, N, _ = velocity_in.shape
        return velocity_in.transpose(1, 0, 2).reshape(N, T_in * 3).astype(np.float32)
