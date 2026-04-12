"""
Flow Forecast Validator
=======================
Compare predicted velocity fields against ground-truth targets and
produce a comprehensive metric table.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.backend.predictors.regressors.flow_loss import (
    compute_all_metrics,
)

logger = logging.getLogger(__name__)


class FlowForecastValidator:
    """
    Validate flow-field predictions against ground truth.

    Inputs
    ------
    predicted_velocity_out : (T_out, N, 3)
    velocity_out           : (T_out, N, 3)

    Outputs
    -------
    metric_table    : dict  – all computed metrics
    validator_meta  : dict  – summary metadata
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self._hp = hyperparams or {}

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # ── Multi-sample mode (GFF pipeline) ─────────────────────────────
        if "samples" in inputs and "metrics" in inputs:
            return self._execute_multi(inputs)

        # ── Single-sample mode ───────────────────────────────────────────
        pred = np.asarray(inputs["predicted_velocity_out"], dtype=np.float32)
        target = np.asarray(inputs["velocity_out"], dtype=np.float32)

        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: predicted {pred.shape} vs target {target.shape}"
            )

        metrics = compute_all_metrics(pred, target)

        result: dict[str, Any] = {**inputs}
        result["metric_table"] = metrics
        result["validator_meta"] = {
            "shape": list(pred.shape),
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "relative_l2": metrics["relative_l2"],
        }

        logger.info(
            "FlowForecastValidator: MSE=%.6f RMSE=%.6f MAE=%.6f relL2=%.4f",
            metrics["mse"], metrics["rmse"], metrics["mae"], metrics["relative_l2"],
        )
        return result

    def _execute_multi(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Validate multi-sample GFF outputs using pre-computed streaming metrics."""
        gff_metrics = inputs.get("metrics", {})
        test_indices = np.asarray(inputs.get("test_indices", []), dtype=np.int64)

        result: dict[str, Any] = {**inputs}
        result["metric_table"] = gff_metrics
        result["validator_meta"] = {
            "r2_train": gff_metrics.get("r2_train"),
            "r2_test": gff_metrics.get("r2_test"),
            "r2_overall": gff_metrics.get("r2_overall"),
            "num_test_samples": len(test_indices),
        }

        logger.info(
            "FlowForecastValidator[multi]: R2(train)=%s R2(test)=%s R2(overall)=%s",
            gff_metrics.get("r2_train", "N/A"),
            gff_metrics.get("r2_test", "N/A"),
            gff_metrics.get("r2_overall", "N/A"),
        )
        return result
