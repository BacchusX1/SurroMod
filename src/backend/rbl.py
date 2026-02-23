"""
Residual-Based Learning (RBL) Nodes
====================================
Two non-parametric data-flow nodes that set up and aggregate residual
learning within a regressor branch.

``RBLNode``
    Intercepts the output of an upstream regressor (z) and optional
    representation inputs (h_i).  Passes downstream:
      - ``X = [z, h_1, h_2, …]`` — concatenated features
      - ``y = y_original - z``    — residual target
    Also stores ``z_original`` and ``y_original`` for the aggregator.

``RBLAggregatorNode``
    Reads the residual prediction *r* from an upstream regressor and the
    stored ``z_original``.  Produces the final prediction:
      ``ŷ = z + r``
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RBLNode:
    """
    Residual-Based Learning node — sets up residual target and features.

    The node expects:
      - ``X`` from an upstream regressor whose predictions serve as *z*.
      - ``y`` (original labels) from the data flow.
      - optionally, additional representation arrays via ``representations``
        (populated by the pipeline executor from top-handle edges).

    Configuration (from node data):
      ``lambda_kernel``   – weight on the kernel loss ``MSE(z, y)``
      ``lambda_residual`` – weight on the residual regulariser ``mean(r²)``
    """

    def __init__(self, node_data: dict[str, Any]) -> None:
        self._data = node_data
        self.lambda_kernel: float = float(node_data.get("lambda_kernel", 1.0))
        self.lambda_residual: float = float(node_data.get("lambda_residual", 0.01))

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Transform the data flow for residual learning.

        Parameters
        ----------
        inputs : merged upstream dict containing at least ``X`` and ``y``.
            ``X`` is treated as the primary prediction *z*.
            ``representations`` (list[np.ndarray], optional) are extra
            feature arrays from the top-handle edges.
        """
        z = np.asarray(inputs["X"], dtype=np.float32)
        y = np.asarray(inputs["y"], dtype=np.float32)

        if z.ndim == 1:
            z = z.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Residual target
        y_residual = y - z

        # Concatenate features: [z, h_1, h_2, …]
        feature_parts: list[np.ndarray] = [z]
        rep_list: list[np.ndarray] = inputs.get("representations", [])
        for h in rep_list:
            h_arr = np.asarray(h, dtype=np.float32)
            if h_arr.ndim == 1:
                h_arr = h_arr.reshape(-1, 1)
            feature_parts.append(h_arr)

        X_out = np.hstack(feature_parts) if len(feature_parts) > 1 else z

        logger.info(
            "RBLNode: z %s + %d repr → X %s  |  y_residual %s",
            z.shape, len(rep_list), X_out.shape, y_residual.shape,
        )

        n_features = X_out.shape[1] if X_out.ndim > 1 else 1
        return {
            **inputs,
            "X": X_out,
            "y": y_residual,
            "z_original": z,
            "y_original": y,
            "lambda_kernel": self.lambda_kernel,
            "lambda_residual": self.lambda_residual,
            "feature_names": [f"rbl_{i}" for i in range(n_features)],
        }


class RBLAggregatorNode:
    """
    Aggregator for Residual-Based Learning — computes ``ŷ = z + r``.

    Reads:
      - ``X`` — the residual prediction *r* from the upstream regressor.
      - ``z_original`` — the primary prediction from the RBL node
        (carried through the data-flow dict).
      - ``y_original`` — the original labels (also carried through).

    Produces ``ŷ = z + r`` and computes R², RMSE, MAE.
    """

    def __init__(self, node_data: dict[str, Any]) -> None:
        self._data = node_data

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        r = np.asarray(inputs["X"], dtype=np.float32)
        z = np.asarray(inputs["z_original"], dtype=np.float32)
        y = np.asarray(inputs["y_original"], dtype=np.float32)

        if r.ndim == 1:
            r = r.reshape(-1, 1)
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        y_hat = z + r

        # Compute aggregated metrics
        y_flat = y.ravel()
        yh_flat = y_hat.ravel()

        ss_res = float(np.sum((y_flat - yh_flat) ** 2))
        ss_tot = float(np.sum((y_flat - y_flat.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rmse = float(np.sqrt(np.mean((y_flat - yh_flat) ** 2)))
        mae = float(np.mean(np.abs(y_flat - yh_flat)))

        metrics = {"r2": round(r2, 6), "rmse": round(rmse, 6), "mae": round(mae, 6)}

        logger.info(
            "RBLAggregatorNode: ŷ = z + r  →  R²=%.4f  RMSE=%.4f",
            metrics["r2"], metrics["rmse"],
        )

        return {
            **inputs,
            "X": y_hat,
            "y": y,
            "y_hat": y_hat,
            # Pre-computed prediction for downstream validators — they
            # must NOT call model.predict() because the RBL chain cannot
            # be replayed through a single model.predict() call.
            "y_pred": y_hat,
            "metrics": metrics,
            # Carry the model from the upstream regressor for validators
            "model": inputs.get("model"),
        }
