"""
Flow Model Inference
====================
Run a trained GraphFlowForecaster on a specified data subset (e.g. test split).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FlowModelInference:
    """
    Run inference with a trained flow forecasting model.

    Inputs
    ------
    model_artifact          : GraphFlowForecaster instance
    point_features          : (N, D)
    pos                     : (N, 3)
    edge_index              : (2, E)
    edge_attr               : (E, edge_dim) optional
    velocity_in             : (T_in, N, 3)

    Outputs
    -------
    predicted_velocity_out  : (T_out, N, 3)
    inference_meta          : dict
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self._hp = hyperparams or {}

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        model = inputs.get("model_artifact")
        if model is None:
            raise ValueError(
                "FlowModelInference: no 'model_artifact' in inputs. "
                "Connect a trained GraphFlowForecaster upstream."
            )

        pred_result = model.predict(inputs)

        result: dict[str, Any] = {**inputs}
        result["predicted_velocity_out"] = pred_result["predicted_velocity_out"]
        result["inference_meta"] = {
            "shape": list(pred_result["predicted_velocity_out"].shape),
            "model_type": "GraphFlowForecaster",
        }

        logger.info(
            "FlowModelInference: predicted shape %s",
            pred_result["predicted_velocity_out"].shape,
        )
        return result
