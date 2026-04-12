"""
Flow Metrics Summary
====================
Compute and persist interpretable scalar metrics for velocity-field
predictions. Optionally saves results to CSV and JSON.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from src.backend.predictors.regressors.flow_loss import compute_all_metrics

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "proj_dir" / "outputs"


def _ensure_output_dir() -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


class FlowMetricsSummary:
    """
    Produce scalar metrics and save them to disk.

    Inputs
    ------
    predicted_velocity_out : (T_out, N, 3)
    velocity_out           : (T_out, N, 3)

    Outputs
    -------
    metrics_summary   : dict
    metrics_file_path : str  — path to saved JSON file
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = hyperparams or {}
        self._save_csv: bool = bool(hp.get("save_csv", True))
        self._save_json: bool = bool(hp.get("save_json", True))

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pred = np.asarray(inputs["predicted_velocity_out"], dtype=np.float32)
        target = np.asarray(inputs["velocity_out"], dtype=np.float32)

        metrics = compute_all_metrics(pred, target)

        out_dir = _ensure_output_dir()
        uid = uuid.uuid4().hex[:8]
        file_path = ""

        if self._save_json:
            json_path = out_dir / f"flow_metrics_{uid}.json"
            with open(json_path, "w") as f:
                json.dump(metrics, f, indent=2, default=_json_default)
            file_path = str(json_path)
            logger.info("FlowMetricsSummary: saved JSON → %s", json_path)

        if self._save_csv:
            csv_path = out_dir / f"flow_metrics_{uid}.csv"
            self._write_csv(metrics, csv_path)
            logger.info("FlowMetricsSummary: saved CSV → %s", csv_path)

        result: dict[str, Any] = {**inputs}
        result["metrics_summary"] = metrics
        result["metrics_file_path"] = file_path

        logger.info(
            "FlowMetricsSummary: MSE=%.6f RMSE=%.6f MAE=%.6f",
            metrics["mse"], metrics["rmse"], metrics["mae"],
        )
        return result

    @staticmethod
    def _write_csv(metrics: dict, path: Path) -> None:
        lines = ["metric,value"]
        for key in ("mse", "rmse", "mae", "relative_l2", "max_abs_error"):
            if key in metrics:
                lines.append(f"{key},{metrics[key]}")
        # Per-component
        per_comp = metrics.get("per_component", {})
        for comp, vals in per_comp.items():
            for mk, mv in vals.items():
                lines.append(f"{comp}_{mk},{mv}")
        path.write_text("\n".join(lines) + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
