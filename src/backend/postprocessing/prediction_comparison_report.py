"""
Prediction Comparison Report
=============================
Aggregate training history, metrics, and slice plots into a compact
HTML evaluation report.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "proj_dir" / "outputs"


def _ensure_output_dir() -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


class PredictionComparisonReport:
    """
    Aggregate postprocessing outputs into one HTML report.

    Inputs (all optional — whatever the upstream provides)
    ------
    metric_table      : dict
    slice_plot_paths  : list[str]
    training_history  : list[dict]
    inference_meta    : dict
    best_epoch_meta   : dict
    split_meta        : dict
    latent_meta       : dict

    Outputs
    -------
    report_meta : dict
    report_path : str — path to generated HTML file
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self._hp = hyperparams or {}

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        out_dir = _ensure_output_dir()
        uid = uuid.uuid4().hex[:8]
        report_path = out_dir / f"comparison_report_{uid}.html"

        html = self._build_html(inputs)
        report_path.write_text(html, encoding="utf-8")

        result: dict[str, Any] = {**inputs}
        result["report_path"] = str(report_path)
        result["report_meta"] = {
            "report_file": str(report_path),
            "has_metrics": "metric_table" in inputs,
            "has_plots": bool(inputs.get("slice_plot_paths")),
            "has_training_history": bool(inputs.get("training_history")),
        }

        logger.info("PredictionComparisonReport: saved → %s", report_path)
        return result

    def _build_html(self, inputs: dict[str, Any]) -> str:
        sections: list[str] = []

        # Header
        sections.append("<h1>SurroMod — Flow Forecast Comparison Report</h1>")

        # Model settings
        latent_meta = inputs.get("latent_meta", {})
        if latent_meta:
            sections.append("<h2>Model Configuration</h2>")
            sections.append("<table><tr><th>Parameter</th><th>Value</th></tr>")
            for k, v in latent_meta.items():
                sections.append(f"<tr><td>{_esc(k)}</td><td>{_esc(str(v))}</td></tr>")
            sections.append("</table>")

        # Split info
        split_meta = inputs.get("split_meta", {})
        if split_meta:
            sections.append("<h2>Dataset Split</h2>")
            sections.append("<table><tr><th>Parameter</th><th>Value</th></tr>")
            for k, v in split_meta.items():
                sections.append(f"<tr><td>{_esc(k)}</td><td>{_esc(str(v))}</td></tr>")
            sections.append("</table>")

        # Scalar metrics
        mt = inputs.get("metric_table", {})
        if mt:
            sections.append("<h2>Evaluation Metrics</h2>")
            sections.append("<table><tr><th>Metric</th><th>Value</th></tr>")
            for key in ("mse", "rmse", "mae", "relative_l2", "max_abs_error"):
                if key in mt:
                    sections.append(
                        f"<tr><td>{_esc(key)}</td><td>{mt[key]:.6f}</td></tr>"
                    )
            sections.append("</table>")

            # Per-component
            per_comp = mt.get("per_component", {})
            if per_comp:
                sections.append("<h3>Per-Component Metrics</h3>")
                sections.append(
                    "<table><tr><th>Component</th><th>MSE</th><th>RMSE</th><th>MAE</th></tr>"
                )
                for comp, vals in per_comp.items():
                    sections.append(
                        f"<tr><td>{_esc(comp)}</td>"
                        f"<td>{vals.get('mse', 0):.6f}</td>"
                        f"<td>{vals.get('rmse', 0):.6f}</td>"
                        f"<td>{vals.get('mae', 0):.6f}</td></tr>"
                    )
                sections.append("</table>")

        # Training curves
        history = inputs.get("training_history", [])
        if history:
            sections.append("<h2>Training History</h2>")
            sections.append(
                "<table><tr><th>Epoch</th><th>Loss</th><th>MAE</th></tr>"
            )
            # Show first 5, last 5 if long
            display = history
            if len(history) > 20:
                display = history[:5] + [{"epoch": "...", "train_loss": "...", "train_mae": "..."}] + history[-5:]
            for entry in display:
                ep = entry.get("epoch", "")
                loss = entry.get("train_loss", "")
                mae = entry.get("train_mae", "")
                loss_s = f"{loss:.6f}" if isinstance(loss, (int, float)) else str(loss)
                mae_s = f"{mae:.6f}" if isinstance(mae, (int, float)) else str(mae)
                sections.append(
                    f"<tr><td>{ep}</td><td>{loss_s}</td><td>{mae_s}</td></tr>"
                )
            sections.append("</table>")

        # Best epoch meta
        best = inputs.get("best_epoch_meta", {})
        if best:
            sections.append("<h3>Best Epoch</h3>")
            sections.append(f"<p>Best loss: {best.get('best_loss', 'N/A'):.6f}, "
                            f"Total epochs: {best.get('total_epochs', 'N/A')}</p>")

        # Slice plots
        plot_paths = inputs.get("slice_plot_paths", [])
        if plot_paths:
            sections.append("<h2>Slice Visualisations</h2>")
            for p in plot_paths:
                fname = Path(p).name
                sections.append(
                    f'<p><img src="{_esc(fname)}" style="max-width:100%;" '
                    f'alt="{_esc(fname)}"></p>'
                )

        body = "\n".join(sections)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SurroMod Comparison Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2em auto; color: #1e293b; }}
  h1 {{ color: #3b82f6; }}
  h2 {{ color: #475569; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.3em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 6px 12px; text-align: left; }}
  th {{ background: #f1f5f9; }}
  img {{ border: 1px solid #e2e8f0; border-radius: 4px; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _esc(s: str) -> str:
    """Basic HTML escaping."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
