"""
Regressor Validator
===================
Evaluates one **or many** trained regression models: computes metrics
(R², RMSE, MAE, …), generates per-model true-vs-predicted scatter plots,
and – when multiple models are connected – produces comparison bar charts.

All plots are encoded as base64 PNGs so the frontend can render them.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Consistent colour palette for multi-model plots ─────────────────────────

_MODEL_COLOURS = [
    "#6366f1",  # indigo
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#ef4444",  # red
    "#3b82f6",  # blue
    "#8b5cf6",  # violet
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#f97316",  # orange
    "#6d28d9",  # purple
]


def _fig_to_base64(fig: Any) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    return b64


def _colour_for(index: int) -> str:
    return _MODEL_COLOURS[index % len(_MODEL_COLOURS)]


class RegressorValidator:
    """
    Pipeline node that evaluates one or many trained regressors.

    When a single model is connected the output is unchanged from before.
    When **multiple** models are connected the output gains:
      • ``multi_model: True``
      • ``model_results``: per-model metrics + per-label scatter plots
      • ``comparison_bar_plot``: grouped-bar comparison of all metrics
    """

    def __init__(self, node_data: dict[str, Any] | None = None) -> None:
        self._results: dict[str, Any] = {}
        self._plots_per_row: int = 4
        if node_data:
            self._plots_per_row = int(node_data.get("plotsPerRow", 4))

    # ── metrics ──────────────────────────────────────────────────────────────

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        y_t = y_true.ravel().astype(np.float64)
        y_p = y_pred.ravel().astype(np.float64)

        ss_res = float(np.sum((y_t - y_p) ** 2))
        ss_tot = float(np.sum((y_t - y_t.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
        mae = float(np.mean(np.abs(y_t - y_p)))
        max_err = float(np.max(np.abs(y_t - y_p)))

        return {
            "r2": round(r2, 6),
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "max_error": round(max_err, 6),
        }

    # ── single-model scatter plot ────────────────────────────────────────────

    @staticmethod
    def true_vs_predicted_plot(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_name: str = "y",
        model_name: str | None = None,
        colour: str = "#6366f1",
    ) -> str:
        """Return a base64-encoded PNG of a true-vs-predicted scatter plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        ax.scatter(y_true, y_pred, alpha=0.5, s=14, edgecolors="none", c=colour)

        lo = min(float(y_true.min()), float(y_pred.min()))
        hi = max(float(y_true.max()), float(y_pred.max()))
        margin = (hi - lo) * 0.05
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            "k--", linewidth=0.8, alpha=0.6,
        )
        ax.set_xlabel(f"True ({label_name})")
        ax.set_ylabel(f"Predicted ({label_name})")
        title = f"True vs Predicted – {label_name}"
        if model_name:
            title = f"{model_name} – {label_name}"
        ax.set_title(title)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()

        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64

    # ── comparison bar chart ─────────────────────────────────────────────────

    @staticmethod
    def comparison_bar_plot(
        model_names: list[str],
        all_metrics: list[dict[str, float]],
    ) -> str:
        """
        Grouped bar chart with one subplot per metric.

        Returns a base64-encoded PNG.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metric_keys = list(all_metrics[0].keys())
        n_metrics = len(metric_keys)
        n_models = len(model_names)

        cols = min(n_metrics, 4)
        rows = math.ceil(n_metrics / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.2 * rows))
        if n_metrics == 1:
            axes = np.array([axes])
        axes_flat = np.asarray(axes).ravel()

        colours = [_colour_for(i) for i in range(n_models)]
        x = np.arange(n_models)

        for idx, key in enumerate(metric_keys):
            ax = axes_flat[idx]
            vals = [m[key] for m in all_metrics]
            bars = ax.bar(x, vals, color=colours, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
            ax.set_title(key.upper().replace("_", " "), fontsize=10, fontweight="bold")
            ax.set_ylabel(key.upper())
            # Add value annotations above bars
            for bar, v in zip(bars, vals):
                ax.annotate(
                    f"{v:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7,
                )
            ax.grid(axis="y", alpha=0.3)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.tight_layout()
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64

    # ── helpers ──────────────────────────────────────────────────────────────

    def _evaluate_single_model(
        self,
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        label_names: list[str],
        y_scaler: Any | None,
        model_name: str | None = None,
        colour: str = "#6366f1",
    ) -> dict[str, Any]:
        """Run prediction + compute metrics + generate per-label plots for one model."""
        y_pred = np.asarray(model.predict(X), dtype=np.float32)

        # Inverse-transform if scaler was applied
        yt = y_true.copy()
        yp = y_pred.copy()
        if y_scaler is not None:
            if yt.ndim == 1:
                yt = y_scaler.inverse_transform(yt.reshape(-1, 1)).ravel()
                yp = y_scaler.inverse_transform(yp.reshape(-1, 1)).ravel()
            else:
                yt = y_scaler.inverse_transform(yt)
                yp = y_scaler.inverse_transform(yp)

        # Ensure 2-D for per-label iteration
        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)
        if yp.ndim == 1:
            yp = yp.reshape(-1, 1)

        overall = self.compute_metrics(yt, yp)

        per_label: list[dict[str, Any]] = []
        for i, name in enumerate(label_names):
            yti = yt[:, i] if yt.shape[1] > i else yt.ravel()
            ypi = yp[:, i] if yp.shape[1] > i else yp.ravel()
            m = self.compute_metrics(yti, ypi)
            plot_b64 = self.true_vs_predicted_plot(
                yti, ypi,
                label_name=name,
                model_name=model_name,
                colour=colour,
            )
            per_label.append({"label": name, "metrics": m, "plot": plot_b64})

        return {"metrics": overall, "per_label": per_label}

    # ── pipeline interface ───────────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Validate one or many trained models.

        Single-model inputs (backward compatible)
        ------------------------------------------
        Keys from the upstream regressor's output dict:

            model       – a trained ModelBase instance
            X           – feature array (in scaled space if a scaler was used)
            y           – label array   (in scaled space if a scaler was used)
            label_names – list[str] of label column names
            y_scaler    – (optional) fitted Scaler for inverse-transforming y

        Multi-model inputs (per-model context)
        --------------------------------------
        ``models`` is a list of dicts, each carrying the **complete
        context from that model's graph path**:

            models = [
                {
                    "name": "MLP",
                    "model": <trained MLP>,
                    "X": <scaled X for this model>,
                    "y": <scaled y for this model>,
                    "y_scaler": <fitted Scaler specific to this model's path>,
                    "label_names": [...],
                    "feature_names": [...]
                },
                ...
            ]

        This ensures every model is evaluated with its own preprocessing
        context, even when different scalers or feature engineering steps
        were applied on different graph branches.
        """
        # ── Determine single-model vs multi-model ────────────────────────
        models_list: list[dict[str, Any]] | None = inputs.get("models")

        if models_list is None and "model" in inputs:
            # ── Single model – backward-compatible path ──────────────────
            X = np.asarray(inputs["X"], dtype=np.float32)
            y_true = np.asarray(inputs["y"], dtype=np.float32)
            label_names: list[str] = inputs.get("label_names") or ["y"]
            y_scaler = inputs.get("y_scaler")

            result = self._evaluate_single_model(
                model=inputs["model"],
                X=X, y_true=y_true,
                label_names=label_names,
                y_scaler=y_scaler,
            )
            self._results = result
            logger.info(
                "RegressorValidator (single): %s  overall=%s",
                ", ".join(label_names), result["metrics"],
            )
            return self._results

        # ── Multi-model path ─────────────────────────────────────────────
        if not models_list:
            raise ValueError("RegressorValidator: no 'model' or 'models' in inputs.")

        model_results: list[dict[str, Any]] = []
        all_overall_metrics: list[dict[str, float]] = []
        model_names: list[str] = []

        for idx, entry in enumerate(models_list):
            name = entry["name"]
            model = entry["model"]
            colour = _colour_for(idx)

            # Each model carries its OWN context from its graph path
            X = np.asarray(entry["X"], dtype=np.float32)
            y_true = np.asarray(entry["y"], dtype=np.float32)
            y_scaler = entry.get("y_scaler")
            label_names_for_model: list[str] = entry.get("label_names") or ["y"]

            logger.info(
                "RegressorValidator: evaluating '%s' with y_scaler=%s",
                name,
                type(y_scaler).__name__ if y_scaler is not None else "None",
            )

            res = self._evaluate_single_model(
                model=model, X=X, y_true=y_true,
                label_names=label_names_for_model, y_scaler=y_scaler,
                model_name=name, colour=colour,
            )
            model_results.append({
                "model_name": name,
                "metrics": res["metrics"],
                "per_label": res["per_label"],
            })
            all_overall_metrics.append(res["metrics"])
            model_names.append(name)

        # Generate comparison bar chart
        bar_plot_b64 = self.comparison_bar_plot(model_names, all_overall_metrics)

        self._results = {
            "multi_model": True,
            "plots_per_row": self._plots_per_row,
            "model_results": model_results,
            "comparison_bar_plot": bar_plot_b64,
        }

        logger.info(
            "RegressorValidator (multi): %d models  [%s]",
            len(model_names), ", ".join(model_names),
        )
        return self._results
