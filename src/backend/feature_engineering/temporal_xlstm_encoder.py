"""
Temporal xLSTM Encoder
======================
Feature engineering carrier node for the mLSTM-based per-node temporal
encoder.

This node does **not** process data itself — the actual mLSTM is an
``nn.Module`` inside ``GraphFlowForecaster`` and is trained end-to-end
with the GNN.  This node's role is:

1. To appear on the canvas so the pipeline architecture is visible.
2. To carry the encoder hyperparameters (head_dim, num_layers,
   output_dim, include_pressure) in the ``fe_pipeline`` list so that
   ``GraphFlowForecaster`` can discover and configure the mLSTM.
3. To tell ``GraphFlowForecaster`` to load ``velocity_in`` and
   ``pressure`` as raw arrays instead of relying on
   ``PointFeatureFusion`` for velocity history.

Ports
-----
Input:  ``velocity_in``   (T_in, N, 3)
        ``pressure_in``   (T_in, N, 1)   — optional, from InputNode
Output: ``temporal_features``  — symbolic; actual tensor produced inside GFF
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TemporalXLSTMEncoder:
    """
    Lazy carrier node for the per-node mLSTM temporal encoder.

    Expected ``hyperparams``:
        head_dim        : int  – mLSTM matrix memory dimension H  (default 16)
                                 Memory cost per sample: N × H × H × 4 bytes
                                 H=16 → ~102 MB for N=100K  (RTX 4060: fine)
        num_layers      : int  – stacked mLSTM cells                (default 2)
        output_dim      : int  – projected output per node           (default 64)
        include_pressure: bool – concatenate pressure to vel_in      (default True)
    """

    # Sentinel so GFF can detect this class without importing it
    IS_TEMPORAL_XLSTM_ENCODER: bool = True

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = hyperparams or {}
        self._head_dim: int = int(hp.get("head_dim", 16))
        self._num_layers: int = int(hp.get("num_layers", 2))
        self._output_dim: int = int(hp.get("output_dim", 64))
        self._include_pressure: bool = bool(hp.get("include_pressure", True))
        self._seed = seed

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Register self in ``fe_pipeline`` so GFF discovers our HPs.
        Does not modify sample data.
        """
        samples = inputs.get("samples")
        if samples is not None and isinstance(samples, list) and len(samples) > 0:
            return self._execute_multi(inputs)
        # Single-sample (legacy) path: nothing to do, GFF handles encoding
        logger.info(
            "TemporalXLSTMEncoder[single]: registered (head_dim=%d, layers=%d, out=%d, pressure=%s)",
            self._head_dim, self._num_layers, self._output_dim, self._include_pressure,
        )
        return {**inputs}

    def _execute_multi(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Register self in fe_pipeline — GFF discovers HPs from there."""
        if "fe_pipeline" not in inputs:
            inputs["fe_pipeline"] = []
        inputs["fe_pipeline"].append(self)
        logger.info(
            "TemporalXLSTMEncoder[lazy]: registered for %d samples "
            "(head_dim=%d, layers=%d, out=%d, pressure=%s)",
            len(inputs["samples"]),
            self._head_dim, self._num_layers, self._output_dim, self._include_pressure,
        )
        return {**inputs}

    # ── lazy per-sample processing ───────────────────────────────────────

    def process_sample(self, sample_data: dict[str, Any]) -> None:
        """No-op: raw velocity_in / pressure are loaded by GFF's _prepare_sample."""
        pass
