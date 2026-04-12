"""
Graph Flow Forecaster
=====================
Graph-based surrogate regressor for predicting future velocity fields
from past velocity fields and geometry-aware features on unstructured
point clouds.

Architecture
------------
1. **Baseline extrapolation** — persistence or linear extrapolation
2. **Point embedding** — MLP mapping point_features to latent space
3. **Graph message passing** — local neighbourhood processing (pure PyTorch)
4. **Temporal LSTM decoder** — per-point recurrent decoder over future steps
5. **Residual velocity head** — MLP producing delta velocity

Final prediction = baseline_future + learned_delta
"""

from __future__ import annotations

import copy
import json
import logging
import math
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Torch imports (deferred so the module can be imported without torch)
# ──────────────────────────────────────────────────────────────────────────────

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "GraphFlowForecaster requires PyTorch. "
            "Install it with: pip install torch"
        )


def _get_device() -> "torch.device":
    _require_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# Baseline Extrapolators
# ══════════════════════════════════════════════════════════════════════════════


class BaselineExtrapolator:
    """
    Compute a naive baseline future velocity prediction.

    Modes
    -----
    persistence
        Repeat the last input time step for all output steps.
    linear_extrapolation
        Extrapolate linearly from the last two input steps.
    mean_field
        Use the temporal mean of velocity_in (average over T_in) as a
        constant prediction for every future step.
    polynomial
        Fit a degree-2 polynomial per point and component over T_in
        and evaluate at future time steps.
    exponential_smoothing
        Single exponential smoothing (Holt's method) extrapolated into
        the future.  Smoothing factor ``alpha`` is fixed at 0.3.
    """

    _ALPHA_ES: float = 0.3  # exponential-smoothing factor

    def __init__(self, mode: str = "persistence") -> None:
        self.mode = mode

    def __call__(
        self, velocity_in: np.ndarray, T_out: int,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        velocity_in : (T_in, N, 3)
        T_out       : number of future time steps

        Returns
        -------
        baseline : (T_out, N, 3)
        """
        if self.mode == "persistence":
            last = velocity_in[-1:]  # (1, N, 3)
            return np.tile(last, (T_out, 1, 1))

        if self.mode == "linear_extrapolation":
            if velocity_in.shape[0] < 2:
                return np.tile(velocity_in[-1:], (T_out, 1, 1))
            slope = velocity_in[-1] - velocity_in[-2]  # (N, 3)
            out = np.empty((T_out, *velocity_in.shape[1:]), dtype=velocity_in.dtype)
            for t in range(T_out):
                out[t] = velocity_in[-1] + slope * (t + 1)
            return out

        if self.mode == "mean_field":
            mean = velocity_in.mean(axis=0, keepdims=True)  # (1, N, 3)
            return np.tile(mean, (T_out, 1, 1))

        if self.mode == "polynomial":
            T_in = velocity_in.shape[0]
            t_in = np.arange(T_in, dtype=np.float64)
            t_out = np.arange(T_in, T_in + T_out, dtype=np.float64)
            N, C = velocity_in.shape[1], velocity_in.shape[2]
            out = np.empty((T_out, N, C), dtype=velocity_in.dtype)
            for n in range(N):
                for c in range(C):
                    coeffs = np.polyfit(t_in, velocity_in[:, n, c].astype(np.float64), 2)
                    out[:, n, c] = np.polyval(coeffs, t_out).astype(velocity_in.dtype)
            return out

        if self.mode == "none":
            # Direct prediction: model outputs absolute velocity, no residual subtraction
            return np.zeros((T_out, *velocity_in.shape[1:]), dtype=velocity_in.dtype)

        if self.mode == "exponential_smoothing":
            alpha = self._ALPHA_ES
            # Compute smoothed level and trend from velocity_in
            T_in = velocity_in.shape[0]
            level = velocity_in[0].copy().astype(np.float64)
            trend = (velocity_in[1] - velocity_in[0]).astype(np.float64) if T_in > 1 else np.zeros_like(level)
            for t in range(1, T_in):
                prev_level = level.copy()
                level = alpha * velocity_in[t] + (1 - alpha) * (level + trend)
                trend = alpha * (level - prev_level) + (1 - alpha) * trend
            out = np.empty((T_out, *velocity_in.shape[1:]), dtype=velocity_in.dtype)
            l, tr = level.copy(), trend.copy()
            for t in range(T_out):
                forecast = l + (t + 1) * tr
                out[t] = forecast.astype(velocity_in.dtype)
            return out

        raise ValueError(f"Unknown baseline mode: {self.mode}")


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Modules
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE:

    class PointEncoder(nn.Module):
        """MLP mapping per-point features to latent space."""

        def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int, dropout: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class GraphMessagePassingBlock(nn.Module):
        """
        Single round of local message passing on the graph.

        For each edge (i → j):
            message_ij = MLP([h_i || h_j || edge_attr])
        Then aggregate messages per destination node using one of:
            mean      – scatter-mean  (default, original behaviour)
            max       – scatter-max
            attention – scalar gate per edge, softmax-normalised per
                        destination node, then weighted sum

        Update:
            h_i' = LayerNorm(h_i + MLP(h_i || agg_messages_i))
        """

        def __init__(
            self,
            latent_dim: int,
            hidden_dim: int,
            edge_dim: int = 0,
            dropout: float = 0.0,
            aggregation_mode: str = "mean",
        ):
            super().__init__()
            self.aggregation_mode = aggregation_mode
            msg_in = 2 * latent_dim + edge_dim
            self.message_mlp = nn.Sequential(
                nn.Linear(msg_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.update_mlp = nn.Sequential(
                nn.Linear(2 * latent_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.norm = nn.LayerNorm(latent_dim)
            # Attention gate: scalar weight per edge from message vector
            if aggregation_mode == "attention":
                self.attn_gate = nn.Linear(latent_dim, 1, bias=False)

        def forward(
            self,
            h: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """
            Parameters
            ----------
            h          : (N, latent_dim)
            edge_index : (2, E) long tensor
            edge_attr  : (E, edge_dim) or None

            Returns
            -------
            h_updated  : (N, latent_dim)
            """
            src, dst = edge_index  # src → dst
            h_src = h[src]  # (E, latent_dim)
            h_dst = h[dst]  # (E, latent_dim)

            if edge_attr is not None:
                msg_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
            else:
                msg_input = torch.cat([h_src, h_dst], dim=-1)

            messages = self.message_mlp(msg_input)  # (E, latent_dim)
            N = h.size(0)

            if self.aggregation_mode == "mean":
                agg = torch.zeros(N, messages.size(1), device=h.device, dtype=h.dtype)
                count = torch.zeros(N, 1, device=h.device, dtype=h.dtype)
                agg.index_add_(0, dst, messages)
                count.index_add_(0, dst, torch.ones(dst.size(0), 1, device=h.device, dtype=h.dtype))
                agg = agg / count.clamp(min=1.0)

            elif self.aggregation_mode == "max":
                agg = torch.full((N, messages.size(1)), float("-inf"), device=h.device, dtype=h.dtype)
                agg.index_reduce_(0, dst, messages, reduce="amax", include_self=True)
                # Replace -inf nodes (no incoming edges) with zeros
                agg = torch.where(agg == float("-inf"), torch.zeros_like(agg), agg)

            elif self.aggregation_mode == "attention":
                # Compute per-edge scalar logit from the message
                logits = self.attn_gate(messages).squeeze(-1)  # (E,)
                # Softmax per destination node
                # Numerically stable: subtract per-dst max before exp
                logit_max = torch.full((N,), float("-inf"), device=h.device, dtype=h.dtype)
                logit_max.index_reduce_(0, dst, logits, "amax", include_self=True)
                logit_max = logit_max.clamp(min=-1e9)
                shifted = logits - logit_max[dst]
                exp_logits = shifted.exp()
                sum_exp = torch.zeros(N, device=h.device, dtype=h.dtype)
                sum_exp.index_add_(0, dst, exp_logits)
                alpha = exp_logits / (sum_exp[dst] + 1e-9)  # (E,)
                weighted_msgs = messages * alpha.unsqueeze(-1)  # (E, latent_dim)
                agg = torch.zeros(N, messages.size(1), device=h.device, dtype=h.dtype)
                agg.index_add_(0, dst, weighted_msgs)

            else:
                raise ValueError(f"Unknown aggregation_mode: {self.aggregation_mode}")

            # Residual update
            update = self.update_mlp(torch.cat([h, agg], dim=-1))
            h_out = self.norm(h + update)
            return h_out

    class TemporalLSTMDecoder(nn.Module):
        """
        Per-point LSTM decoder that unrolls over T_out future steps.

        Takes the latent per-point state and produces a sequence of
        latent states for each output time step.
        """

        def __init__(self, latent_dim: int, hidden_dim: int, T_out: int):
            super().__init__()
            self.T_out = T_out
            self.lstm = nn.LSTMCell(latent_dim, hidden_dim)
            self.init_h = nn.Linear(latent_dim, hidden_dim)
            self.init_c = nn.Linear(latent_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, latent_dim)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            z : (N, latent_dim)

            Returns
            -------
            out : (T_out, N, latent_dim)
            """
            h = self.init_h(z)  # (N, hidden_dim)
            c = self.init_c(z)  # (N, hidden_dim)
            inp = z             # (N, latent_dim)

            outputs = []
            for _ in range(self.T_out):
                h, c = self.lstm(inp, (h, c))
                out_t = self.out_proj(h)  # (N, latent_dim)
                outputs.append(out_t)
                inp = out_t  # auto-regressive feeding
            return torch.stack(outputs, dim=0)  # (T_out, N, latent_dim)

    class DirectBroadcastDecoder(nn.Module):
        """
        Simple non-recurrent decoder: broadcast the latent state
        to all T_out output steps independently (no temporal autoregression).

        Faster and lower-capacity than TemporalLSTMDecoder — useful when
        the task is quasi-steady and temporal ordering is unimportant.
        """

        def __init__(self, latent_dim: int, T_out: int):
            super().__init__()
            self.T_out = T_out
            self.latent_dim = latent_dim

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            """
            z : (N, latent_dim)  →  (T_out, N, latent_dim)
            """
            return z.unsqueeze(0).expand(self.T_out, -1, -1)

    class ResidualVelocityHead(nn.Module):
        """MLP mapping latent features to delta velocity (3-D)."""

        def __init__(self, latent_dim: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            """z: (T_out, N, latent_dim) → (T_out, N, 3)"""
            return self.net(z)

    class GraphFlowForecasterModel(nn.Module):
        """
        Full end-to-end model combining:
        - PointEncoder
        - GraphMessagePassingBlock stack  (with configurable aggregation)
        - Optional skip connections       (none / dense / initial)
        - TemporalLSTMDecoder
        - ResidualVelocityHead

        Skip-connection modes
        ---------------------
        none    – plain sequential message passing (original behaviour)
        dense   – DenseNet-style: input to layer L = concat of all
                  previous outputs; a projection MLP re-maps to latent_dim
        initial – JKNet initial residual: h_L = LayerNorm(h_0 + h_L)
        """

        def __init__(
            self,
            point_feature_dim: int,
            latent_dim: int = 128,
            hidden_dim: int = 256,
            edge_dim: int = 0,
            num_mp_layers: int = 3,
            T_out: int = 5,
            dropout: float = 0.0,
            use_edge_attr: bool = True,
            aggregation_mode: str = "mean",
            skip_connection_mode: str = "none",
            use_temporal_decoder: bool = True,
        ):
            super().__init__()
            self.skip_connection_mode = skip_connection_mode
            self.num_mp_layers = num_mp_layers
            self.use_temporal_decoder = use_temporal_decoder
            act_edge_dim = edge_dim if use_edge_attr else 0

            self.encoder = PointEncoder(point_feature_dim, latent_dim, hidden_dim, dropout)

            # For dense skip: each layer receives growing concatenation
            self.mp_layers = nn.ModuleList()
            for layer_idx in range(num_mp_layers):
                if skip_connection_mode == "dense" and layer_idx > 0:
                    # input to this layer = (layer_idx + 1) * latent_dim
                    # project down to latent_dim before passing to MP block
                    in_channels = (layer_idx + 1) * latent_dim
                    self.mp_layers.append(
                        nn.ModuleDict({
                            "proj": nn.Linear(in_channels, latent_dim, bias=False),
                            "mp": GraphMessagePassingBlock(
                                latent_dim, hidden_dim,
                                edge_dim=act_edge_dim,
                                dropout=dropout,
                                aggregation_mode=aggregation_mode,
                            ),
                        })
                    )
                else:
                    self.mp_layers.append(
                        GraphMessagePassingBlock(
                            latent_dim, hidden_dim,
                            edge_dim=act_edge_dim,
                            dropout=dropout,
                            aggregation_mode=aggregation_mode,
                        )
                    )

            if skip_connection_mode == "initial":
                self.skip_norm = nn.LayerNorm(latent_dim)

            if use_temporal_decoder:
                self.decoder = TemporalLSTMDecoder(latent_dim, hidden_dim, T_out)
            else:
                self.decoder = DirectBroadcastDecoder(latent_dim, T_out)
            self.head = ResidualVelocityHead(latent_dim, hidden_dim)
            self.use_edge_attr = use_edge_attr

        def forward(
            self,
            point_features: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """
            Parameters
            ----------
            point_features : (N, D)
            edge_index     : (2, E)
            edge_attr      : (E, edge_dim) or None

            Returns
            -------
            delta_velocity : (T_out, N, 3)
            """
            h = self.encoder(point_features)  # (N, latent_dim)
            ea = edge_attr if self.use_edge_attr else None

            if self.skip_connection_mode == "none":
                for mp in self.mp_layers:
                    h = mp(h, edge_index, ea)

            elif self.skip_connection_mode == "dense":
                h_initial = h
                all_h = [h]
                for layer_idx, mp_entry in enumerate(self.mp_layers):
                    if layer_idx == 0:
                        h = mp_entry(h, edge_index, ea)
                    else:
                        # Concat all previous outputs
                        h_concat = torch.cat(all_h, dim=-1)  # (N, (k+1)*latent_dim)
                        h_proj = mp_entry["proj"](h_concat)   # (N, latent_dim)
                        h = mp_entry["mp"](h_proj, edge_index, ea)
                    all_h.append(h)

            elif self.skip_connection_mode == "initial":
                h_initial = h
                for mp in self.mp_layers:
                    h = mp(h, edge_index, ea)
                # Add initial residual and normalise
                h = self.skip_norm(h + h_initial)

            z_seq = self.decoder(h)  # (T_out, N, latent_dim)
            delta = self.head(z_seq)  # (T_out, N, 3)
            return delta


# ══════════════════════════════════════════════════════════════════════════════
# Hierarchical architecture (mLSTM + U-Net GNN)
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE:

    class mLSTMCell(nn.Module):
        """
        Single mLSTM cell (matrix-memory LSTM from xLSTM, Hochreiter 2024).

        Unlike a standard LSTM whose cell state is a vector, the mLSTM
        stores a *matrix* C ∈ R^{H×H} that can represent multiple
        simultaneous "facts" about the input sequence.  This is especially
        useful for short sequences with rich cross-feature interactions
        (e.g. vx/vy/vz/pressure evolving over 5 timesteps).

        Stabilisation uses the max-trick on log-gates so that the
        forget/input gates never overflow.
        """

        def __init__(self, input_dim: int, head_dim: int) -> None:
            super().__init__()
            H = head_dim
            self.head_dim = H
            self.W_q = nn.Linear(input_dim, H, bias=True)
            self.W_k = nn.Linear(input_dim, H, bias=True)
            self.W_v = nn.Linear(input_dim, H, bias=True)
            self.w_i = nn.Linear(input_dim, 1, bias=True)   # log input gate
            self.w_f = nn.Linear(input_dim, 1, bias=True)   # log forget gate
            self.W_o = nn.Linear(input_dim, H, bias=True)   # output gate

        def forward(
            self,
            x: "torch.Tensor",   # (N, input_dim)
            C: "torch.Tensor",   # (N, H, H) matrix memory
            n: "torch.Tensor",   # (N, H)    normaliser
            m: "torch.Tensor",   # (N,)      max-stabiliser
        ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
            H = self.head_dim
            N = x.shape[0]

            q = self.W_q(x)               # (N, H)
            k = self.W_k(x) / (H ** 0.5) # (N, H)  — key scaling
            v = self.W_v(x)               # (N, H)
            i_log = self.w_i(x).squeeze(-1)  # (N,)
            f_log = self.w_f(x).squeeze(-1)  # (N,)
            o = torch.sigmoid(self.W_o(x))   # (N, H)

            # Stabilised gates: m_t = max(f + m_{t-1}, i)
            m_new = torch.maximum(f_log + m, i_log)        # (N,)
            i_gate = torch.exp(i_log - m_new)              # (N,)
            f_gate = torch.exp(f_log + m - m_new)          # (N,)

            # Matrix memory update: C_t = f·C_{t-1} + i·(v ⊗ k^T)
            C_new = (
                f_gate.view(N, 1, 1) * C
                + i_gate.view(N, 1, 1)
                * torch.bmm(v.unsqueeze(-1), k.unsqueeze(-2))
            )  # (N, H, H)

            # Normaliser: n_t = f·n_{t-1} + i·k
            n_new = f_gate.unsqueeze(-1) * n + i_gate.unsqueeze(-1) * k  # (N, H)

            # Output: h = o ⊙ (C @ q) / max(|n·q|, 1)
            Cq = torch.bmm(C_new, q.unsqueeze(-1)).squeeze(-1)  # (N, H)
            denom = torch.clamp(
                torch.abs((n_new * q).sum(dim=-1, keepdim=True)), min=1.0
            )  # (N, 1)
            h = o * Cq / denom  # (N, H)

            return h, C_new, n_new, m_new

    class NodeTemporalEncoder(nn.Module):
        """
        Per-node mLSTM encoder applied independently over T_in timesteps.

        Each node's velocity (+ optional pressure) sequence is encoded into
        a fixed-size embedding that captures temporal trends and cross-
        component correlations without any spatial aggregation.

        Input : (N, T_in, C_in)  — C_in = 3 (vel) or 4 (vel+pressure)
        Output: (N, output_dim)
        """

        def __init__(
            self,
            input_dim: int,
            head_dim: int,
            num_layers: int,
            output_dim: int,
        ) -> None:
            super().__init__()
            self.head_dim = head_dim
            self.num_layers = num_layers

            # Stack of mLSTM cells; first cell takes raw input_dim,
            # subsequent cells take head_dim (output of previous cell).
            self.cells = nn.ModuleList([
                mLSTMCell(input_dim if i == 0 else head_dim, head_dim)
                for i in range(num_layers)
            ])
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(head_dim) for _ in range(num_layers)
            ])

            self.out_proj = nn.Linear(head_dim, output_dim)
            self.out_norm = nn.LayerNorm(output_dim)

        def forward(self, seq: "torch.Tensor") -> "torch.Tensor":
            """
            seq : (N, T_in, C_in)
            out : (N, output_dim)
            """
            N, T, _ = seq.shape
            H = self.head_dim

            # Initialise states for each layer
            C = [seq.new_zeros(N, H, H) for _ in range(self.num_layers)]
            n = [seq.new_zeros(N, H)    for _ in range(self.num_layers)]
            m = [seq.new_full((N,), -1e9) for _ in range(self.num_layers)]

            h_out = seq[:, 0, :]  # placeholder, overwritten in loop

            for t in range(T):
                x = seq[:, t, :]      # (N, C_in) at timestep t
                for layer_idx, (cell, ln) in enumerate(
                    zip(self.cells, self.layer_norms)
                ):
                    h_new, C_new, n_new, m_new = cell(x, C[layer_idx], n[layer_idx], m[layer_idx])
                    h_new = ln(h_new)
                    C[layer_idx] = C_new
                    n[layer_idx] = n_new
                    m[layer_idx] = m_new
                    x = h_new      # feed into next layer
                h_out = x          # final hidden state after all layers

            return self.out_norm(self.out_proj(h_out))  # (N, output_dim)

    class HierarchicalGFFModel(nn.Module):
        """
        Hierarchical U-Net GNN for flow field prediction.

        Architecture
        ------------
        1. NodeTemporalEncoder  — per-node mLSTM over T_in timesteps
        2. Fusion projection    — concat(temporal_emb, geo_features) → latent
        3. Fine-scale MP layers — local neighbourhood context (full mesh)
        4. Pool                 — index-select coarse nodes from fine latents
        5. Coarse-scale MP      — global context on subsampled mesh
        6. Unpool + skip        — scatter coarse latents back + skip from fine
        7. Output head          — latent → T_out × 3 velocity
        """

        def __init__(
            self,
            temporal_input_dim: int,
            geo_feature_dim: int,
            xlstm_head_dim: int,
            xlstm_num_layers: int,
            xlstm_output_dim: int,
            latent_dim: int,
            hidden_dim: int,
            edge_dim: int,
            num_fine_mp_layers: int,
            num_coarse_mp_layers: int,
            T_out: int,
            dropout: float = 0.0,
            aggregation_mode: str = "mean",
        ) -> None:
            super().__init__()
            self.T_out = T_out

            self.temporal_encoder = NodeTemporalEncoder(
                temporal_input_dim, xlstm_head_dim, xlstm_num_layers, xlstm_output_dim,
            )

            fuse_in = xlstm_output_dim + geo_feature_dim
            self.fuse_proj = nn.Sequential(
                nn.Linear(fuse_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
            )

            self.fine_mp = nn.ModuleList([
                GraphMessagePassingBlock(latent_dim, hidden_dim, edge_dim, dropout, aggregation_mode)
                for _ in range(num_fine_mp_layers)
            ])

            self.coarse_mp = nn.ModuleList([
                GraphMessagePassingBlock(latent_dim, hidden_dim, edge_dim, dropout, aggregation_mode)
                for _ in range(num_coarse_mp_layers)
            ])

            # Skip merge: concat fine + upsampled coarse → latent
            self.upsample_proj = nn.Linear(2 * latent_dim, latent_dim, bias=False)
            self.skip_norm = nn.LayerNorm(latent_dim)

            # Output: latent → T_out * 3 (all timesteps at once)
            self.head = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, T_out * 3),
            )

        def forward(
            self,
            vel_temporal: "torch.Tensor",           # (N, T_in, C_in)
            geo_features: "torch.Tensor",           # (N, D_geo)
            edge_index_fine: "torch.Tensor",        # (2, E_f)
            edge_attr_fine: "torch.Tensor | None",
            edge_index_coarse: "torch.Tensor",      # (2, E_c)
            edge_attr_coarse: "torch.Tensor | None",
            coarse_indices: "torch.Tensor",         # (K,) long
            unpool_ftc_idx: "torch.Tensor",         # (N, k_up) long
            unpool_ftc_w: "torch.Tensor",           # (N, k_up) float
        ) -> "torch.Tensor":  # (T_out, N, 3)
            N = geo_features.shape[0]

            # 1. Temporal encoding
            t_emb = self.temporal_encoder(vel_temporal)   # (N, xlstm_output_dim)

            # 2. Fuse temporal + geometry → latent
            h = self.fuse_proj(torch.cat([t_emb, geo_features], dim=-1))  # (N, latent)

            # 3. Fine-scale message passing
            for mp in self.fine_mp:
                h = mp(h, edge_index_fine, edge_attr_fine)
            h_fine = h

            # 4. Pool to coarse level (simple index select)
            h_coarse = h_fine[coarse_indices]   # (K, latent)

            # 5. Coarse-scale message passing
            for mp in self.coarse_mp:
                h_coarse = mp(h_coarse, edge_index_coarse, edge_attr_coarse)

            # 6. Unpool: weighted scatter of coarse → fine
            #    h_coarse[unpool_ftc_idx]: (N, k_up, latent)
            h_up = (
                h_coarse[unpool_ftc_idx] * unpool_ftc_w.unsqueeze(-1)
            ).sum(dim=1)  # (N, latent)

            # 7. Skip merge + norm
            h = self.skip_norm(
                self.upsample_proj(torch.cat([h_fine, h_up], dim=-1))
            )   # (N, latent)

            # 8. Output head
            out_flat = self.head(h)                        # (N, T_out * 3)
            vel_pred = out_flat.view(N, self.T_out, 3).permute(1, 0, 2)  # (T_out, N, 3)
            return vel_pred


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline-facing Forecaster
# ══════════════════════════════════════════════════════════════════════════════


class GraphFlowForecaster:
    """
    Pipeline-executable wrapper around GraphFlowForecasterModel.

    Supports two execution modes:

    *  **single** — one sample (legacy, backward compatible).
    *  **multi** — a list of samples, each with its own mesh.

    In multi mode the forecaster internally builds per-sample graphs
    and features, trains on training-split windows, and evaluates on
    all windows.
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        _require_torch()
        hp = hyperparams or {}
        self._hp = hp
        self._seed = seed

        # ── Legacy flat-model architecture ───────────────────────────────
        self._latent_dim: int = int(hp.get("latent_dim", 128))
        self._hidden_dim: int = int(hp.get("hidden_dim", 256))
        self._num_mp_layers: int = int(hp.get("num_message_passing_layers", 3))
        self._dropout: float = float(hp.get("dropout", 0.0))
        self._use_edge_attr: bool = bool(hp.get("use_edge_attr", True))
        self._baseline_mode: str = str(hp.get("baseline_mode", "persistence"))
        self._aggregation_mode: str = str(hp.get("aggregation_mode", "mean"))
        self._skip_connection_mode: str = str(hp.get("skip_connection_mode", "none"))
        self._use_temporal_decoder: bool = bool(hp.get("use_temporal_decoder", True))
        self._feed_baseline_as_feature: bool = bool(hp.get("feed_baseline_as_feature", False))

        # ── Hierarchical U-Net GNN HPs (active when TemporalXLSTMEncoder
        #    + HierarchicalGraphBuilder are detected in fe_pipeline) ──────
        # mLSTM temporal encoder
        self._xlstm_head_dim: int = int(hp.get("xlstm_head_dim", 16))
        self._xlstm_num_layers: int = int(hp.get("xlstm_num_layers", 2))
        self._xlstm_output_dim: int = int(hp.get("xlstm_output_dim", 64))
        self._include_pressure: bool = bool(hp.get("include_pressure_in_encoder", True))
        # U-Net GNN layers
        self._num_fine_mp_layers: int = int(hp.get("num_fine_mp_layers", 2))
        self._num_coarse_mp_layers: int = int(hp.get("num_coarse_mp_layers", 3))
        # Proximity-weighted loss
        self._proximity_loss_weight: float = float(hp.get("proximity_loss_weight", 3.0))
        self._proximity_sigma: float = float(hp.get("proximity_sigma", 0.02))

        # ── Training ─────────────────────────────────────────────────────
        self._lr: float = float(hp.get("learning_rate", 1e-3))
        self._batch_size: int = int(hp.get("batch_size", 1))
        self._num_epochs: int = int(hp.get("num_epochs", 50))
        self._optimizer_name: str = str(hp.get("optimizer", "Adam"))
        self._weight_decay: float = float(hp.get("weight_decay", 1e-5))
        self._scheduler: str = str(hp.get("scheduler", "none"))
        self._patience: int = int(hp.get("early_stopping_patience", 10))
        self._physics_loss_weight: float = float(hp.get("physics_loss_weight", 0.0))

        # ── State ────────────────────────────────────────────────────────
        self._model: "GraphFlowForecasterModel | HierarchicalGFFModel | None" = None
        self._baseline = BaselineExtrapolator(self._baseline_mode)
        self._device = _get_device()
        self._training_history: list[dict[str, float]] = []
        self._best_state_dict: dict | None = None
        self._is_trained = False

        # Auto-detected at execute() time from fe_pipeline
        self._use_hierarchical: bool = False
        self._T_in: int = 5
        self._temporal_input_dim: int = 4  # vel(3) + pressure(1)

        # Velocity normaliser (used when baseline_mode="none")
        self._vel_mean: np.ndarray = np.zeros(3, dtype=np.float32)
        self._vel_std: np.ndarray = np.ones(3, dtype=np.float32)

        # Inferred dims
        self._point_feature_dim: int = 0
        self._edge_dim: int = 0
        self._T_out: int = 5

    def _detect_pipeline_components(self, fe_pipeline: list | None) -> None:
        """
        Scan fe_pipeline for TemporalXLSTMEncoder and HierarchicalGraphBuilder.
        Sets self._use_hierarchical and reads their HPs.
        """
        xlstm_node = None
        hgb_node = None
        for step in (fe_pipeline or []):
            if getattr(step, "IS_TEMPORAL_XLSTM_ENCODER", False):
                xlstm_node = step
            if getattr(step, "IS_HIERARCHICAL_GRAPH_BUILDER", False):
                hgb_node = step

        self._use_hierarchical = (xlstm_node is not None and hgb_node is not None)

        if xlstm_node is not None:
            self._xlstm_head_dim = xlstm_node._head_dim
            self._xlstm_num_layers = xlstm_node._num_layers
            self._xlstm_output_dim = xlstm_node._output_dim
            self._include_pressure = xlstm_node._include_pressure

        if self._use_hierarchical:
            logger.info(
                "GFF: hierarchical mode detected "
                "(mLSTM head_dim=%d, layers=%d, out=%d, pressure=%s)",
                self._xlstm_head_dim, self._xlstm_num_layers,
                self._xlstm_output_dim, self._include_pressure,
            )
        else:
            logger.info("GFF: flat model mode (no HGB/TXLSTM in pipeline)")

    def _build_model(self) -> None:
        """Build the PyTorch model — hierarchical or legacy flat."""
        if self._use_hierarchical:
            self._model = HierarchicalGFFModel(
                temporal_input_dim=self._temporal_input_dim,
                geo_feature_dim=self._point_feature_dim,
                xlstm_head_dim=self._xlstm_head_dim,
                xlstm_num_layers=self._xlstm_num_layers,
                xlstm_output_dim=self._xlstm_output_dim,
                latent_dim=self._latent_dim,
                hidden_dim=self._hidden_dim,
                edge_dim=self._edge_dim,
                num_fine_mp_layers=self._num_fine_mp_layers,
                num_coarse_mp_layers=self._num_coarse_mp_layers,
                T_out=self._T_out,
                dropout=self._dropout,
                aggregation_mode=self._aggregation_mode,
            ).to(self._device)
        else:
            self._model = GraphFlowForecasterModel(
                point_feature_dim=self._point_feature_dim,
                latent_dim=self._latent_dim,
                hidden_dim=self._hidden_dim,
                edge_dim=self._edge_dim if self._use_edge_attr else 0,
                num_mp_layers=self._num_mp_layers,
                T_out=self._T_out,
                dropout=self._dropout,
                use_edge_attr=self._use_edge_attr,
                aggregation_mode=self._aggregation_mode,
                skip_connection_mode=self._skip_connection_mode,
                use_temporal_decoder=self._use_temporal_decoder,
            ).to(self._device)

    def _make_optimizer(self) -> "torch.optim.Optimizer":
        if self._optimizer_name == "AdamW":
            return torch.optim.AdamW(
                self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay
            )
        if self._optimizer_name == "SGD":
            return torch.optim.SGD(
                self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay
            )
        return torch.optim.Adam(
            self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )

    def _make_scheduler(self, optimizer: "torch.optim.Optimizer") -> Any:
        if self._scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self._num_epochs
            )
        if self._scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        if self._scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5
            )
        return None

    # ══════════════════════════════════════════════════════════════════════
    # Per-sample preparation helpers
    # ══════════════════════════════════════════════════════════════════════

    def _prepare_sample(
        self,
        sample: dict[str, Any],
        fe_pipeline: list | None = None,
        graph_cache: dict | None = None,
        hgb_cache: dict | None = None,
        max_points: int = 0,
    ) -> dict[str, Any]:
        """
        Convert a sample dict into arrays ready for training/inference.

        Dispatches to hierarchical or legacy path based on
        ``self._use_hierarchical``.
        """
        if self._use_hierarchical:
            return self._prepare_sample_hierarchical(
                sample, fe_pipeline=fe_pipeline,
                hgb_cache=hgb_cache, max_points=max_points,
            )
        return self._prepare_sample_flat(
            sample, fe_pipeline=fe_pipeline,
            graph_cache=graph_cache, max_points=max_points,
        )

    def _prepare_sample_flat(
        self,
        sample: dict[str, Any],
        fe_pipeline: list | None = None,
        graph_cache: dict | None = None,
        max_points: int = 0,
    ) -> dict[str, Any]:
        """Legacy flat-model sample preparation (unchanged behaviour)."""
        if "filepath" in sample and "point_features" not in sample:
            from src.backend.data_digester.temporal_point_cloud_field_digester import (
                TemporalPointCloudFieldDigester,
            )
            arrays = TemporalPointCloudFieldDigester.load_sample(
                sample["filepath"],
                keys={"pos", "velocity_in", "velocity_out", "idcs_airfoil"},
                max_points=max_points,
            )
            data: dict[str, Any] = {**sample, **arrays}
            del arrays

            if fe_pipeline:
                for step in fe_pipeline:
                    if hasattr(step, "process_sample"):
                        if hasattr(step, "_build_knn"):
                            step.process_sample(data, graph_cache=graph_cache)
                        else:
                            step.process_sample(data)
        else:
            data = sample

        pf = np.asarray(data["point_features"], dtype=np.float32)
        edge_index = np.asarray(data["edge_index"], dtype=np.int64)
        edge_attr = data.get("edge_attr")
        if edge_attr is not None:
            edge_attr = np.asarray(edge_attr, dtype=np.float32)

        vel_in = np.asarray(data["velocity_in"], dtype=np.float32)
        vel_out = np.asarray(data["velocity_out"], dtype=np.float32)

        baseline = self._baseline(vel_in, vel_out.shape[0])
        target_delta = vel_out - baseline

        if self._feed_baseline_as_feature:
            baseline_feat = baseline[0].reshape(pf.shape[0], 3)
            pf = np.concatenate([pf, baseline_feat], axis=1)

        return {
            "point_features": pf,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "velocity_in": vel_in,
            "velocity_out": vel_out,
            "baseline": baseline,
            "target_delta": target_delta,
            "idcs_airfoil": data.get("idcs_airfoil"),
        }

    def _prepare_sample_hierarchical(
        self,
        sample: dict[str, Any],
        fe_pipeline: list | None = None,
        hgb_cache: dict | None = None,
        max_points: int = 0,
    ) -> dict[str, Any]:
        """
        Hierarchical-mode sample preparation.

        Loads raw velocity + pressure sequences alongside geometry features
        and the two-level graph.  The mLSTM temporal encoding happens
        inside the model forward pass; only numpy arrays are returned here.
        """
        if "filepath" in sample and "velocity_in" not in sample:
            from src.backend.data_digester.temporal_point_cloud_field_digester import (
                TemporalPointCloudFieldDigester,
            )
            keys = {"pos", "velocity_in", "velocity_out", "idcs_airfoil"}
            if self._include_pressure:
                keys.add("pressure")
            arrays = TemporalPointCloudFieldDigester.load_sample(
                sample["filepath"], keys=keys, max_points=max_points,
            )
            data: dict[str, Any] = {**sample, **arrays}
            del arrays

            # Run FE pipeline steps (geometry features only; skip TXLSTM/HGB)
            if fe_pipeline:
                for step in fe_pipeline:
                    if getattr(step, "IS_TEMPORAL_XLSTM_ENCODER", False):
                        continue   # no-op carrier
                    if getattr(step, "IS_HIERARCHICAL_GRAPH_BUILDER", False):
                        step.process_sample(data, graph_cache=hgb_cache)
                        continue
                    if hasattr(step, "process_sample"):
                        step.process_sample(data)
        else:
            data = sample

        vel_in = np.asarray(data["velocity_in"], dtype=np.float32)   # (T_in, N, 3)
        vel_out = np.asarray(data["velocity_out"], dtype=np.float32)  # (T_out, N, 3)
        N = vel_in.shape[1]
        self._T_in = vel_in.shape[0]

        # Build temporal input: (N, T_in, C_in)
        vel_seq = vel_in.transpose(1, 0, 2)  # (N, T_in, 3)
        if self._include_pressure and "pressure" in data:
            pressure = np.asarray(data["pressure"], dtype=np.float32)  # (T_total, N)
            T_in = vel_in.shape[0]
            pres_seq = pressure[:T_in].T[:, :, None]  # (N, T_in, 1)
            temporal_input = np.concatenate([vel_seq, pres_seq], axis=-1)  # (N, T_in, 4)
        else:
            temporal_input = vel_seq   # (N, T_in, 3)
        self._temporal_input_dim = temporal_input.shape[-1]

        # Geometry features (from PointFeatureFusion → FeatureNormalizer)
        geo_features = np.asarray(data["point_features"], dtype=np.float32)  # (N, D_geo)

        # Hierarchical graph artefacts (populated by HGB process_sample above)
        ei_fine  = np.asarray(data["edge_index_fine"],  dtype=np.int64)
        ea_fine  = np.asarray(data["edge_attr_fine"],   dtype=np.float32) if data.get("edge_attr_fine") is not None else None
        ei_coarse = np.asarray(data["edge_index_coarse"], dtype=np.int64)
        ea_coarse = np.asarray(data["edge_attr_coarse"],  dtype=np.float32) if data.get("edge_attr_coarse") is not None else None
        coarse_indices = np.asarray(data["coarse_indices"],  dtype=np.int64)
        ftc_idx = np.asarray(data["unpool_ftc_idx"], dtype=np.int64)
        ftc_w   = np.asarray(data["unpool_ftc_w"],   dtype=np.float32)
        dist_to_af = data.get("dist_to_af")
        if dist_to_af is not None:
            dist_to_af = np.asarray(dist_to_af, dtype=np.float32)

        return {
            "temporal_input": temporal_input,
            "geo_features": geo_features,
            "edge_index_fine": ei_fine,
            "edge_attr_fine": ea_fine,
            "edge_index_coarse": ei_coarse,
            "edge_attr_coarse": ea_coarse,
            "coarse_indices": coarse_indices,
            "unpool_ftc_idx": ftc_idx,
            "unpool_ftc_w": ftc_w,
            "dist_to_af": dist_to_af,
            "velocity_in": vel_in,
            "velocity_out": vel_out,
            "idcs_airfoil": data.get("idcs_airfoil"),
        }

    @staticmethod
    def _enforce_no_slip(
        prediction: np.ndarray,
        idcs_airfoil: np.ndarray | None,
    ) -> np.ndarray:
        """Zero velocity at airfoil points (no-slip BC)."""
        if idcs_airfoil is not None and len(idcs_airfoil) > 0:
            prediction[:, idcs_airfoil, :] = 0.0
        return prediction

    # ══════════════════════════════════════════════════════════════════════
    # Multi-sample training
    # ══════════════════════════════════════════════════════════════════════

    # ─────────────────────────────────────────────────────────────────────
    # Velocity normaliser helpers (hierarchical / baseline_mode=none)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _welford_batch_update(
        n: int,
        mean: np.ndarray,
        M2: np.ndarray,
        batch: np.ndarray,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """One-pass batch Welford update for component-wise mean+variance."""
        m = batch.shape[0]
        b_mean = batch.mean(axis=0)
        b_M2   = ((batch - b_mean) ** 2).sum(axis=0)
        delta  = b_mean - mean
        n_new  = n + m
        mean_new = mean + delta * (m / n_new)
        M2_new   = M2 + b_M2 + delta ** 2 * (n * m / n_new)
        return n_new, mean_new, M2_new

    def _fit_multi(
        self,
        samples: list[dict[str, Any]],
        train_indices: np.ndarray,
        val_indices: np.ndarray | None = None,
        fe_pipeline: list | None = None,
        graph_cache: dict | None = None,
        hgb_cache: dict | None = None,
        max_points: int = 0,
    ) -> dict[str, Any]:
        """Train on multiple samples (windows), loading one at a time."""

        if self._use_hierarchical:
            return self._fit_multi_hierarchical(
                samples, train_indices,
                val_indices=val_indices,
                fe_pipeline=fe_pipeline,
                hgb_cache=hgb_cache,
                max_points=max_points,
            )

        # ── Legacy flat-model path ───────────────────────────────────────
        normalizers = [
            step for step in (fe_pipeline or [])
            if hasattr(step, "fit_on_samples") and not step._is_fitted
        ]
        if normalizers:
            logger.info(
                "GFF._fit_multi: pre-fitting %d FeatureNormalizer(s) on %d training samples",
                len(normalizers), len(train_indices),
            )
            _saved_fba = self._feed_baseline_as_feature
            self._feed_baseline_as_feature = False
            prefit_pipeline = [s for s in (fe_pipeline or []) if s not in normalizers]
            # ── Stream without accumulation (prevents OOM on large datasets) ──
            # Tell each normalizer that an incremental fit is starting, then
            # feed one array at a time, then finalise.  Falls back to the
            # legacy fit_on_samples path for normalizers that lack the API.
            streaming_norms = [n for n in normalizers if hasattr(n, "begin_incremental_fit")]
            legacy_norms    = [n for n in normalizers if not hasattr(n, "begin_incremental_fit")]
            for norm in streaming_norms:
                norm.begin_incremental_fit()
            legacy_buf: list[dict] = [] if legacy_norms else None  # type: ignore[assignment]
            for idx in train_indices:
                sd = self._prepare_sample(
                    samples[idx],
                    fe_pipeline=prefit_pipeline,
                    graph_cache=graph_cache,
                    max_points=max_points,
                )
                pf = sd["point_features"]
                for norm in streaming_norms:
                    norm.update_incremental_fit(pf)
                if legacy_norms is not None:
                    legacy_buf.append({"point_features": pf})  # type: ignore[union-attr]
                del sd, pf
            for norm in streaming_norms:
                norm.finalize_incremental_fit()
            for norm in legacy_norms:
                norm.fit_on_samples(legacy_buf)  # type: ignore[arg-type]
            del legacy_buf
            self._feed_baseline_as_feature = _saved_fba

        first = self._prepare_sample(
            samples[train_indices[0]],
            fe_pipeline=fe_pipeline,
            graph_cache=graph_cache,
            max_points=max_points,
        )
        self._point_feature_dim = first["point_features"].shape[1]
        self._edge_dim = first["edge_attr"].shape[1] if first["edge_attr"] is not None else 0
        self._T_out = first["target_delta"].shape[0]
        del first

        self._build_model()
        optimizer = self._make_optimizer()
        scheduler = self._make_scheduler(optimizer)

        if self._seed is not None:
            torch.manual_seed(self._seed)

        best_loss = float("inf")
        patience_counter = 0
        self._training_history = []
        n_train = len(train_indices)

        self._model.train()
        for epoch in range(self._num_epochs):
            rng = np.random.RandomState(self._seed + epoch if self._seed else epoch)
            order = rng.permutation(n_train)
            epoch_loss = 0.0
            epoch_mae = 0.0

            for si in order:
                idx = train_indices[si]
                sd = self._prepare_sample(
                    samples[idx],
                    fe_pipeline=fe_pipeline,
                    graph_cache=graph_cache,
                    max_points=max_points,
                )
                pf_t = torch.from_numpy(sd["point_features"]).to(self._device)
                ei_t = torch.from_numpy(sd["edge_index"]).to(self._device)
                ea_t = torch.from_numpy(sd["edge_attr"]).to(self._device) if sd["edge_attr"] is not None else None
                td_t = torch.from_numpy(sd["target_delta"]).to(self._device)
                _af_idcs = sd.get("idcs_airfoil")
                _baseline_np = sd["baseline"] if self._physics_loss_weight > 0.0 else None
                del sd

                optimizer.zero_grad()
                pred_delta = self._model(pf_t, ei_t, ea_t)
                loss = F.mse_loss(pred_delta, td_t)

                if (self._physics_loss_weight > 0.0
                        and _af_idcs is not None and len(_af_idcs) > 0
                        and _baseline_np is not None):
                    baseline_af = torch.from_numpy(_baseline_np[:, _af_idcs, :]).to(self._device)
                    physics_loss = F.mse_loss(pred_delta[:, _af_idcs, :], -baseline_af)
                    loss = loss + self._physics_loss_weight * physics_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                with torch.no_grad():
                    epoch_mae += F.l1_loss(pred_delta, td_t).item()
                del pf_t, ei_t, ea_t, td_t, pred_delta, loss

            avg_loss = epoch_loss / n_train
            avg_mae = epoch_mae / n_train
            entry: dict[str, float] = {
                "epoch": epoch, "train_loss": avg_loss, "train_mae": avg_mae,
            }

            # ── Validation pass ──────────────────────────────────────────
            if val_indices is not None and len(val_indices) > 0:
                self._model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for vi in val_indices:
                        vsd = self._prepare_sample(
                            samples[vi],
                            fe_pipeline=fe_pipeline,
                            graph_cache=graph_cache,
                            max_points=max_points,
                        )
                        vpf = torch.from_numpy(vsd["point_features"]).to(self._device)
                        vei = torch.from_numpy(vsd["edge_index"]).to(self._device)
                        vea = torch.from_numpy(vsd["edge_attr"]).to(self._device) if vsd["edge_attr"] is not None else None
                        vtd = torch.from_numpy(vsd["target_delta"]).to(self._device)
                        vp  = self._model(vpf, vei, vea)
                        val_loss += F.mse_loss(vp, vtd).item()
                        del vpf, vei, vea, vtd, vp, vsd
                val_loss /= len(val_indices)
                entry["val_loss"] = val_loss
                self._model.train()
                monitor_loss = val_loss
            else:
                monitor_loss = avg_loss
            # ─────────────────────────────────────────────────────────────

            self._training_history.append(entry)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(monitor_loss)
                else:
                    scheduler.step()

            if monitor_loss < best_loss:
                best_loss = monitor_loss
                patience_counter = 0
                self._best_state_dict = copy.deepcopy(self._model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self._patience:
                    logger.info("GraphFlowForecaster: early stopping at epoch %d", epoch)
                    break

            if epoch % max(1, self._num_epochs // 10) == 0:
                val_str = f"  val_loss={entry['val_loss']:.6f}" if "val_loss" in entry else ""
                logger.info(
                    "GFF: epoch %d/%d  train_loss=%.6f  mae=%.6f%s",
                    epoch, self._num_epochs, avg_loss, avg_mae, val_str,
                )

        if self._best_state_dict is not None:
            self._model.load_state_dict(self._best_state_dict)

        self._is_trained = True
        logger.info("GraphFlowForecaster: training done, best_monitor_loss=%.6f", best_loss)

        return {
            "training_history": self._training_history,
            "best_epoch_meta": {"best_loss": best_loss, "total_epochs": len(self._training_history)},
        }

    # ─────────────────────────────────────────────────────────────────────
    # Hierarchical training
    # ─────────────────────────────────────────────────────────────────────

    def _fit_multi_hierarchical(
        self,
        samples: list[dict[str, Any]],
        train_indices: np.ndarray,
        val_indices: np.ndarray | None = None,
        fe_pipeline: list | None = None,
        hgb_cache: dict | None = None,
        max_points: int = 0,
    ) -> dict[str, Any]:
        """
        Train ``HierarchicalGFFModel`` on multiple samples.

        Steps
        -----
        1. Pre-fit FeatureNormalizer on geometry features.
        2. Compute velocity mean/std (Welford) for output normalisation
           when ``baseline_mode="none"``.
        3. Infer model dimensions from first sample.
        4. Train with proximity-weighted MSE loss.
        """
        n_train = len(train_indices)

        # ── Step 1: Pre-fit FeatureNormalizer on geometry features ────────
        normalizers = [
            step for step in (fe_pipeline or [])
            if hasattr(step, "fit_on_samples") and not step._is_fitted
        ]
        if normalizers:
            logger.info(
                "GFF[hier]: pre-fitting %d FeatureNormalizer(s) on %d train samples",
                len(normalizers), n_train,
            )
            pipeline_no_norm = [s for s in (fe_pipeline or []) if s not in normalizers]
            streaming_norms_h = [n for n in normalizers if hasattr(n, "begin_incremental_fit")]
            legacy_norms_h    = [n for n in normalizers if not hasattr(n, "begin_incremental_fit")]
            for norm in streaming_norms_h:
                norm.begin_incremental_fit()
            legacy_buf_h: list[dict] = [] if legacy_norms_h else None  # type: ignore[assignment]
            for idx in train_indices:
                sd = self._prepare_sample_hierarchical(
                    samples[idx],
                    fe_pipeline=pipeline_no_norm,
                    hgb_cache=hgb_cache,
                    max_points=max_points,
                )
                pf_h = sd["geo_features"]
                for norm in streaming_norms_h:
                    norm.update_incremental_fit(pf_h)
                if legacy_norms_h is not None:
                    legacy_buf_h.append({"point_features": pf_h})  # type: ignore[union-attr]
                del sd, pf_h
            for norm in streaming_norms_h:
                norm.finalize_incremental_fit()
            for norm in legacy_norms_h:
                norm.fit_on_samples(legacy_buf_h)  # type: ignore[arg-type]
            del legacy_buf_h

        # ── Step 2: Velocity mean/std for output normalisation ────────────
        if self._baseline_mode == "none":
            logger.info("GFF[hier]: computing velocity normaliser from %d train samples", n_train)
            n_stat, mean_stat, M2_stat = 0, np.zeros(3, np.float64), np.zeros(3, np.float64)
            for idx in train_indices:
                if "velocity_out" in samples[idx]:
                    vout = np.asarray(samples[idx]["velocity_out"], np.float64)
                else:
                    from src.backend.data_digester.temporal_point_cloud_field_digester import (
                        TemporalPointCloudFieldDigester,
                    )
                    arr = TemporalPointCloudFieldDigester.load_sample(
                        samples[idx]["filepath"], keys={"velocity_out"}, max_points=max_points,
                    )
                    vout = np.asarray(arr["velocity_out"], np.float64)
                    del arr
                batch = vout.reshape(-1, 3)
                n_stat, mean_stat, M2_stat = self._welford_batch_update(
                    n_stat, mean_stat, M2_stat, batch,
                )
                del vout, batch
            self._vel_mean = mean_stat.astype(np.float32)
            self._vel_std  = (np.sqrt(M2_stat / max(n_stat, 1)) + 1e-8).astype(np.float32)
            logger.info(
                "GFF[hier]: vel_mean=%.3f±%.3f (per component)",
                self._vel_mean.mean(), self._vel_std.mean(),
            )
        else:
            self._vel_mean = np.zeros(3, np.float32)
            self._vel_std  = np.ones(3,  np.float32)

        # ── Step 3: Infer dims from first training sample ─────────────────
        first = self._prepare_sample_hierarchical(
            samples[train_indices[0]],
            fe_pipeline=fe_pipeline,
            hgb_cache=hgb_cache,
            max_points=max_points,
        )
        self._point_feature_dim = first["geo_features"].shape[1]
        self._edge_dim = (
            first["edge_attr_fine"].shape[1]
            if first["edge_attr_fine"] is not None else 0
        )
        self._T_out = first["velocity_out"].shape[0]
        self._T_in  = first["temporal_input"].shape[1]
        self._temporal_input_dim = first["temporal_input"].shape[2]
        del first

        self._build_model()
        optimizer = self._make_optimizer()
        scheduler = self._make_scheduler(optimizer)

        if self._seed is not None:
            torch.manual_seed(self._seed)

        # Pre-build normaliser tensors on device
        vel_mean_t = torch.from_numpy(self._vel_mean).to(self._device)  # (3,)
        vel_std_t  = torch.from_numpy(self._vel_std).to(self._device)   # (3,)

        best_loss = float("inf")
        patience_counter = 0
        self._training_history = []

        self._model.train()
        for epoch in range(self._num_epochs):
            rng = np.random.RandomState(
                self._seed + epoch if self._seed is not None else epoch
            )
            order = rng.permutation(n_train)
            epoch_loss = 0.0
            epoch_mae  = 0.0

            for si in order:
                idx = train_indices[si]
                sd = self._prepare_sample_hierarchical(
                    samples[idx],
                    fe_pipeline=fe_pipeline,
                    hgb_cache=hgb_cache,
                    max_points=max_points,
                )

                # ── Move to device ────────────────────────────────────────
                vel_t   = torch.from_numpy(sd["temporal_input"]).to(self._device)   # (N,T,C)
                geo_t   = torch.from_numpy(sd["geo_features"]).to(self._device)     # (N, D)
                ei_f    = torch.from_numpy(sd["edge_index_fine"]).to(self._device)  # (2, Ef)
                ea_f    = (torch.from_numpy(sd["edge_attr_fine"]).to(self._device)
                           if sd["edge_attr_fine"] is not None else None)
                ei_c    = torch.from_numpy(sd["edge_index_coarse"]).to(self._device)
                ea_c    = (torch.from_numpy(sd["edge_attr_coarse"]).to(self._device)
                           if sd["edge_attr_coarse"] is not None else None)
                ci_t    = torch.from_numpy(sd["coarse_indices"]).to(self._device)   # (K,)
                ftc_idx = torch.from_numpy(sd["unpool_ftc_idx"]).to(self._device)   # (N, k)
                ftc_w   = torch.from_numpy(sd["unpool_ftc_w"]).to(self._device)     # (N, k)

                vel_out_np  = sd["velocity_out"]                                     # (T_out,N,3)
                _af_idcs    = sd.get("idcs_airfoil")
                _dist_to_af = sd.get("dist_to_af")
                del sd

                # ── Normalise target ──────────────────────────────────────
                if self._baseline_mode == "none":
                    target_np = ((vel_out_np - self._vel_mean) / self._vel_std)
                else:
                    vel_in_np = np.asarray(
                        samples[idx].get("velocity_in",
                                         vel_out_np[:1].repeat(self._T_in, axis=0)),
                        np.float32,
                    )
                    baseline_np = self._baseline(vel_in_np, self._T_out)
                    target_np   = vel_out_np - baseline_np

                target_t = torch.from_numpy(target_np.astype(np.float32)).to(self._device)

                # ── Forward ───────────────────────────────────────────────
                optimizer.zero_grad()
                pred_t = self._model(
                    vel_t, geo_t,
                    ei_f, ea_f,
                    ei_c, ea_c,
                    ci_t, ftc_idx, ftc_w,
                )   # (T_out, N, 3)

                # ── Proximity-weighted MSE ────────────────────────────────
                if _dist_to_af is not None and self._proximity_loss_weight > 0:
                    dist_np = _dist_to_af.astype(np.float32)
                    w_np = (1.0 + self._proximity_loss_weight
                            * np.exp(-dist_np / self._proximity_sigma))
                    w_t = torch.from_numpy(w_np).to(self._device)   # (N,)
                    # per-node mean over time and components, then weighted mean
                    loss_per_node = ((pred_t - target_t) ** 2).mean(dim=(0, 2))  # (N,)
                    loss = (loss_per_node * w_t).mean()
                else:
                    loss = F.mse_loss(pred_t, target_t)

                # ── No-slip physics loss ──────────────────────────────────
                if (self._physics_loss_weight > 0.0
                        and _af_idcs is not None and len(_af_idcs) > 0):
                    if self._baseline_mode == "none":
                        # Normalised target at airfoil = (0 - mean) / std
                        af_target = (-vel_mean_t / vel_std_t).view(1, 1, 3).expand(
                            self._T_out, len(_af_idcs), 3
                        )
                    else:
                        af_target = torch.zeros(
                            self._T_out, len(_af_idcs), 3,
                            device=self._device,
                        )
                    physics_loss = F.mse_loss(pred_t[:, _af_idcs, :], af_target)
                    loss = loss + self._physics_loss_weight * physics_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                with torch.no_grad():
                    epoch_mae += F.l1_loss(pred_t, target_t).item()
                del vel_t, geo_t, ei_f, ea_f, ei_c, ea_c, ci_t
                del ftc_idx, ftc_w, pred_t, target_t, loss

            avg_loss = epoch_loss / n_train
            avg_mae  = epoch_mae  / n_train
            entry_h: dict[str, float] = {
                "epoch": epoch, "train_loss": avg_loss, "train_mae": avg_mae,
            }

            # ── Validation pass ──────────────────────────────────────────
            if val_indices is not None and len(val_indices) > 0:
                self._model.eval()
                val_loss_h = 0.0
                with torch.no_grad():
                    for vi in val_indices:
                        vsd = self._prepare_sample_hierarchical(
                            samples[vi],
                            fe_pipeline=fe_pipeline,
                            hgb_cache=hgb_cache,
                            max_points=max_points,
                        )
                        vvel  = torch.from_numpy(vsd["temporal_input"]).to(self._device)
                        vgeo  = torch.from_numpy(vsd["geo_features"]).to(self._device)
                        veif  = torch.from_numpy(vsd["edge_index_fine"]).to(self._device)
                        veaf  = (torch.from_numpy(vsd["edge_attr_fine"]).to(self._device)
                                 if vsd["edge_attr_fine"] is not None else None)
                        veic  = torch.from_numpy(vsd["edge_index_coarse"]).to(self._device)
                        veac  = (torch.from_numpy(vsd["edge_attr_coarse"]).to(self._device)
                                 if vsd["edge_attr_coarse"] is not None else None)
                        vci   = torch.from_numpy(vsd["coarse_indices"]).to(self._device)
                        vftci = torch.from_numpy(vsd["unpool_ftc_idx"]).to(self._device)
                        vftcw = torch.from_numpy(vsd["unpool_ftc_w"]).to(self._device)
                        vout_np = vsd["velocity_out"]
                        if self._baseline_mode == "none":
                            vtgt_np = (vout_np - self._vel_mean) / self._vel_std
                        else:
                            vin_np = vsd["velocity_in"]
                            vtgt_np = vout_np - self._baseline(vin_np, self._T_out)
                        vtgt = torch.from_numpy(vtgt_np.astype(np.float32)).to(self._device)
                        vpred = self._model(vvel, vgeo, veif, veaf, veic, veac, vci, vftci, vftcw)
                        val_loss_h += F.mse_loss(vpred, vtgt).item()
                        del vvel, vgeo, veif, veaf, veic, veac, vci, vftci, vftcw, vpred, vtgt, vsd
                val_loss_h /= len(val_indices)
                entry_h["val_loss"] = val_loss_h
                self._model.train()
                monitor_loss_h = val_loss_h
            else:
                monitor_loss_h = avg_loss
            # ─────────────────────────────────────────────────────────────

            self._training_history.append(entry_h)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(monitor_loss_h)
                else:
                    scheduler.step()

            if monitor_loss_h < best_loss:
                best_loss = monitor_loss_h
                patience_counter = 0
                self._best_state_dict = copy.deepcopy(self._model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self._patience:
                    logger.info("GFF[hier]: early stopping at epoch %d", epoch)
                    break

            if epoch % max(1, self._num_epochs // 10) == 0:
                val_str_h = f"  val_loss={entry_h['val_loss']:.6f}" if "val_loss" in entry_h else ""
                logger.info(
                    "GFF[hier]: epoch %d/%d  train_loss=%.6f  mae=%.6f%s",
                    epoch, self._num_epochs, avg_loss, avg_mae, val_str_h,
                )

        if self._best_state_dict is not None:
            self._model.load_state_dict(self._best_state_dict)

        self._is_trained = True
        logger.info("GFF[hier]: training done, best_monitor_loss=%.6f", best_loss)

        return {
            "training_history": self._training_history,
            "best_epoch_meta": {
                "best_loss": best_loss,
                "total_epochs": len(self._training_history),
            },
        }

    def _predict_sample(
        self,
        sample: dict[str, Any],
        fe_pipeline: list | None = None,
        graph_cache: dict | None = None,
        hgb_cache: dict | None = None,
        max_points: int = 0,
    ) -> np.ndarray:
        """Run inference on a single sample.  Returns (T_out, N, 3)."""
        if self._use_hierarchical:
            return self._predict_sample_hierarchical(
                sample, fe_pipeline=fe_pipeline,
                hgb_cache=hgb_cache, max_points=max_points,
            )

        prepared = self._prepare_sample_flat(
            sample, fe_pipeline=fe_pipeline,
            graph_cache=graph_cache, max_points=max_points,
        )
        self._model.eval()
        with torch.no_grad():
            pf_t = torch.from_numpy(prepared["point_features"]).to(self._device)
            ei_t = torch.from_numpy(prepared["edge_index"]).to(self._device)
            ea_t = (torch.from_numpy(prepared["edge_attr"]).to(self._device)
                    if prepared["edge_attr"] is not None else None)
            delta = self._model(pf_t, ei_t, ea_t).cpu().numpy()
            del pf_t, ei_t, ea_t

        pred = prepared["baseline"] + delta
        pred = self._enforce_no_slip(pred, prepared["idcs_airfoil"])
        return pred.astype(np.float32)

    def _predict_sample_hierarchical(
        self,
        sample: dict[str, Any],
        fe_pipeline: list | None = None,
        hgb_cache: dict | None = None,
        max_points: int = 0,
    ) -> np.ndarray:
        """Hierarchical-model inference.  Returns (T_out, N, 3)."""
        sd = self._prepare_sample_hierarchical(
            sample, fe_pipeline=fe_pipeline,
            hgb_cache=hgb_cache, max_points=max_points,
        )
        self._model.eval()
        with torch.no_grad():
            vel_t   = torch.from_numpy(sd["temporal_input"]).to(self._device)
            geo_t   = torch.from_numpy(sd["geo_features"]).to(self._device)
            ei_f    = torch.from_numpy(sd["edge_index_fine"]).to(self._device)
            ea_f    = (torch.from_numpy(sd["edge_attr_fine"]).to(self._device)
                       if sd["edge_attr_fine"] is not None else None)
            ei_c    = torch.from_numpy(sd["edge_index_coarse"]).to(self._device)
            ea_c    = (torch.from_numpy(sd["edge_attr_coarse"]).to(self._device)
                       if sd["edge_attr_coarse"] is not None else None)
            ci_t    = torch.from_numpy(sd["coarse_indices"]).to(self._device)
            ftc_idx = torch.from_numpy(sd["unpool_ftc_idx"]).to(self._device)
            ftc_w   = torch.from_numpy(sd["unpool_ftc_w"]).to(self._device)

            pred_norm = self._model(
                vel_t, geo_t, ei_f, ea_f, ei_c, ea_c, ci_t, ftc_idx, ftc_w,
            ).cpu().numpy()   # (T_out, N, 3)

        # Denormalise
        if self._baseline_mode == "none":
            pred = pred_norm * self._vel_std + self._vel_mean
        else:
            vel_in_np = sd["velocity_in"]
            baseline  = self._baseline(vel_in_np, self._T_out)
            pred      = pred_norm + baseline

        pred = self._enforce_no_slip(pred, sd.get("idcs_airfoil"))
        return pred.astype(np.float32)

    def _predict_multi(
        self,
        samples: list[dict[str, Any]],
        indices: np.ndarray | None = None,
        fe_pipeline: list | None = None,
        graph_cache: dict | None = None,
        hgb_cache: dict | None = None,
        max_points: int = 0,
    ) -> list[np.ndarray]:
        """Run inference on selected samples.  Returns list of (T_out,N,3)."""
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model not trained.")

        idxs = indices if indices is not None else np.arange(len(samples))
        predictions: list[np.ndarray] = []

        for i in idxs:
            pred = self._predict_sample(
                samples[i],
                fe_pipeline=fe_pipeline,
                graph_cache=graph_cache,
                hgb_cache=hgb_cache,
                max_points=max_points,
            )
            predictions.append(pred)

        return predictions

    # ══════════════════════════════════════════════════════════════════════
    # Single-sample training (legacy)
    # ══════════════════════════════════════════════════════════════════════

    def fit(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Train on a single sample (legacy mode)."""
        point_features = np.asarray(inputs["point_features"], dtype=np.float32)
        edge_index = np.asarray(inputs["edge_index"], dtype=np.int64)
        velocity_in = np.asarray(inputs["velocity_in"], dtype=np.float32)
        velocity_out = np.asarray(inputs["velocity_out"], dtype=np.float32)
        edge_attr = inputs.get("edge_attr")
        if edge_attr is not None:
            edge_attr = np.asarray(edge_attr, dtype=np.float32)

        self._T_out = velocity_out.shape[0]
        self._point_feature_dim = point_features.shape[1]
        self._edge_dim = edge_attr.shape[1] if edge_attr is not None else 0

        baseline = self._baseline(velocity_in, self._T_out)
        target_delta = velocity_out - baseline

        self._build_model()
        optimizer = self._make_optimizer()
        scheduler = self._make_scheduler(optimizer)

        pf_t = torch.from_numpy(point_features).to(self._device)
        ei_t = torch.from_numpy(edge_index).to(self._device)
        ea_t = torch.from_numpy(edge_attr).to(self._device) if edge_attr is not None else None
        td_t = torch.from_numpy(target_delta).to(self._device)

        best_loss = float("inf")
        patience_counter = 0
        self._training_history = []

        if self._seed is not None:
            torch.manual_seed(self._seed)

        self._model.train()
        for epoch in range(self._num_epochs):
            optimizer.zero_grad()
            pred_delta = self._model(pf_t, ei_t, ea_t)
            loss = F.mse_loss(pred_delta, td_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            entry: dict[str, float] = {"epoch": epoch, "train_loss": loss_val}
            with torch.no_grad():
                mae_val = F.l1_loss(pred_delta, td_t).item()
                entry["train_mae"] = mae_val
            self._training_history.append(entry)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss_val)
                else:
                    scheduler.step()

            if loss_val < best_loss:
                best_loss = loss_val
                patience_counter = 0
                self._best_state_dict = copy.deepcopy(self._model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self._patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

            if epoch % max(1, self._num_epochs // 10) == 0:
                logger.info(
                    "GraphFlowForecaster: epoch %d/%d loss=%.6f mae=%.6f",
                    epoch, self._num_epochs, loss_val, mae_val,
                )

        if self._best_state_dict is not None:
            self._model.load_state_dict(self._best_state_dict)

        self._is_trained = True
        logger.info("GraphFlowForecaster: training complete, best_loss=%.6f", best_loss)

        return {
            "training_history": self._training_history,
            "best_epoch_meta": {
                "best_loss": best_loss,
                "total_epochs": len(self._training_history),
            },
        }

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run inference on a single sample (legacy mode)."""
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        point_features = np.asarray(inputs["point_features"], dtype=np.float32)
        edge_index = np.asarray(inputs["edge_index"], dtype=np.int64)
        velocity_in = np.asarray(inputs["velocity_in"], dtype=np.float32)
        edge_attr = inputs.get("edge_attr")
        if edge_attr is not None:
            edge_attr = np.asarray(edge_attr, dtype=np.float32)

        baseline = self._baseline(velocity_in, self._T_out)

        self._model.eval()
        with torch.no_grad():
            pf_t = torch.from_numpy(point_features).to(self._device)
            ei_t = torch.from_numpy(edge_index).to(self._device)
            ea_t = torch.from_numpy(edge_attr).to(self._device) if edge_attr is not None else None
            delta = self._model(pf_t, ei_t, ea_t).cpu().numpy()

        predicted = baseline + delta
        return {"predicted_velocity_out": predicted.astype(np.float32)}

    # ══════════════════════════════════════════════════════════════════════
    # Pipeline interface
    # ══════════════════════════════════════════════════════════════════════

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Full pipeline execution.  Detects multi-sample mode when
        ``samples`` key is present.
        """
        samples = inputs.get("samples")
        if samples is not None and isinstance(samples, list):
            return self._execute_multi(inputs)
        return self._execute_single(inputs)

    def _execute_single(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Single-sample execute (legacy)."""
        fit_result = self.fit(inputs)
        pred_result = self.predict(inputs)

        predicted = pred_result["predicted_velocity_out"]
        velocity_out = np.asarray(inputs["velocity_out"], dtype=np.float32)

        from src.backend.predictors.regressors.flow_loss import r_squared
        metrics: dict[str, Any] = {"r2_overall": r_squared(predicted, velocity_out)}

        train_ids = inputs.get("train_ids")
        test_ids = inputs.get("test_ids")
        if train_ids is not None:
            train_ids = np.asarray(train_ids)
            metrics["r2_train"] = r_squared(predicted[:, train_ids], velocity_out[:, train_ids])
        if test_ids is not None:
            test_ids = np.asarray(test_ids)
            metrics["r2_test"] = r_squared(predicted[:, test_ids], velocity_out[:, test_ids])

        result: dict[str, Any] = {**inputs}
        result["predicted_velocity_out"] = predicted
        result["training_history"] = fit_result["training_history"]
        result["best_epoch_meta"] = fit_result["best_epoch_meta"]
        result["metrics"] = metrics
        result["model_artifact"] = self
        result["latent_meta"] = {
            "latent_dim": self._latent_dim,
            "num_mp_layers": self._num_mp_layers,
            "point_feature_dim": self._point_feature_dim,
            "edge_dim": self._edge_dim,
            "T_out": self._T_out,
            "baseline_mode": self._baseline_mode,
            "use_temporal_decoder": self._use_temporal_decoder,
            "feed_baseline_as_feature": self._feed_baseline_as_feature,
            "use_hierarchical": getattr(self, "_use_hierarchical", False),
            "xlstm_head_dim": getattr(self, "_xlstm_head_dim", None),
            "xlstm_num_layers": getattr(self, "_xlstm_num_layers", None),
            "xlstm_output_dim": getattr(self, "_xlstm_output_dim", None),
        }
        return result

    def _execute_multi(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Multi-sample execute: train on train split, predict all."""
        samples = inputs["samples"]
        train_indices = np.asarray(inputs["train_indices"], dtype=np.int64)
        val_indices_raw = inputs.get("val_indices")
        val_indices = np.asarray(val_indices_raw, dtype=np.int64) if val_indices_raw is not None and len(val_indices_raw) > 0 else None
        test_indices = np.asarray(inputs.get("test_indices", []), dtype=np.int64)
        all_indices = np.arange(len(samples))

        # Grab lazy-pipeline components from upstream FE nodes
        fe_pipeline = inputs.get("fe_pipeline")
        graph_cache = inputs.get("graph_cache")    # legacy SGB cache
        hgb_cache   = inputs.get("hgb_cache")     # new HGB cache
        max_points  = int(inputs.get("max_points", 0))

        # Auto-detect hierarchical mode from the pipeline
        self._detect_pipeline_components(fe_pipeline)

        # Train
        fit_result = self._fit_multi(
            samples, train_indices,
            val_indices=val_indices,
            fe_pipeline=fe_pipeline,
            graph_cache=graph_cache,
            hgb_cache=hgb_cache,
            max_points=max_points,
        )

        # ── Streaming R² computation (one sample at a time) ─────────────
        from src.backend.predictors.regressors.flow_loss import r_squared

        def _streaming_r2(idxs: np.ndarray) -> float:
            """Compute R² over *idxs* without storing all arrays."""
            n = 0
            ss_res = 0.0
            sum_y = 0.0
            sum_y2 = 0.0
            for i in idxs:
                pred = self._predict_sample(
                    samples[i],
                    fe_pipeline=fe_pipeline,
                    graph_cache=graph_cache,
                    hgb_cache=hgb_cache,
                    max_points=max_points,
                )
                # Load target from disk
                if "velocity_out" in samples[i]:
                    tgt = np.asarray(samples[i]["velocity_out"], dtype=np.float32)
                else:
                    from src.backend.data_digester.temporal_point_cloud_field_digester import (
                        TemporalPointCloudFieldDigester,
                    )
                    arrays = TemporalPointCloudFieldDigester.load_sample(
                        samples[i]["filepath"],
                        keys={"velocity_out"},
                        max_points=max_points,
                    )
                    tgt = np.asarray(arrays["velocity_out"], dtype=np.float32)
                    del arrays

                y = tgt.ravel().astype(np.float64)
                yhat = pred.ravel().astype(np.float64)
                ss_res += float(np.sum((y - yhat) ** 2))
                sum_y += float(np.sum(y))
                sum_y2 += float(np.sum(y ** 2))
                n += len(y)
                del pred, tgt, y, yhat

            ss_tot = sum_y2 - (sum_y ** 2) / max(n, 1)
            return 1.0 - ss_res / max(ss_tot, 1e-15)

        r2_train = _streaming_r2(train_indices)
        r2_test = _streaming_r2(test_indices) if len(test_indices) > 0 else None
        r2_overall = _streaming_r2(all_indices)

        metrics: dict[str, Any] = {
            "r2_overall": r2_overall,
            "r2_train": r2_train,
        }
        if r2_test is not None:
            metrics["r2_test"] = r2_test

        logger.info(
            "GFF multi: R²(train)=%.4f  R²(test)=%s  R²(all)=%.4f",
            r2_train,
            f"{r2_test:.4f}" if r2_test is not None else "N/A",
            r2_overall,
        )

        # ── Materialise a few samples for downstream FieldSlicePlot ──────
        # FieldSlicePlot can generate predictions on-the-fly via
        # model_artifact, so we only need to pass through the pipeline
        # components.

        result: dict[str, Any] = {**inputs}
        result["samples"] = samples
        result["training_history"] = fit_result["training_history"]
        result["best_epoch_meta"] = fit_result["best_epoch_meta"]
        result["metrics"] = metrics
        result["model_artifact"] = self
        result["latent_meta"] = {
            "latent_dim": self._latent_dim,
            "num_mp_layers": self._num_mp_layers,
            "point_feature_dim": self._point_feature_dim,
            "edge_dim": self._edge_dim,
            "T_out": self._T_out,
            "baseline_mode": self._baseline_mode,
            "use_temporal_decoder": self._use_temporal_decoder,
            "feed_baseline_as_feature": self._feed_baseline_as_feature,
            "use_hierarchical": getattr(self, "_use_hierarchical", False),
            "xlstm_head_dim": getattr(self, "_xlstm_head_dim", None),
            "xlstm_num_layers": getattr(self, "_xlstm_num_layers", None),
            "xlstm_output_dim": getattr(self, "_xlstm_output_dim", None),
        }
        return result

    # ── Serialisation ────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model state and config to disk."""
        _require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "hp": self._hp,
            "point_feature_dim": self._point_feature_dim,
            "edge_dim": self._edge_dim,
            "T_out": self._T_out,
            "state_dict": self._model.state_dict() if self._model else None,
            "training_history": self._training_history,
        }
        torch.save(bundle, str(path))
        logger.info("GraphFlowForecaster saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "GraphFlowForecaster":
        """Load a saved model from disk."""
        _require_torch()
        path = Path(path)
        bundle = torch.load(str(path), map_location=device or "cpu", weights_only=False)
        inst = cls(hyperparams=bundle["hp"])
        inst._point_feature_dim = bundle["point_feature_dim"]
        inst._edge_dim = bundle["edge_dim"]
        inst._T_out = bundle["T_out"]
        inst._training_history = bundle.get("training_history", [])
        inst._build_model()
        if bundle["state_dict"] is not None:
            inst._model.load_state_dict(bundle["state_dict"])
        inst._is_trained = True
        if device:
            inst._device = torch.device(device)
            inst._model.to(inst._device)
        return inst

    def parameter_count(self) -> int:
        """Return total trainable parameter count."""
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)
