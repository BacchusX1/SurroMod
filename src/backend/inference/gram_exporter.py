"""
GRaM Model Exporter
===================
Pipeline-executable node that, after a GraphFlowForecaster finishes training,
automatically generates a fully self-contained submission for the

    GRaM Competition @ ICLR 2026
    https://github.com/gram-competition/iclr-2026

Submission structure created inside <gram_repo_dir>/models/<model_name>/:
    model.py          – submission class implementing the GRaM __call__ API
    arch.py           – extracted PyTorch GFF architecture (no SurroMod deps)
    weights.pt        – torch.save'd model state_dict
    config.pt         – normaliser stats + all FE hyperparams
    README.md         – approach description

models/__init__.py is updated with the new import entry.

Optionally: git-commits, pushes a feature branch, and opens a PR via `gh`.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)


# ─── Template: arch.py ────────────────────────────────────────────────────────

_ARCH_PY_HEADER = '''\
"""
GFF Architecture — extracted from SurroMod for GRaM submission.
Auto-generated. Do not edit manually.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
'''

def _extract_arch_source() -> str:
    """Extract the PyTorch model classes from graph_flow_forecaster.py."""
    try:
        from src.backend.predictors.regressors.graph_flow_forecaster import (
            PointEncoder,
            GraphMessagePassingBlock,
            TemporalLSTMDecoder,
            ResidualVelocityHead,
            GraphFlowForecasterModel,
        )
        classes = [
            PointEncoder,
            GraphMessagePassingBlock,
            TemporalLSTMDecoder,
            ResidualVelocityHead,
            GraphFlowForecasterModel,
        ]
        parts = [_ARCH_PY_HEADER]
        for cls in classes:
            src = inspect.getsource(cls)
            # Remove any leading indentation (classes are inside `if _TORCH_AVAILABLE:`)
            src = textwrap.dedent(src)
            parts.append("\n\n" + src)
        return "\n".join(parts)
    except Exception as e:
        logger.warning("GRAMExporter: could not extract arch via inspect: %s. Using stub.", e)
        return _ARCH_PY_HEADER + "\n# TODO: paste GraphFlowForecasterModel here\n"


# ─── Template: model.py ───────────────────────────────────────────────────────

_MODEL_PY_TEMPLATE = '''\
"""
{model_name} — GRaM ICLR-2026 submission
Team: {team_name}

Self-contained wrapper around a trained GraphFlowForecaster (GFF).

Usage (matches the GRaM evaluation harness):
    model = {class_name}()
    vel_out = model(t, pos, idcs_airfoil, velocity_in)
"""
from __future__ import annotations

import os, sys
# Allow sibling imports (arch.py lives in the same directory)
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from arch import GraphFlowForecasterModel


# ─── Feature-engineering helpers (inlined, no SurroMod dependency) ───────────

def _build_knn(pos_np: np.ndarray, k: int):
    """Build k-NN edge_index (2, E) and edge_attr (E, 4)."""
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(pos_np)
    dists_all, idxs_all = nn.kneighbors(pos_np)
    src_list, dst_list, d_list = [], [], []
    for i, (ds, ixs) in enumerate(zip(dists_all, idxs_all)):
        for d, j in zip(ds, ixs):
            if int(j) == i:
                continue
            src_list.append(int(j))
            dst_list.append(i)
            d_list.append(float(d))
    src_arr = np.array(src_list, dtype=np.int64)
    dst_arr = np.array(dst_list, dtype=np.int64)
    edge_index = np.stack([src_arr, dst_arr], axis=0)          # (2, E)
    rel_disp   = (pos_np[dst_arr] - pos_np[src_arr]).astype(np.float32)  # (E, 3)
    edge_dists = np.array(d_list, dtype=np.float32).reshape(-1, 1)       # (E, 1)
    edge_attr  = np.concatenate([rel_disp, edge_dists], axis=1)          # (E, 4)
    return edge_index, edge_attr


def _spectral_fft(vel_in: np.ndarray, cutoff_freq: float = {cutoff_freq}):
    """FFT decompose vel_in (T, N, 3) → (low_freq, high_freq) both (T, N, 3)."""
    T = vel_in.shape[0]
    V   = np.fft.rfft(vel_in, axis=0)
    cut = max(1, int(cutoff_freq * T))
    lo  = V.copy();  lo[cut:]  = 0
    hi  = V.copy();  hi[:cut]  = 0
    return (np.fft.irfft(lo, n=T, axis=0).astype(np.float32),
            np.fft.irfft(hi, n=T, axis=0).astype(np.float32))


def _surface_distance(pos_np: np.ndarray, idcs_airfoil: np.ndarray):
    """Compute dist_to_surface (N,1) and geometry_mask (N,1)."""
    N = pos_np.shape[0]
    surface_pts = pos_np[idcs_airfoil]
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(surface_pts)
    dists, _ = nn.kneighbors(pos_np)
    geo_mask = np.zeros((N, 1), dtype=np.float32)
    geo_mask[idcs_airfoil, 0] = 1.0
    return dists.astype(np.float32), geo_mask


def _fuse_and_normalise(
    pos_np, vel_history, low_freq_feats,
    geo_mask, dist_to_surface,
    center: np.ndarray, scale: np.ndarray,
):
    pf = np.concatenate(
        [pos_np, vel_history, low_freq_feats, geo_mask, dist_to_surface],
        axis=1,
    )  # (N, 35)
    return ((pf - center) / scale).astype(np.float32)


# ─── Submission class ─────────────────────────────────────────────────────────

class {class_name}:
    """
    GraphFlowForecaster surrogate model for 3-D airfoil flow prediction.

    Trained on the warped-ifw dataset with 162 simulations.
    Architecture: flat GFF (latent={latent_dim}, hidden={hidden_dim},
    {num_mp_layers} MP layers, mean aggregation, initial skip connection,
    mean-field baseline, cosine LR schedule).
    """

    def __init__(self) -> None:
        _dir = os.path.dirname(__file__)

        # ── Load config (normaliser stats + FE hyperparams) ──────────────
        cfg = torch.load(os.path.join(_dir, "config.pt"), map_location="cpu")
        self._k:             int        = int(cfg["k"])
        self._cutoff_freq:   float      = float(cfg["cutoff_freq"])
        self._T_out:         int        = int(cfg["T_out"])
        self._center: np.ndarray        = np.asarray(cfg["normalizer_center"], dtype=np.float32)
        self._scale:  np.ndarray        = np.asarray(cfg["normalizer_scale"],  dtype=np.float32)

        # ── Build model ──────────────────────────────────────────────────
        self._model = GraphFlowForecasterModel(
            point_feature_dim=int(cfg["point_feature_dim"]),
            latent_dim=int(cfg["latent_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            edge_dim=int(cfg["edge_dim"]),
            num_mp_layers=int(cfg["num_mp_layers"]),
            T_out=self._T_out,
            dropout=0.0,
            use_edge_attr=True,
            aggregation_mode=str(cfg.get("aggregation_mode", "mean")),
            skip_connection_mode=str(cfg.get("skip_connection_mode", "initial")),
            use_temporal_decoder=bool(cfg.get("use_temporal_decoder", True)),
        )

        weights_path = os.path.join(_dir, "weights.pt")
        self._model.load_state_dict(
            torch.load(weights_path, map_location="cpu")
        )
        self._model.eval()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)

    # ── GRaM API ──────────────────────────────────────────────────────────────

    def __call__(
        self,
        t:            "torch.Tensor",         # (B, 10)       — time grid (unused)
        pos:          "torch.Tensor",         # (B, N, 3)     — point positions
        idcs_airfoil: "list[torch.Tensor]",   # B × (K_b,)    — airfoil indices
        velocity_in:  "torch.Tensor",         # (B, 5, N, 3)  — input velocity
    ) -> "torch.Tensor":                      # (B, 5, N, 3)  — predicted velocity
        B = pos.shape[0]
        preds: list[np.ndarray] = []

        for b in range(B):
            pos_np  = pos[b].cpu().numpy().astype(np.float32)          # (N, 3)
            vel_np  = velocity_in[b].cpu().numpy().astype(np.float32)  # (5, N, 3)
            af_np   = idcs_airfoil[b].cpu().numpy().astype(np.int64)   # (K,)
            N       = pos_np.shape[0]

            # ── Feature engineering ──────────────────────────────��───────
            edge_index, edge_attr = _build_knn(pos_np, k=self._k)

            # Temporal flatten: (5, N, 3) → (N, 15)
            vel_history = vel_np.transpose(1, 0, 2).reshape(N, -1)

            # Spectral decompose + temporal flatten of low-freq: (N, 15)
            low_freq, _ = _spectral_fft(vel_np, cutoff_freq=self._cutoff_freq)
            low_freq_feats = low_freq.transpose(1, 0, 2).reshape(N, -1)

            dist_to_surface, geo_mask = _surface_distance(pos_np, af_np)

            point_features = _fuse_and_normalise(
                pos_np, vel_history, low_freq_feats,
                geo_mask, dist_to_surface,
                self._center, self._scale,
            )  # (N, 35)

            # ── Baseline: mean-field ─────────────────────────────────────
            baseline = np.tile(vel_np.mean(axis=0, keepdims=True), (self._T_out, 1, 1))  # (5, N, 3)

            # ── Model forward ────────────────────────────────────────────
            with torch.no_grad():
                pf_t = torch.from_numpy(point_features).float().to(self._device)
                ei_t = torch.from_numpy(edge_index).long().to(self._device)
                ea_t = torch.from_numpy(edge_attr).float().to(self._device)
                delta_np = self._model(pf_t, ei_t, ea_t).cpu().numpy()  # (5, N, 3)

            pred = (baseline + delta_np).astype(np.float32)

            # ── No-slip boundary condition ───────────────────────────────
            pred[:, af_np, :] = 0.0

            preds.append(pred)

        out = np.stack(preds, axis=0)  # (B, 5, N, 3)
        return torch.from_numpy(out)
'''


def _make_class_name(model_name: str) -> str:
    """Turn 'surromod_gff' → 'SurromodGff'."""
    return "".join(p.capitalize() for p in re.split(r"[_\-\s]+", model_name))


# ─── Template: README.md ──────────────────────────────────────────────────────

_README_TEMPLATE = """\
# {model_name}

**Team:** {team_name}

## Approach

Graph-based surrogate model (GraphFlowForecaster, GFF) trained on the warped-ifw dataset.

### Architecture
- **Encoder:** MLP (point_features → latent_dim={latent_dim})
- **Message passing:** {num_mp_layers} × GraphMessagePassingBlock (hidden_dim={hidden_dim}, mean aggregation, initial-residual skip)
- **Decoder:** TemporalLSTMDecoder + ResidualVelocityHead
- **Baseline:** mean-field extrapolation

### Feature engineering
- k-NN graph (k={k})
- Temporal stack flatten of velocity history (5 × 3 = 15 features)
- FFT spectral decomposition (cutoff_freq={cutoff_freq}), low-freq stack (15 features)
- Distance-to-surface + geometry mask (2 features)
- 3-D coordinates (3 features)
- **Total:** 35 node features, 4 edge features (Δx, Δy, Δz, dist)

### Training
- Dataset: warped-ifw, 162 simulations (5 temporal windows each)
- Split: 71% train / 14% val / 14% test (by simulation, no leakage)
- Optimizer: AdamW, lr=5e-4, weight_decay=1e-4
- Scheduler: cosine annealing, 100 epochs, patience=20
- Normalisation: per-feature standardisation (Welford online stats)

## Dependencies
- PyTorch ≥ 2.0
- NumPy
- scikit-learn
"""


# ─── Main exporter class ──────────────────────────────────────────────────────

class GRAMExporter:
    """
    Pipeline node: export a trained GFF as a GRaM competition submission.

    Hyperparams
    -----------
    gram_repo_dir  : str  – path to local clone of gram-competition/iclr-2026
    model_name     : str  – submission directory name (snake_case)
    team_name      : str  – team name for README
    create_pr      : bool – open a PR via `gh pr create` after commit
    auto_push      : bool – push branch even without creating a PR
    github_token   : str  – GITHUB_TOKEN env var value (optional)
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = hyperparams or {}
        self._gram_repo_dir = str(hp.get("gram_repo_dir", "./gram_repo"))
        self._model_name    = str(hp.get("model_name", "surromod_gff"))
        self._team_name     = str(hp.get("team_name", "SurroMod Team"))
        self._create_pr     = bool(hp.get("create_pr", False))
        self._auto_push     = bool(hp.get("auto_push", False))
        self._github_token  = str(hp.get("github_token", ""))

    # ── Public entry point ────────────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        model_artifact = inputs.get("model_artifact")
        if model_artifact is None:
            raise RuntimeError(
                "GRAMExporter: no model_artifact in inputs — connect this node "
                "downstream of a trained GraphFlowForecaster."
            )

        fe_pipeline = inputs.get("fe_pipeline") or []
        config = self._extract_config(model_artifact, fe_pipeline)

        gram_dir  = Path(self._gram_repo_dir)
        model_dir = gram_dir / "models" / self._model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        self._save_artifacts(model_artifact, config, model_dir)
        self._write_arch_py(model_dir)
        self._write_model_py(config, model_dir)
        self._write_readme(config, model_dir)
        self._update_init_py(gram_dir, config)

        pr_url = None
        if self._auto_push or self._create_pr:
            pr_url = self._git_submit(gram_dir)

        logger.info(
            "GRAMExporter: submission written to %s%s",
            model_dir,
            f"  PR: {pr_url}" if pr_url else "",
        )

        return {
            **inputs,
            "gram_export_status": "done",
            "gram_export_dir": str(model_dir),
            "gram_pr_url": pr_url,
        }

    # ── Config extraction ──────────────────────────────────────────────────────

    def _extract_config(self, gff: Any, fe_pipeline: list) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            # GFF architecture
            "latent_dim":        getattr(gff, "_latent_dim",        64),
            "hidden_dim":        getattr(gff, "_hidden_dim",        128),
            "num_mp_layers":     getattr(gff, "_num_mp_layers",     3),
            "aggregation_mode":  getattr(gff, "_aggregation_mode",  "mean"),
            "skip_connection_mode": getattr(gff, "_skip_connection_mode", "initial"),
            "use_temporal_decoder": getattr(gff, "_use_temporal_decoder", True),
            "point_feature_dim": getattr(gff, "_point_feature_dim", 35),
            "edge_dim":          getattr(gff, "_edge_dim",          4),
            "T_out":             getattr(gff, "_T_out",             5),
            "baseline_mode":     getattr(gff, "_baseline_mode",     "mean_field"),
            "dropout":           getattr(gff, "_dropout",           0.0),
            # FE defaults (overwritten below)
            "k":             8,
            "cutoff_freq":   0.2,
            "spectral_method": "fft",
            "normalizer_center": None,
            "normalizer_scale":  None,
        }

        for step in fe_pipeline:
            # SpatialGraphBuilder
            if hasattr(step, "_k"):
                cfg["k"] = int(step._k)
            # SpectralDecomposer
            if hasattr(step, "_cutoff_freq"):
                cfg["cutoff_freq"] = float(step._cutoff_freq)
            if hasattr(step, "_method") and hasattr(step, "_cutoff_freq"):
                cfg["spectral_method"] = str(step._method)
            # FeatureNormalizer
            if (hasattr(step, "_center") and hasattr(step, "_is_fitted")
                    and step._is_fitted and step._center is not None):
                cfg["normalizer_center"] = step._center.tolist()
                cfg["normalizer_scale"]  = step._scale.tolist()

        if cfg["normalizer_center"] is None:
            logger.warning(
                "GRAMExporter: FeatureNormalizer stats not found — "
                "export will use identity normalisation (center=0, scale=1)."
            )
            D = cfg["point_feature_dim"]
            cfg["normalizer_center"] = [0.0] * D
            cfg["normalizer_scale"]  = [1.0] * D

        return cfg

    # ── File writers ───────────────────────────────────────────────────────────

    def _save_artifacts(self, gff: Any, config: dict, model_dir: Path) -> None:
        import torch

        # weights.pt
        model = getattr(gff, "_model", None)
        if model is None or not getattr(gff, "_is_trained", False):
            raise RuntimeError("GRAMExporter: model is not trained yet.")
        weights_path = model_dir / "weights.pt"
        torch.save(model.state_dict(), weights_path)
        logger.info("GRAMExporter: weights saved → %s", weights_path)

        # config.pt  (serialise Python-native types for portability)
        cfg_serialisable = {}
        for k, v in config.items():
            if isinstance(v, np.ndarray):
                cfg_serialisable[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                cfg_serialisable[k] = v.item()
            else:
                cfg_serialisable[k] = v
        config_path = model_dir / "config.pt"
        torch.save(cfg_serialisable, config_path)
        logger.info("GRAMExporter: config saved → %s", config_path)

    def _write_arch_py(self, model_dir: Path) -> None:
        arch_src = _extract_arch_source()
        (model_dir / "arch.py").write_text(arch_src, encoding="utf-8")

    def _write_model_py(self, config: dict, model_dir: Path) -> None:
        class_name = _make_class_name(self._model_name)
        src = _MODEL_PY_TEMPLATE.format(
            model_name    = self._model_name,
            team_name     = self._team_name,
            class_name    = class_name,
            cutoff_freq   = config["cutoff_freq"],
            latent_dim    = config["latent_dim"],
            hidden_dim    = config["hidden_dim"],
            num_mp_layers = config["num_mp_layers"],
        )
        (model_dir / "model.py").write_text(src, encoding="utf-8")

    def _write_readme(self, config: dict, model_dir: Path) -> None:
        content = _README_TEMPLATE.format(
            model_name    = self._model_name,
            team_name     = self._team_name,
            latent_dim    = config["latent_dim"],
            hidden_dim    = config["hidden_dim"],
            num_mp_layers = config["num_mp_layers"],
            k             = config["k"],
            cutoff_freq   = config["cutoff_freq"],
        )
        (model_dir / "README.md").write_text(content, encoding="utf-8")

    def _update_init_py(self, gram_dir: Path, config: dict) -> None:
        init_path = gram_dir / "models" / "__init__.py"
        class_name = _make_class_name(self._model_name)
        import_line = (
            f"from models.{self._model_name}.model import {class_name}  "
            f"# SurroMod GFF submission\n"
        )
        if not init_path.exists():
            init_path.write_text(import_line, encoding="utf-8")
            return

        existing = init_path.read_text(encoding="utf-8")
        # Avoid duplicate entries
        if self._model_name in existing:
            pattern = rf"^from models\.{re.escape(self._model_name)}\.model.*\n?"
            existing = re.sub(pattern, "", existing, flags=re.MULTILINE)
        init_path.write_text(existing.rstrip("\n") + "\n" + import_line, encoding="utf-8")
        logger.info("GRAMExporter: updated models/__init__.py")

    # ── Git / PR ───────────────────────────────────────────────────────────────

    def _git_submit(self, gram_dir: Path) -> str | None:
        """Commit the submission and (optionally) push + open a PR."""
        if not gram_dir.exists():
            logger.warning("GRAMExporter: gram_repo_dir %s does not exist �� skipping git.", gram_dir)
            return None

        env = os.environ.copy()
        if self._github_token:
            env["GITHUB_TOKEN"] = self._github_token

        branch = f"submission/{self._model_name}"
        try:
            # Create branch (ignore if already exists)
            _run(["git", "checkout", "-B", branch], gram_dir)

            _run(["git", "add",
                  f"models/{self._model_name}/",
                  "models/__init__.py"],
                 gram_dir)

            _run(["git", "commit", "-m",
                  f"Add {self._model_name} submission"],
                 gram_dir)
            logger.info("GRAMExporter: committed on branch %s", branch)

            if self._auto_push or self._create_pr:
                _run(["git", "push", "--force-with-lease", "origin", branch], gram_dir)
                logger.info("GRAMExporter: pushed branch %s", branch)

            if self._create_pr:
                res = _run([
                    "gh", "pr", "create",
                    "--title", f"[Submission] {self._model_name}",
                    "--body", (
                        f"GRaM ICLR-2026 competition submission.\n\n"
                        f"**Team:** {self._team_name}\n"
                        f"**Model:** {self._model_name}\n"
                        f"**Architecture:** GFF latent={gram_dir}, "
                        f"auto-generated by SurroMod."
                    ),
                    "--base", "main",
                    "--head", branch,
                ], gram_dir, check=False)

                pr_url = res.stdout.strip().split("\n")[-1]
                logger.info("GRAMExporter: PR created → %s", pr_url)
                return pr_url

        except subprocess.CalledProcessError as e:
            logger.error(
                "GRAMExporter: git operation failed:\n%s\n%s",
                e.stdout, e.stderr,
            )

        return None
