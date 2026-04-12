"""
Code Generator
==============
Traverses a SurroMod pipeline graph backwards from a CodeExporter node and
emits a standalone ``train.py`` that replicates the pipeline without the GUI.

Two code paths
--------------
GFF path   – model == "GraphFlowForecaster": full standalone script with all
             model architecture classes inline, NPZ data loading, FE helpers,
             incremental normalizer, and a complete training loop.

Classic    – MLP / sklearn regressors and classifiers: sequential X/y
             transformation chain ending in a fit() call.
"""

from __future__ import annotations

import textwrap
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Graph utilities
# ─────────────────────────────────────────────────────────────────────────────

def _topo_sort(nodes: list[dict], edges: list[dict]) -> list[str]:
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        children[e["source"]].append(e["target"])
        in_degree.setdefault(e["target"], 0)
        in_degree[e["target"]] += 1
    queue = deque(nid for nid, d in in_degree.items() if d == 0)
    order: list[str] = []
    while queue:
        nid = queue.popleft()
        order.append(nid)
        for child in children.get(nid, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    return order


def _find_connected_regressor(
    nodes: list[dict], edges: list[dict], exporter_node_id: str
) -> dict | None:
    """Return the regressor/classifier node whose output connects to the exporter."""
    src_ids = {e["source"] for e in edges if e["target"] == exporter_node_id}
    node_map = {n["id"]: n for n in nodes}
    for sid in src_ids:
        nd = node_map.get(sid)
        if nd and nd.get("data", {}).get("category") in ("regressor", "classifier"):
            return nd
    return None


def _collect_upstream(all_nodes: list[dict], all_edges: list[dict], start_id: str) -> list[dict]:
    """BFS backwards from start_id; returns all ancestor nodes including start."""
    parents: dict[str, list[str]] = defaultdict(list)
    for e in all_edges:
        parents[e["target"]].append(e["source"])
    node_map = {n["id"]: n for n in all_nodes}
    visited: set[str] = set()
    queue: deque[str] = deque([start_id])
    result: list[dict] = []
    while queue:
        nid = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        if nid in node_map:
            result.append(node_map[nid])
        for p in parents.get(nid, []):
            if p not in visited:
                queue.append(p)
    return result


def _node_data(node: dict) -> dict:
    return node.get("data", node)


def _hp(node_data: dict) -> dict:
    return node_data.get("hyperparams", {})


# ─────────────────────────────────────────────────────────────────────────────
# GFF inline model classes (verbatim PyTorch – no SurroMod deps)
# ─────────────────────────────────────────────────────────────────────────────

_GFF_MODEL_CLASSES = '''\
# =============================================================================
# Model Architecture  (PointEncoder → GraphMP stack → LSTM decoder → head)
# =============================================================================

class PointEncoder(nn.Module):
    """MLP: per-point features → latent space."""
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphMessagePassingBlock(nn.Module):
    """One round of message passing with mean / max / attention aggregation."""
    def __init__(self, latent_dim: int, hidden_dim: int, edge_dim: int = 0,
                 dropout: float = 0.0, aggregation_mode: str = "mean"):
        super().__init__()
        self.aggregation_mode = aggregation_mode
        msg_in = 2 * latent_dim + edge_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(msg_in, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        if aggregation_mode == "attention":
            self.attn_gate = nn.Linear(latent_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        src, dst = edge_index
        msg_input = torch.cat(
            [h[src], h[dst]] + ([edge_attr] if edge_attr is not None else []), dim=-1
        )
        messages = self.message_mlp(msg_input)
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
            agg = torch.where(agg == float("-inf"), torch.zeros_like(agg), agg)
        elif self.aggregation_mode == "attention":
            logits = self.attn_gate(messages).squeeze(-1)
            logit_max = torch.full((N,), float("-inf"), device=h.device, dtype=h.dtype)
            logit_max.index_reduce_(0, dst, logits, "amax", include_self=True)
            shifted = logits - logit_max[dst].clamp(min=-1e9)
            exp_l = shifted.exp()
            sum_exp = torch.zeros(N, device=h.device, dtype=h.dtype)
            sum_exp.index_add_(0, dst, exp_l)
            alpha = exp_l / (sum_exp[dst] + 1e-9)
            agg = torch.zeros(N, messages.size(1), device=h.device, dtype=h.dtype)
            agg.index_add_(0, dst, messages * alpha.unsqueeze(-1))
        else:
            raise ValueError(f"Unknown aggregation_mode: {self.aggregation_mode}")
        return self.norm(h + self.update_mlp(torch.cat([h, agg], dim=-1)))


class TemporalLSTMDecoder(nn.Module):
    """Per-point LSTM unrolled over T_out future steps."""
    def __init__(self, latent_dim: int, hidden_dim: int, T_out: int):
        super().__init__()
        self.T_out = T_out
        self.lstm = nn.LSTMCell(latent_dim, hidden_dim)
        self.init_h = nn.Linear(latent_dim, hidden_dim)
        self.init_c = nn.Linear(latent_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h, c, inp = self.init_h(z), self.init_c(z), z
        outputs = []
        for _ in range(self.T_out):
            h, c = self.lstm(inp, (h, c))
            out_t = self.out_proj(h)
            outputs.append(out_t)
            inp = out_t
        return torch.stack(outputs, dim=0)   # (T_out, N, latent_dim)


class ResidualVelocityHead(nn.Module):
    """MLP: latent → delta velocity (3-D)."""
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3),
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)   # (T_out, N, 3)


class GraphFlowForecasterModel(nn.Module):
    """
    End-to-end: PointEncoder → MP-stack → TemporalLSTMDecoder → ResidualHead.
    Predicts *delta* velocity (add to baseline to get absolute prediction).
    """
    def __init__(self, point_feature_dim: int, latent_dim: int = 64,
                 hidden_dim: int = 128, edge_dim: int = 0, num_mp_layers: int = 3,
                 T_out: int = 5, dropout: float = 0.0, use_edge_attr: bool = True,
                 aggregation_mode: str = "mean", skip_connection_mode: str = "none",
                 use_temporal_decoder: bool = True):
        super().__init__()
        self.skip_connection_mode = skip_connection_mode
        self.use_edge_attr = use_edge_attr
        act_edge_dim = edge_dim if use_edge_attr else 0
        self.encoder = PointEncoder(point_feature_dim, latent_dim, hidden_dim, dropout)
        self.mp_layers = nn.ModuleList([
            GraphMessagePassingBlock(latent_dim, hidden_dim, act_edge_dim, dropout, aggregation_mode)
            for _ in range(num_mp_layers)
        ])
        if skip_connection_mode == "initial":
            self.skip_norm = nn.LayerNorm(latent_dim)
        if use_temporal_decoder:
            self.decoder = TemporalLSTMDecoder(latent_dim, hidden_dim, T_out)
        else:
            self.decoder = type("DirectDecoder", (), {
                "T_out": T_out,
                "__call__": lambda self, z: z.unsqueeze(0).expand(self.T_out, -1, -1),
            })()
        self.head = ResidualVelocityHead(latent_dim, hidden_dim)

    def forward(self, point_features: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(point_features)
        ea = edge_attr if self.use_edge_attr else None
        if self.skip_connection_mode == "none":
            for mp in self.mp_layers:
                h = mp(h, edge_index, ea)
        elif self.skip_connection_mode == "initial":
            h0 = h
            for mp in self.mp_layers:
                h = mp(h, edge_index, ea)
            h = self.skip_norm(h + h0)
        else:
            for mp in self.mp_layers:
                h = mp(h, edge_index, ea)
        return self.head(self.decoder(h))   # (T_out, N, 3)
'''


# ─────────────────────────────────────────────────────────────────────────────
# GFF script generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_gff_script(
    subgraph_nodes: list[dict],
    all_edges: list[dict],
    regressor_nid: str,
    output_dir: str,
) -> str:
    """Build a fully standalone train.py for a GFF pipeline."""

    # ── collect node hyperparams by method / category ─────────────────────
    node_map = {n["id"]: _node_data(n) for n in subgraph_nodes}

    input_data: dict = {}
    fe: dict[str, dict] = {}  # method → node_data
    ds_data: dict = {}        # DatasetSplit node

    for nd in subgraph_nodes:
        d = _node_data(nd)
        cat = d.get("category", "")
        method = d.get("method", "")
        if cat == "input":
            input_data = d
        elif cat == "feature_engineering":
            if method == "DatasetSplit":
                ds_data = d
            else:
                fe[method] = d

    reg_data = node_map.get(regressor_nid, {})

    # ── hyperparams with safe defaults ───────────────────────────────────
    hp_in  = _hp(input_data)
    hp_sgb = _hp(fe.get("SpatialGraphBuilder", {}))
    hp_sd  = _hp(fe.get("SpectralDecomposer", {}))
    hp_sdf = _hp(fe.get("SurfaceDistanceFeature", {}))
    hp_pff = _hp(fe.get("PointFeatureFusion", {}))
    hp_fn  = _hp(fe.get("FeatureNormalizer", {}))
    hp_ds  = _hp(ds_data)
    hp_reg = _hp(reg_data)

    batch_dir        = hp_in.get("batch_dir", "./data")
    max_sims         = int(hp_in.get("max_simulations", 0))
    max_points       = int(hp_in.get("max_points", 0))
    geometry_filter  = hp_in.get("geometry_filter", [])
    geo_filter_repr  = repr(list(geometry_filter) if isinstance(geometry_filter, (list, tuple)) else [])

    knn_k            = int(hp_sgb.get("k", 8))
    knn_mode         = str(hp_sgb.get("graph_mode", "knn"))
    knn_radius       = float(hp_sgb.get("radius", 0.1))
    knn_max_nb       = int(hp_sgb.get("max_neighbors", 32))
    incl_disp        = bool(hp_sgb.get("include_relative_displacement", True))
    incl_dist        = bool(hp_sgb.get("include_distance", True))
    norm_edge        = bool(hp_sgb.get("normalize_edge_attr", False))

    has_spectral     = "SpectralDecomposer" in fe
    spectral_method  = str(hp_sd.get("spectral_method", hp_sd.get("method", "fft")))
    cutoff_freq      = float(hp_sd.get("cutoff_freq", 0.2))

    has_sdf          = "SurfaceDistanceFeature" in fe
    return_vec       = bool(hp_sdf.get("return_vector", False))
    norm_dist        = bool(hp_sdf.get("normalize_distance", False))

    has_pff          = "PointFeatureFusion" in fe
    incl_pos         = bool(hp_pff.get("include_pos", True))
    incl_vel_hist    = bool(hp_pff.get("include_velocity_history", True))
    incl_geo_mask    = bool(hp_pff.get("include_geometry_mask", True))
    incl_dist_surf   = bool(hp_pff.get("include_dist_to_surface", True))
    incl_surf_vec    = bool(hp_pff.get("include_nearest_surface_vec", False))
    incl_pressure    = bool(hp_pff.get("include_pressure", False))
    incl_low_freq    = bool(hp_pff.get("include_low_freq", False))
    incl_high_freq   = bool(hp_pff.get("include_high_freq", False))

    has_fn           = "FeatureNormalizer" in fe
    norm_mode        = str(hp_fn.get("normalizer_mode", hp_fn.get("mode", "standard")))
    norm_eps         = float(hp_fn.get("epsilon", 1e-8))

    train_ratio      = float(hp_ds.get("train_ratio", 0.7))
    val_ratio        = float(hp_ds.get("val_ratio", 0.15))
    test_ratio       = float(hp_ds.get("test_ratio", 0.15))
    rand_seed        = int(hp_ds.get("random_seed", 42))

    latent_dim       = int(hp_reg.get("latent_dim", 64))
    hidden_dim       = int(hp_reg.get("hidden_dim", 128))
    num_mp           = int(hp_reg.get("num_message_passing_layers", hp_reg.get("num_mp_layers", 3)))
    dropout          = float(hp_reg.get("dropout", 0.1))
    use_edge_attr    = bool(hp_reg.get("use_edge_attr", True))
    baseline_mode    = str(hp_reg.get("baseline_mode", "mean_field"))
    agg_mode         = str(hp_reg.get("aggregation_mode", "mean"))
    skip_mode        = str(hp_reg.get("skip_connection_mode", "none"))
    use_lstm         = bool(hp_reg.get("use_temporal_decoder", True))
    feed_baseline    = bool(hp_reg.get("feed_baseline_as_feature", False))
    phys_weight      = float(hp_reg.get("physics_loss_weight", 0.0))

    lr               = float(hp_reg.get("learning_rate", 5e-4))
    num_epochs       = int(hp_reg.get("num_epochs", 100))
    optimizer_name   = str(hp_reg.get("optimizer", "AdamW"))
    weight_decay     = float(hp_reg.get("weight_decay", 1e-4))
    scheduler_name   = str(hp_reg.get("scheduler", "cosine"))
    patience         = int(hp_reg.get("early_stopping_patience", 20))

    reg_label        = reg_data.get("label", "GFF")
    timestamp        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── build the script ─────────────────────────────────────────────────
    lines: list[str] = []

    lines.append(f'''\
#!/usr/bin/env python3
"""
train.py  —  Generated by SurroMod
Regressor : {reg_label} (GraphFlowForecaster)
Generated : {timestamp}

Standalone script — no SurroMod installation required.
Dependencies:
    pip install torch numpy scikit-learn
"""

from __future__ import annotations
import copy, json, logging, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration  (baked in from SurroMod canvas)
# =============================================================================

# ── Data loading ──────────────────────────────────────────────────────────────
BATCH_DIR        = r"{batch_dir}"
MAX_SIMULATIONS  = {max_sims}        # 0 = all simulations
MAX_POINTS       = {max_points}      # 0 = no subsampling
GEOMETRY_FILTER  = {geo_filter_repr} # [] = all geometries
RANDOM_SEED      = {rand_seed}

# ── Dataset split ─────────────────────────────────────────────────────────────
TRAIN_RATIO      = {train_ratio}
VAL_RATIO        = {val_ratio}
TEST_RATIO       = {test_ratio}

# ── Graph construction (SpatialGraphBuilder) ──────────────────────────────────
GRAPH_MODE       = "{knn_mode}"      # "knn" | "radius"
KNN_K            = {knn_k}           # neighbours in knn mode
KNN_RADIUS       = {knn_radius}      # radius in radius mode
KNN_MAX_NBRS     = {knn_max_nb}      # cap per point in radius mode
INCL_DISPLACEMENT= {incl_disp}
INCL_DISTANCE    = {incl_dist}
NORMALIZE_EDGES  = {norm_edge}

# ── Spectral decomposer ───────────────────────────────────────────────────────
USE_SPECTRAL     = {has_spectral}
SPECTRAL_METHOD  = "{spectral_method}"
CUTOFF_FREQ      = {cutoff_freq}

# ── Surface distance features ─────────────────────────────────────────────────
USE_SURF_DIST    = {has_sdf}
SURF_DIST_NORM   = {norm_dist}
SURF_VEC         = {return_vec}

# ── Point feature fusion ──────────────────────────────────────────────────────
INCL_POS         = {incl_pos}
INCL_VEL_HIST    = {incl_vel_hist}
INCL_GEO_MASK    = {incl_geo_mask}
INCL_DIST_SURF   = {incl_dist_surf}
INCL_SURF_VEC    = {incl_surf_vec}
INCL_PRESSURE    = {incl_pressure}
INCL_LOW_FREQ    = {incl_low_freq}
INCL_HIGH_FREQ   = {incl_high_freq}

# ── Feature normalizer ────────────────────────────────────────────────────────
USE_NORMALIZER   = {has_fn}
NORM_MODE        = "{norm_mode}"     # "standard" | "minmax" | "robust"
NORM_EPSILON     = {norm_eps}

# ── Model (GraphFlowForecaster) ───────────────────────────────────────────────
LATENT_DIM       = {latent_dim}
HIDDEN_DIM       = {hidden_dim}
NUM_MP_LAYERS    = {num_mp}
DROPOUT          = {dropout}
USE_EDGE_ATTR    = {use_edge_attr}
BASELINE_MODE    = "{baseline_mode}" # "persistence"|"mean_field"|"linear_extrapolation"|"none"
AGGREGATION_MODE = "{agg_mode}"      # "mean" | "max" | "attention"
SKIP_MODE        = "{skip_mode}"     # "none" | "initial"
USE_LSTM_DECODER = {use_lstm}
FEED_BASELINE_FEAT = {feed_baseline}
PHYSICS_LOSS_W   = {phys_weight}

# ── Training ──────────────────────────────────────────────────────────────────
LEARNING_RATE    = {lr}
NUM_EPOCHS       = {num_epochs}
OPTIMIZER        = "{optimizer_name}"
WEIGHT_DECAY     = {weight_decay}
SCHEDULER        = "{scheduler_name}" # "cosine" | "step" | "plateau" | "none"
PATIENCE         = {patience}

OUTPUT_DIR       = "{output_dir}"
''')

    lines.append('''\
# =============================================================================
# Data Loading
# =============================================================================

def _parse_npz_filename(name: str) -> tuple[str, str, str]:
    """Parse "{geo}_{sample}-{subset}.npz" → (geo_id, sample_id, subset)."""
    stem = name.replace(".npz", "")
    parts = stem.split("_", 1)
    geo_id = parts[0]
    rest = parts[1] if len(parts) > 1 else stem
    if "-" in rest:
        sample_id, subset = rest.rsplit("-", 1)
    else:
        sample_id, subset = rest, "unknown"
    return geo_id, sample_id, subset


def load_dataset(
    batch_dir: str,
    max_simulations: int = 0,
    max_points: int = 0,
    geometry_filter: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """
    Scan *batch_dir* for .npz files and return lightweight sample descriptors.
    No arrays are loaded; downstream code calls load_npz() on demand.

    Returns
    -------
    dict with keys:
        samples      – list of metadata dicts (filepath, sim_id, geo_id, ...)
        sim_ids      – unique simulation identifiers
        num_samples  – total number of temporal windows
    """
    bd = Path(batch_dir)
    if not bd.is_absolute():
        bd = Path(__file__).parent / batch_dir
    if not bd.is_dir():
        raise FileNotFoundError(f"Batch directory not found: {bd}")

    npz_files = sorted(bd.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {bd}")

    if geometry_filter:
        geo_set = set(geometry_filter)
        npz_files = [f for f in npz_files if _parse_npz_filename(f.name)[0] in geo_set]

    # Group by simulation
    sim_files: dict[str, list[Path]] = {}
    for fp in npz_files:
        geo_id, sample_id, _ = _parse_npz_filename(fp.name)
        sim_id = f"{geo_id}_{sample_id}"
        sim_files.setdefault(sim_id, []).append(fp)

    all_sim_ids = list(sim_files.keys())
    if max_simulations > 0:
        all_sim_ids = all_sim_ids[:max_simulations]

    samples = []
    for sid in all_sim_ids:
        for fp in sorted(sim_files[sid]):
            geo_id, sample_id, subset = _parse_npz_filename(fp.name)
            samples.append({
                "filepath": str(fp),
                "sim_id": sid,
                "geo_id": geo_id,
                "sample_id": sample_id,
                "subset": subset,
            })

    logger.info("load_dataset: %d simulations, %d temporal windows", len(all_sim_ids), len(samples))
    return {"samples": samples, "sim_ids": all_sim_ids, "num_samples": len(samples)}


def load_npz(filepath: str, max_points: int = 0, seed: int = 42) -> dict:
    """Load a single NPZ sample; optionally subsample to *max_points*."""
    with np.load(filepath, allow_pickle=False) as npz:
        raw = {k: npz[k] for k in npz.files}
    out = {k: np.asarray(v, dtype=np.float32) if v.dtype.kind == "f" else v
           for k, v in raw.items()}
    if max_points > 0 and out.get("pos", np.empty(0)).shape[0] > max_points:
        rng = np.random.RandomState(seed)
        N = out["pos"].shape[0]
        idcs_af = out.get("idcs_airfoil")
        if idcs_af is not None and len(idcs_af) > 0:
            n_af = min(len(idcs_af), max_points // 2)
            af_keep = rng.choice(idcs_af, n_af, replace=False)
            field_mask = np.ones(N, bool)
            field_mask[idcs_af] = False
            field_idx = np.where(field_mask)[0]
            fk = rng.choice(field_idx, min(max_points - n_af, len(field_idx)), replace=False)
            keep = np.sort(np.concatenate([af_keep, fk]))
        else:
            keep = np.sort(rng.choice(N, max_points, replace=False))
        out["pos"] = out["pos"][keep]
        if "velocity_in"  in out: out["velocity_in"]  = out["velocity_in"][:, keep, :]
        if "velocity_out" in out: out["velocity_out"] = out["velocity_out"][:, keep, :]
        if "pressure"     in out: out["pressure"]     = out["pressure"][:, keep]
        if idcs_af is not None and len(idcs_af) > 0:
            old2new = {int(o): i for i, o in enumerate(keep)}
            out["idcs_airfoil"] = np.array(
                [old2new[int(i)] for i in af_keep if int(i) in old2new], dtype=np.int32
            )
    return out
''')

    lines.append('''\
# =============================================================================
# Feature Engineering Helpers
# =============================================================================

def build_graph(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Build k-NN or radius graph from (N,3) positions.
    Returns edge_index (2,E) and edge_attr (E, D) or None."""
    N = pos.shape[0]
    if GRAPH_MODE == "knn":
        k_act = min(KNN_K + 1, N)
        nn_ = NearestNeighbors(n_neighbors=k_act, algorithm="auto").fit(pos)
        _, indices = nn_.kneighbors(pos)
        src, tgt = [], []
        for i in range(N):
            for j in range(k_act):
                nb = indices[i, j]
                if nb != i:
                    src.append(i); tgt.append(nb)
        edge_index = np.array([src, tgt], dtype=np.int64)
    else:  # radius
        nn_ = NearestNeighbors(radius=KNN_RADIUS, algorithm="auto").fit(pos)
        _, indices = nn_.radius_neighbors(pos, sort_results=True)
        src, tgt = [], []
        for i in range(N):
            cnt = 0
            for nb in indices[i]:
                if nb != i:
                    if cnt >= KNN_MAX_NBRS: break
                    src.append(i); tgt.append(nb); cnt += 1
        edge_index = np.zeros((2, 0), dtype=np.int64) if not src else np.array([src, tgt], dtype=np.int64)

    if not (INCL_DISPLACEMENT or INCL_DISTANCE) or edge_index.shape[1] == 0:
        return edge_index, None

    s_idx, t_idx = edge_index[0], edge_index[1]
    diff = pos[t_idx] - pos[s_idx]
    parts = []
    if INCL_DISPLACEMENT: parts.append(diff)
    if INCL_DISTANCE:     parts.append(np.linalg.norm(diff, axis=1, keepdims=True).astype(np.float32))
    ea = np.concatenate(parts, axis=1).astype(np.float32)

    if NORMALIZE_EDGES and ea.shape[0] > 0:
        if INCL_DISPLACEMENT and INCL_DISTANCE:
            med = float(np.median(ea[:, 3:4]))
            if med > 1e-12:
                ea[:, :3] /= med
                ea[:, 3:4] /= (ea[:, 3:4].max() + 1e-12)
        elif INCL_DISPLACEMENT:
            med = float(np.median(np.linalg.norm(ea, axis=1)))
            if med > 1e-12: ea /= med
        elif INCL_DISTANCE:
            mx = float(ea.max())
            if mx > 1e-12: ea /= mx
    return edge_index, ea


def spectral_fft(velocity_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FFT decompose (T_in, N, 3) → low-freq, high-freq both (T_in, N, 3)."""
    T = velocity_in.shape[0]
    cutoff = max(1, int(CUTOFF_FREQ * (T // 2 + 1)))
    spec = np.fft.rfft(velocity_in, axis=0)
    spec_low = np.zeros_like(spec); spec_low[:cutoff] = spec[:cutoff]
    spec_high = spec.copy();        spec_high[:cutoff] = 0.0
    low  = np.fft.irfft(spec_low,  n=T, axis=0).astype(np.float32)
    high = np.fft.irfft(spec_high, n=T, axis=0).astype(np.float32)
    return low, high


def surface_distance(pos: np.ndarray, idcs_airfoil: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Returns geometry_mask (N,1), dist_to_surface (N,1), nearest_surface_vec (N,3) or None."""
    N = pos.shape[0]
    geo_mask = np.zeros((N, 1), dtype=np.float32)
    if idcs_airfoil is None or len(idcs_airfoil) == 0:
        return geo_mask, np.zeros((N, 1), dtype=np.float32), None
    geo_mask[idcs_airfoil, 0] = 1.0
    surf_pts = pos[idcs_airfoil]
    nn_ = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(surf_pts)
    dists, idcs = nn_.kneighbors(pos)
    dist_arr = dists.astype(np.float32)
    if SURF_DIST_NORM and dist_arr.max() > 0:
        dist_arr /= dist_arr.max()
    vec = None
    if SURF_VEC:
        vec = (surf_pts[idcs.ravel()] - pos).astype(np.float32)
    return geo_mask, dist_arr, vec


def fuse_features(data: dict) -> np.ndarray:
    """Concatenate enabled per-point features → (N, D)."""
    vel_in = np.asarray(data["velocity_in"], dtype=np.float32)  # (T_in, N, 3)
    T_in, N, _ = vel_in.shape
    parts = []
    if INCL_POS and "pos" in data:
        parts.append(np.asarray(data["pos"], dtype=np.float32))
    if INCL_VEL_HIST:
        parts.append(vel_in.transpose(1, 0, 2).reshape(N, T_in * 3))
    if INCL_GEO_MASK and "geometry_mask" in data:
        parts.append(np.asarray(data["geometry_mask"], dtype=np.float32))
    if INCL_DIST_SURF and "dist_to_surface" in data:
        parts.append(np.asarray(data["dist_to_surface"], dtype=np.float32))
    if INCL_SURF_VEC and "nearest_surface_vec" in data:
        parts.append(np.asarray(data["nearest_surface_vec"], dtype=np.float32))
    if INCL_PRESSURE and "pressure" in data:
        p = np.asarray(data["pressure"], dtype=np.float32)
        parts.append(p.T if p.shape[1] == N else p)
    if INCL_LOW_FREQ and "vel_low_freq" in data:
        lf = np.asarray(data["vel_low_freq"], dtype=np.float32)
        parts.append(lf.transpose(1, 0, 2).reshape(N, T_in * 3))
    if INCL_HIGH_FREQ and "vel_high_freq" in data:
        hf = np.asarray(data["vel_high_freq"], dtype=np.float32)
        parts.append(hf.transpose(1, 0, 2).reshape(N, T_in * 3))
    if not parts:
        raise ValueError("fuse_features: no features enabled or available.")
    return np.concatenate(parts, axis=1).astype(np.float32)


def baseline_extrapolate(velocity_in: np.ndarray, T_out: int) -> np.ndarray:
    """Compute baseline future velocity prediction.  Shape (T_out, N, 3)."""
    if BASELINE_MODE == "persistence":
        return np.tile(velocity_in[-1:], (T_out, 1, 1))
    if BASELINE_MODE == "mean_field":
        return np.tile(velocity_in.mean(axis=0, keepdims=True), (T_out, 1, 1))
    if BASELINE_MODE == "linear_extrapolation":
        if velocity_in.shape[0] < 2:
            return np.tile(velocity_in[-1:], (T_out, 1, 1))
        slope = velocity_in[-1] - velocity_in[-2]
        return np.stack([velocity_in[-1] + slope * (t + 1) for t in range(T_out)])
    if BASELINE_MODE == "none":
        return np.zeros((T_out, *velocity_in.shape[1:]), dtype=np.float32)
    # fallback
    return np.tile(velocity_in[-1:], (T_out, 1, 1))
''')

    lines.append('''\
# =============================================================================
# Incremental Feature Normalizer
# =============================================================================

class IncrementalNormalizer:
    """Welford-based incremental normalizer — never holds all data in RAM."""
    def __init__(self, mode: str = "standard", epsilon: float = 1e-8):
        self.mode = mode
        self.epsilon = epsilon
        self._center: np.ndarray | None = None
        self._scale:  np.ndarray | None = None
        self.is_fitted = False

    # ── streaming fit API ────────────────────────────────────────────────
    def begin_fit(self) -> None:
        self._n = 0; self._D = None
        self._mean = self._M2 = None
        self._min = self._max = None
        self._reservoir: list[np.ndarray] = []; self._res_n = 0
        self._res_cap = 200_000

    def update_fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64); m = X.shape[0]; D = X.shape[1]
        if self._D is None:
            self._D = D
            if self.mode == "standard":
                self._mean = np.zeros(D, np.float64); self._M2 = np.zeros(D, np.float64)
            elif self.mode == "minmax":
                self._min = np.full(D, np.inf, np.float64)
                self._max = np.full(D, -np.inf, np.float64)
        if self.mode == "standard":
            bm = X.mean(0); bM2 = ((X - bm) ** 2).sum(0)
            d = bm - self._mean; n_new = self._n + m
            self._mean += d * (m / n_new)
            self._M2   += bM2 + d ** 2 * (self._n * m / n_new)
        elif self.mode == "minmax":
            self._min = np.minimum(self._min, X.min(0))
            self._max = np.maximum(self._max, X.max(0))
        else:
            if self._res_n < self._res_cap:
                keep = min(m, self._res_cap - self._res_n)
                self._reservoir.append(X[:keep].astype(np.float32)); self._res_n += keep
        self._n += m

    def finalize_fit(self) -> None:
        if self._n == 0: return
        if self.mode == "standard":
            self._center = self._mean.astype(np.float32)
            self._scale  = np.sqrt(self._M2 / max(self._n, 1)).astype(np.float32)
        elif self.mode == "minmax":
            self._center = self._min.astype(np.float32)
            self._scale  = (self._max - self._min).astype(np.float32)
        else:
            combined = np.concatenate(self._reservoir, 0)
            self._center = np.median(combined, 0).astype(np.float32)
            self._scale  = (np.percentile(combined, 75, 0) - np.percentile(combined, 25, 0)).astype(np.float32)
        self._scale = np.where(self._scale < self.epsilon, 1.0, self._scale).astype(np.float32)
        self.is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call begin_fit / update_fit / finalize_fit first.")
        return ((X - self._center) / self._scale).astype(np.float32)
''')

    lines.append(_GFF_MODEL_CLASSES)

    lines.append('''\
# =============================================================================
# Per-sample preparation (load NPZ + full FE pipeline)
# =============================================================================

def prepare_sample(
    filepath: str,
    graph_cache: dict,
    normalizer: "IncrementalNormalizer | None",
    T_out_override: int = 5,
) -> dict:
    """Load one NPZ file and apply all FE steps.  Returns training-ready arrays."""
    data = load_npz(filepath, max_points=MAX_POINTS, seed=RANDOM_SEED)

    # ── Graph ──────────────────────────────────────────────────────────────
    sim_id = filepath  # use filepath as unique key (already sim-level in batch)
    # graph_cache may hold prebuilt graphs keyed by filepath or sim_id
    if filepath in graph_cache:
        edge_index, edge_attr = graph_cache[filepath]
    else:
        edge_index, edge_attr = build_graph(data["pos"])
        graph_cache[filepath] = (edge_index, edge_attr)

    data["edge_index"] = edge_index
    if edge_attr is not None:
        data["edge_attr"] = edge_attr

    # ── Spectral decomposition ─────────────────────────────────────────────
    if USE_SPECTRAL and "velocity_in" in data:
        low, high = spectral_fft(data["velocity_in"])
        data["vel_low_freq"]  = low
        data["vel_high_freq"] = high

    # ── Surface distance ───────────────────────────────────────────────────
    if USE_SURF_DIST and "pos" in data:
        geo_mask, dist, vec = surface_distance(data["pos"], data.get("idcs_airfoil"))
        data["geometry_mask"]   = geo_mask
        data["dist_to_surface"] = dist
        if vec is not None:
            data["nearest_surface_vec"] = vec

    # ── Feature fusion ─────────────────────────────────────────────────────
    pf = fuse_features(data)

    # ── Normalisation ──────────────────────────────────────────────────────
    if USE_NORMALIZER and normalizer is not None and normalizer.is_fitted:
        pf = normalizer.transform(pf)

    # ── Baseline + target delta ────────────────────────────────────────────
    vel_in  = np.asarray(data["velocity_in"],  dtype=np.float32)
    vel_out = np.asarray(data["velocity_out"], dtype=np.float32)
    T_out = vel_out.shape[0] if vel_out.ndim >= 3 else T_out_override
    baseline     = baseline_extrapolate(vel_in, T_out)
    target_delta = vel_out - baseline

    if FEED_BASELINE_FEAT:
        bf = baseline[0].reshape(pf.shape[0], 3)
        pf = np.concatenate([pf, bf], axis=1)

    return {
        "point_features": pf,
        "edge_index":     edge_index,
        "edge_attr":      edge_attr,
        "velocity_in":    vel_in,
        "velocity_out":   vel_out,
        "baseline":       baseline,
        "target_delta":   target_delta,
        "idcs_airfoil":   data.get("idcs_airfoil"),
    }
''')

    lines.append(f'''\
# =============================================================================
# Training
# =============================================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("SurroMod  –  GraphFlowForecaster  standalone training")
    logger.info("=" * 60)

    # ── Load dataset ────────────────────────────────────────────────────────
    ds = load_dataset(BATCH_DIR, MAX_SIMULATIONS, MAX_POINTS, GEOMETRY_FILTER, RANDOM_SEED)
    samples = ds["samples"]
    N = len(samples)
    logger.info("Total samples (temporal windows): %d", N)

    # ── Split ───────────────────────────────────────────────────────────────
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.permutation(N)
    n_train = max(1, int(N * TRAIN_RATIO))
    n_val   = max(0, int(N * VAL_RATIO))
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]
    logger.info("Split: %d train / %d val / %d test", len(train_idx), len(val_idx), len(test_idx))

    # ── Build graph cache (one graph per unique geometry) ───────────────────
    logger.info("Building graph cache for %d unique simulations…", len(ds["sim_ids"]))
    # Key by sim_id so windows sharing a mesh reuse the graph
    sim_graph_cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {{}}
    graph_cache: dict[str, tuple] = {{}}
    for info in samples:
        sid = info["sim_id"]
        if sid not in sim_graph_cache:
            raw = load_npz(info["filepath"], max_points=MAX_POINTS, seed=RANDOM_SEED)
            ei, ea = build_graph(raw["pos"])
            sim_graph_cache[sid] = (ei, ea)
            del raw
        graph_cache[info["filepath"]] = sim_graph_cache[sid]
    logger.info("Graph cache built (%d unique graphs)", len(sim_graph_cache))

    # ── Fit normalizer on training set ──────────────────────────────────────
    normalizer: IncrementalNormalizer | None = None
    if USE_NORMALIZER:
        normalizer = IncrementalNormalizer(NORM_MODE, NORM_EPSILON)
        logger.info("Fitting feature normalizer on %d training samples…", len(train_idx))
        normalizer.begin_fit()
        for ti in train_idx:
            sd = prepare_sample(samples[ti]["filepath"], dict(graph_cache), None)
            normalizer.update_fit(sd["point_features"])
            del sd
        normalizer.finalize_fit()
        logger.info("Normalizer fitted (mode=%s)", NORM_MODE)

    # ── Infer dims from first sample ────────────────────────────────────────
    first = prepare_sample(samples[train_idx[0]]["filepath"], dict(graph_cache), normalizer)
    point_feature_dim = first["point_features"].shape[1]
    edge_dim          = first["edge_attr"].shape[1] if first["edge_attr"] is not None else 0
    T_out             = first["target_delta"].shape[0]
    del first
    logger.info("Dims: point_feature_dim=%d  edge_dim=%d  T_out=%d", point_feature_dim, edge_dim, T_out)

    # ── Build model ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    model = GraphFlowForecasterModel(
        point_feature_dim = point_feature_dim,
        latent_dim        = LATENT_DIM,
        hidden_dim        = HIDDEN_DIM,
        edge_dim          = edge_dim if USE_EDGE_ATTR else 0,
        num_mp_layers     = NUM_MP_LAYERS,
        T_out             = T_out,
        dropout           = DROPOUT,
        use_edge_attr     = USE_EDGE_ATTR,
        aggregation_mode  = AGGREGATION_MODE,
        skip_connection_mode = SKIP_MODE,
        use_temporal_decoder = USE_LSTM_DECODER,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{{total_params:,}}")

    # ── Optimizer & scheduler ───────────────────────────────────────────────
    if OPTIMIZER == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = None
    if SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    elif SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    elif SCHEDULER == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # ── Training loop ───────────────────────────────────────────────────────
    best_loss        = float("inf")
    best_state       = None
    patience_counter = 0
    history          = []

    logger.info("Starting training for up to %d epochs…", NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_mae  = 0.0
        order = rng.permutation(len(train_idx))

        for si in order:
            fp  = samples[train_idx[si]]["filepath"]
            sd  = prepare_sample(fp, graph_cache, normalizer)
            pf  = torch.from_numpy(sd["point_features"]).to(device)
            ei  = torch.from_numpy(sd["edge_index"]).to(device)
            ea  = torch.from_numpy(sd["edge_attr"]).to(device) if sd["edge_attr"] is not None else None
            td  = torch.from_numpy(sd["target_delta"]).to(device)
            af  = sd.get("idcs_airfoil")
            bl  = sd["baseline"] if PHYSICS_LOSS_W > 0.0 else None
            del sd

            optimizer.zero_grad()
            pred = model(pf, ei, ea)
            loss = F.mse_loss(pred, td)
            if PHYSICS_LOSS_W > 0.0 and af is not None and len(af) > 0 and bl is not None:
                bl_af  = torch.from_numpy(bl[:, af, :]).to(device)
                loss   = loss + PHYSICS_LOSS_W * F.mse_loss(pred[:, af, :], -bl_af)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            with torch.no_grad():
                epoch_mae += F.l1_loss(pred, td).item()
            del pf, ei, ea, td, pred, loss

        avg_loss = epoch_loss / len(train_idx)
        avg_mae  = epoch_mae  / len(train_idx)
        entry    = {{"epoch": epoch, "train_loss": avg_loss, "train_mae": avg_mae}}

        # Validation pass
        monitor_loss = avg_loss
        if len(val_idx) > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for vi in val_idx:
                    vfp  = samples[vi]["filepath"]
                    vsd  = prepare_sample(vfp, graph_cache, normalizer)
                    vpf  = torch.from_numpy(vsd["point_features"]).to(device)
                    vei  = torch.from_numpy(vsd["edge_index"]).to(device)
                    vea  = torch.from_numpy(vsd["edge_attr"]).to(device) if vsd["edge_attr"] is not None else None
                    vtd  = torch.from_numpy(vsd["target_delta"]).to(device)
                    vp   = model(vpf, vei, vea)
                    val_loss += F.mse_loss(vp, vtd).item()
                    del vpf, vei, vea, vtd, vp, vsd
            val_loss /= len(val_idx)
            entry["val_loss"] = val_loss
            monitor_loss = val_loss

        history.append(entry)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_loss)
            else:
                scheduler.step()

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, PATIENCE)
                break

        if epoch % max(1, NUM_EPOCHS // 10) == 0 or epoch < 5:
            val_str = f"  val={{entry.get('val_loss', float('nan')):.6f}}" if len(val_idx) > 0 else ""
            logger.info("Epoch %3d/%d  train=%.6f  mae=%.6f%s", epoch, NUM_EPOCHS, avg_loss, avg_mae, val_str)

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("Training complete.  Best monitor loss: %.6f", best_loss)

    # ── Save ────────────────────────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(OUTPUT_DIR) / "gff_model.pt"
    torch.save({{
        "model_state_dict": model.state_dict(),
        "config": {{
            "point_feature_dim": point_feature_dim,
            "edge_dim":          edge_dim,
            "T_out":             T_out,
            "LATENT_DIM":        LATENT_DIM,
            "HIDDEN_DIM":        HIDDEN_DIM,
            "NUM_MP_LAYERS":     NUM_MP_LAYERS,
            "DROPOUT":           DROPOUT,
            "USE_EDGE_ATTR":     USE_EDGE_ATTR,
            "AGGREGATION_MODE":  AGGREGATION_MODE,
            "SKIP_MODE":         SKIP_MODE,
            "USE_LSTM_DECODER":  USE_LSTM_DECODER,
        }},
        "normalizer": {{
            "mode":    NORM_MODE,
            "center":  normalizer._center.tolist() if normalizer and normalizer.is_fitted else None,
            "scale":   normalizer._scale.tolist()  if normalizer and normalizer.is_fitted else None,
        }},
        "training_history": history,
        "best_epoch_loss":  best_loss,
    }}, str(out_path))
    logger.info("Model saved to %s", out_path)

    # Save history as JSON
    with open(Path(OUTPUT_DIR) / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s/training_history.json", OUTPUT_DIR)


if __name__ == "__main__":
    main()
''')

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Classic (MLP / sklearn) code generators  —  largely unchanged, improved
# ─────────────────────────────────────────────────────────────────────────────

def _upload_path(source: str) -> str:
    if not source:
        return "<PATH_TO_DATA_FILE>"
    if len(source.replace("-", "")) == 32 or (len(source) > 30 and "." in source and "/" not in source):
        return f"uploads/{source}"
    return source


def _gen_input(nd_data: dict, var: str) -> str:
    kind     = nd_data.get("inputKind", "scalar")
    source   = nd_data.get("source", "")
    features = nd_data.get("features", [])
    labels   = nd_data.get("labels", [])
    filepath = _upload_path(source)
    fname    = nd_data.get("fileName", source)

    if kind == "scalar":
        return (
            f"# ── Input: {nd_data.get('label', 'Scalar Data')} ──\n"
            f"import pandas as pd\n"
            f"_df_{var} = pd.read_csv(r\"{filepath}\")\n"
            f"{var}_X = _df_{var}[{repr(features)}].values.astype(float)\n"
            f"{var}_y = _df_{var}[{repr(labels)}].values.astype(float)\n"
            f"{var}_feature_names = {repr(features)}\n"
            f"{var}_label_names   = {repr(labels)}\n"
        )

    if kind == "3d_field":
        hp       = _hp(nd_data)
        batch_dir = hp.get("batch_dir", filepath)
        max_sims  = int(hp.get("max_simulations", 0))
        return (
            f"# ── Input: {nd_data.get('label', '3D Field')} ──\n"
            f"# NOTE: 3D-field data is handled by the GFF code path.\n"
            f"# If you see this comment the code exporter was connected to a\n"
            f"# non-GFF regressor.  Use GraphFlowForecaster for 3D field data.\n"
            f"BATCH_DIR       = r\"{batch_dir}\"\n"
            f"MAX_SIMULATIONS = {max_sims}\n"
            f"{var}_X = None  # replace with actual loading\n"
            f"{var}_y = None\n"
        )

    return (
        f"# ── Input: {nd_data.get('label', kind)} ──\n"
        f"# TODO: load {kind} data from r\"{filepath}\"\n"
        f"{var}_X = None\n"
        f"{var}_y = None\n"
    )


def _gen_feature_engineering(nd_data: dict, var: str, up_var: str) -> str:
    method = nd_data.get("method", "")
    hp     = _hp(nd_data)

    if method == "Scaler":
        mode = hp.get("method", "MinMax")
        if mode == "Standard":
            return (
                f"# ── FE: StandardScaler ──\n"
                f"from sklearn.preprocessing import StandardScaler\n"
                f"_{var}_sc = StandardScaler()\n"
                f"{var}_X = _{var}_sc.fit_transform({up_var}_X)\n"
                f"{var}_y = {up_var}_y\n"
                f"{var}_feature_names = {up_var}_feature_names\n"
                f"{var}_label_names   = {up_var}_label_names\n"
            )
        if mode == "LogTransform":
            return (
                f"# ── FE: Log Transform ──\n"
                f"import numpy as np\n"
                f"{var}_X = np.log1p({up_var}_X)\n"
                f"{var}_y = {up_var}_y\n"
                f"{var}_feature_names = {up_var}_feature_names\n"
                f"{var}_label_names   = {up_var}_label_names\n"
            )
        return (
            f"# ── FE: MinMaxScaler ──\n"
            f"from sklearn.preprocessing import MinMaxScaler\n"
            f"_{var}_sc = MinMaxScaler()\n"
            f"{var}_X = _{var}_sc.fit_transform({up_var}_X)\n"
            f"{var}_y = {up_var}_y\n"
            f"{var}_feature_names = {up_var}_feature_names\n"
            f"{var}_label_names   = {up_var}_label_names\n"
        )

    if method == "PCA":
        n = hp.get("n_components", 2)
        return (
            f"# ── FE: PCA (n={n}) ──\n"
            f"from sklearn.decomposition import PCA\n"
            f"_{var}_pca = PCA(n_components={n}, whiten={hp.get('whiten', False)})\n"
            f"{var}_X = _{var}_pca.fit_transform({up_var}_X)\n"
            f"{var}_y = {up_var}_y\n"
            f"{var}_feature_names = [f'PC{{i+1}}' for i in range({n})]\n"
            f"{var}_label_names   = {up_var}_label_names\n"
        )

    if method == "DatasetSplit":
        tr = hp.get("train_ratio", 0.7)
        vr = hp.get("val_ratio", 0.15)
        te = hp.get("test_ratio", 0.15)
        seed = hp.get("random_seed", 42)
        shuf = hp.get("shuffle", True)
        te_frac  = round(1.0 - tr, 4)
        val_frac = round(vr / (vr + te), 4) if (vr + te) > 0 else 0.5
        return (
            f"# ── FE: DatasetSplit (train={tr}, val={vr}, test={te}) ──\n"
            f"from sklearn.model_selection import train_test_split\n"
            f"{var}_X_tr, {var}_X_tmp, {var}_y_tr, {var}_y_tmp = train_test_split(\n"
            f"    {up_var}_X, {up_var}_y, test_size={te_frac}, random_state={seed}, shuffle={shuf})\n"
            f"{var}_X_val, {var}_X_te, {var}_y_val, {var}_y_te = train_test_split(\n"
            f"    {var}_X_tmp, {var}_y_tmp, test_size={val_frac}, random_state={seed}, shuffle={shuf})\n"
            f"{var}_X = {var}_X_tr\n"
            f"{var}_y = {var}_y_tr\n"
            f"{var}_feature_names = {up_var}_feature_names\n"
            f"{var}_label_names   = {up_var}_label_names\n"
        )

    if method == "FeatureNormalizer":
        mode = hp.get("normalizer_mode", hp.get("mode", "standard"))
        cls_map = {"standard": "StandardScaler", "minmax": "MinMaxScaler"}
        cls = cls_map.get(mode, "StandardScaler")
        imp = "from sklearn.preprocessing import " + cls
        return (
            f"# ── FE: FeatureNormalizer ({mode}) ──\n"
            f"{imp}\n"
            f"_{var}_norm = {cls}()\n"
            f"{var}_X = _{var}_norm.fit_transform({up_var}_X)\n"
            f"{var}_y = {up_var}_y\n"
            f"{var}_feature_names = {up_var}_feature_names\n"
            f"{var}_label_names   = {up_var}_label_names\n"
        )

    if method == "Autoencoder":
        latent  = hp.get("latent_dim", 16)
        hidden  = hp.get("hidden_layers", 2)
        neurons = hp.get("neurons_per_layer", 64)
        lr      = hp.get("learning_rate", 0.001)
        epochs  = hp.get("epochs", 100)
        batch   = hp.get("batch_size", 32)
        return (
            f"# ── FE: Autoencoder (latent={latent}) ──\n"
            f"import torch, torch.nn as _nn\n"
            f"from torch.utils.data import DataLoader, TensorDataset\n"
            f"class _AE_{var}(_nn.Module):\n"
            f"    def __init__(self, in_d, ld):\n"
            f"        super().__init__()\n"
            f"        hs = [{neurons}]*{hidden}\n"
            f"        enc = []; d = in_d\n"
            f"        for h in hs: enc += [_nn.Linear(d,h), _nn.ReLU()]; d=h\n"
            f"        enc.append(_nn.Linear(d, ld))\n"
            f"        self.encoder = _nn.Sequential(*enc)\n"
            f"        dec = []; d = ld\n"
            f"        for h in reversed(hs): dec += [_nn.Linear(d,h), _nn.ReLU()]; d=h\n"
            f"        dec.append(_nn.Linear(d, in_d))\n"
            f"        self.decoder = _nn.Sequential(*dec)\n"
            f"    def forward(self, x): return self.decoder(self.encoder(x))\n"
            f"import numpy as np\n"
            f"_{var}_t  = torch.tensor({up_var}_X, dtype=torch.float32)\n"
            f"_{var}_ae = _AE_{var}(_{var}_t.shape[1], {latent})\n"
            f"_{var}_opt = torch.optim.Adam(_{var}_ae.parameters(), lr={lr})\n"
            f"for _ep in range({epochs}):\n"
            f"    for (_b,) in DataLoader(TensorDataset(_{var}_t), batch_size={batch}, shuffle=True):\n"
            f"        _{var}_opt.zero_grad()\n"
            f"        _nn.MSELoss()(_{var}_ae(_b), _b).backward()\n"
            f"        _{var}_opt.step()\n"
            f"_{var}_ae.eval()\n"
            f"with torch.no_grad():\n"
            f"    {var}_X = _{var}_ae.encoder(_{var}_t).numpy()\n"
            f"{var}_y = {up_var}_y\n"
            f"{var}_feature_names = [f'AE{{i}}' for i in range({latent})]\n"
            f"{var}_label_names   = {up_var}_label_names\n"
        )

    # generic fallback
    return (
        f"# ── FE: {method} (pass-through – TODO implement) ──\n"
        f"{var}_X = {up_var}_X\n"
        f"{var}_y = {up_var}_y\n"
        f"{var}_feature_names = {up_var}_feature_names\n"
        f"{var}_label_names   = {up_var}_label_names\n"
    )


def _gen_regressor(nd_data: dict, var: str, up_var: str) -> str:
    model = nd_data.get("model", "MLP")
    hp    = _hp(nd_data)

    if model == "MLP":
        hidden  = int(hp.get("hidden_layers", 3))
        neurons = int(hp.get("neurons_per_layer", 64))
        act     = str(hp.get("activation", "ReLU"))
        lr      = float(hp.get("learning_rate", 0.001))
        epochs  = int(hp.get("epochs", 100))
        batch   = int(hp.get("batch_size", 32))
        drop    = float(hp.get("dropout", 0.0))
        act_map = {"ReLU": "nn.ReLU()", "Tanh": "nn.Tanh()", "Sigmoid": "nn.Sigmoid()",
                   "GELU": "nn.GELU()", "LeakyReLU": "nn.LeakyReLU()"}
        act_str = act_map.get(act, "nn.ReLU()")
        return (
            f"# ── Regressor: MLP ──\n"
            f"import torch, torch.nn as nn\n"
            f"from torch.utils.data import DataLoader, TensorDataset\n"
            f"import numpy as np\n"
            f"_{var}_in = {up_var}_X.shape[1]\n"
            f"_{var}_out = {up_var}_y.shape[1] if {up_var}_y.ndim > 1 else 1\n"
            f"_{var}_layers = []\n"
            f"_{var}_d = _{var}_in\n"
            f"for _ in range({hidden}):\n"
            f"    _{var}_layers += [nn.Linear(_{var}_d, {neurons}), {act_str}]\n"
            f"    if {drop} > 0: _{var}_layers.append(nn.Dropout({drop}))\n"
            f"    _{var}_d = {neurons}\n"
            f"_{var}_layers.append(nn.Linear(_{var}_d, _{var}_out))\n"
            f"{var}_model = nn.Sequential(*_{var}_layers)\n"
            f"_{var}_opt  = torch.optim.Adam({var}_model.parameters(), lr={lr})\n"
            f"_{var}_crit = nn.MSELoss()\n"
            f"_{var}_Xt = torch.tensor({up_var}_X, dtype=torch.float32)\n"
            f"_{var}_yt = torch.tensor({up_var}_y.reshape(-1,1) if {up_var}_y.ndim==1 else {up_var}_y, dtype=torch.float32)\n"
            f"_{var}_dl = DataLoader(TensorDataset(_{var}_Xt, _{var}_yt), batch_size={batch}, shuffle=True)\n"
            f"for _ep in range({epochs}):\n"
            f"    for _xb, _yb in _{var}_dl:\n"
            f"        _{var}_opt.zero_grad()\n"
            f"        _{var}_crit({var}_model(_xb), _yb).backward()\n"
            f"        _{var}_opt.step()\n"
            f"print(f\"MLP training complete  (epochs={epochs})\")\n"
        )

    sklearn_map = {
        "KRR":           ("from sklearn.kernel_ridge import KernelRidge",
                          lambda h: f"KernelRidge(alpha={h.get('alpha',1.0)}, kernel='{h.get('kernel','rbf')}')"),
        "Polynomial":    ("from sklearn.linear_model import Ridge\nfrom sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.pipeline import Pipeline",
                          lambda h: f"Pipeline([('poly', PolynomialFeatures({h.get('degree',2)})), ('ridge', Ridge({h.get('alpha',1.0)}))])"),
        "RandomForest":  ("from sklearn.ensemble import RandomForestRegressor",
                          lambda h: f"RandomForestRegressor(n_estimators={h.get('n_estimators',100)}, max_depth={h.get('max_depth',None)}, random_state=42)"),
        "GradientBoosting": ("from sklearn.ensemble import GradientBoostingRegressor",
                             lambda h: f"GradientBoostingRegressor(n_estimators={h.get('n_estimators',100)}, learning_rate={h.get('learning_rate',0.1)}, max_depth={h.get('max_depth',3)})"),
        "SVR":           ("from sklearn.svm import SVR",
                          lambda h: f"SVR(C={h.get('C',1.0)}, kernel='{h.get('kernel','rbf')}')"),
    }
    if model in sklearn_map:
        imp, builder = sklearn_map[model]
        return (
            f"# ── Regressor: {model} ──\n"
            f"import numpy as np\n"
            f"{imp}\n"
            f"{var}_model = {builder(hp)}\n"
            f"_{var}_yf = {up_var}_y.ravel() if {up_var}_y.ndim > 1 and {up_var}_y.shape[1]==1 else {up_var}_y\n"
            f"{var}_model.fit({up_var}_X, _{var}_yf)\n"
            f"print(f\"{model} fitting complete\")\n"
        )

    return (
        f"# ── Regressor: {model} (TODO) ──\n"
        f"{var}_model = None  # implement {model}\n"
    )


def _gen_classifier(nd_data: dict, var: str, up_var: str) -> str:
    model = nd_data.get("model", "")
    hp    = _hp(nd_data)
    sklearn_map = {
        "RandomForest":     ("from sklearn.ensemble import RandomForestClassifier",
                             lambda h: f"RandomForestClassifier(n_estimators={h.get('n_estimators',100)}, max_depth={h.get('max_depth',None)}, random_state=42)"),
        "KNN":              ("from sklearn.neighbors import KNeighborsClassifier",
                             lambda h: f"KNeighborsClassifier(n_neighbors={h.get('n_neighbors',5)})"),
        "GradientBoosting": ("from sklearn.ensemble import GradientBoostingClassifier",
                             lambda h: f"GradientBoostingClassifier(n_estimators={h.get('n_estimators',100)}, learning_rate={h.get('learning_rate',0.1)}, max_depth={h.get('max_depth',3)})"),
        "LogisticRegression": ("from sklearn.linear_model import LogisticRegression",
                               lambda h: f"LogisticRegression(C={h.get('C',1.0)}, max_iter=1000)"),
    }
    if model in sklearn_map:
        imp, builder = sklearn_map[model]
        return (
            f"# ── Classifier: {model} ──\n"
            f"import numpy as np\n"
            f"{imp}\n"
            f"{var}_model = {builder(hp)}\n"
            f"{var}_model.fit({up_var}_X, {up_var}_y.ravel())\n"
            f"print(f\"{model} fitting complete\")\n"
        )
    return f"# ── Classifier: {model} (TODO) ──\n{var}_model = None\n"


def _gen_save(var: str, label: str, output_dir: str) -> str:
    return (
        f"# ── Save: {label} ──\n"
        f"import pickle\n"
        f"from pathlib import Path\n"
        f"Path(\"{output_dir}\").mkdir(parents=True, exist_ok=True)\n"
        f"with open(\"{output_dir}/{var}_model.pkl\", \"wb\") as _f:\n"
        f"    pickle.dump({var}_model, _f, protocol=pickle.HIGHEST_PROTOCOL)\n"
        f"print(f\"Model '{label}' saved to {output_dir}/{var}_model.pkl\")\n"
    )


def _generate_classic_script(
    subgraph_nodes: list[dict],
    all_edges: list[dict],
    exporter_node_id: str,
    output_dir: str,
) -> str:
    """Generate a sequential X/y training script for classic pipelines."""
    active_nodes = [n for n in subgraph_nodes if n["id"] != exporter_node_id]
    active_edges = [
        e for e in all_edges
        if e["source"] != exporter_node_id and e["target"] != exporter_node_id
    ]
    order     = _topo_sort(active_nodes, active_edges)
    node_map  = {n["id"]: _node_data(n) for n in active_nodes}
    upstream: dict[str, list[str]] = defaultdict(list)
    for e in active_edges:
        upstream[e["target"]].append(e["source"])
    node_var  = {nid: f"n{i}" for i, nid in enumerate(order)}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "#!/usr/bin/env python3",
        '"""',
        "train.py  —  Generated by SurroMod",
        f"Generated : {timestamp}",
        "",
        "Standalone script  —  no SurroMod installation required.",
        "Dependencies:  pip install numpy pandas scikit-learn torch",
        '"""',
        "",
        "import numpy as np",
        "import pandas as pd",
        "",
    ]

    model_nodes: list[str] = []

    for nid in order:
        data    = node_map.get(nid, {})
        cat     = data.get("category", "")
        var     = node_var[nid]
        up_ids  = upstream.get(nid, [])
        up_var  = node_var[up_ids[0]] if up_ids else ""

        if cat == "input":
            lines.append(_gen_input(data, var))
        elif cat == "feature_engineering":
            if up_var:
                lines.append(_gen_feature_engineering(data, var, up_var))
            else:
                lines.append(f"# FE node '{data.get('label')}' has no upstream – skipped\n")
        elif cat == "regressor":
            if up_var:
                lines.append(_gen_regressor(data, var, up_var))
                model_nodes.append(nid)
        elif cat == "classifier":
            if up_var:
                lines.append(_gen_classifier(data, var, up_var))
                model_nodes.append(nid)
        elif cat in ("validator", "inference", "postprocessing", "hp_tuner",
                     "gram_exporter", "code_exporter"):
            pass  # not exported
        else:
            lines.append(f"# Node '{data.get('label', cat)}' ({cat}) – skipped\n")

    if model_nodes:
        lines.append("# ─── Save trained models ──────────────────────────────────────────────\n")
        for nid in model_nodes:
            data = node_map[nid]
            lines.append(_gen_save(node_var[nid], data.get("label", node_var[nid]), output_dir))
    else:
        lines.append("# No trainable model nodes found in pipeline.\n")

    lines.append("print(\"\\nTraining complete!\")\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_train_script(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    exporter_node_id: str,
    output_dir: str = "outputs",
) -> str:
    """
    Generate a standalone train.py from the pipeline graph.

    Detects whether the connected regressor is a GraphFlowForecaster and
    dispatches to the appropriate generator (full GFF script vs classic).

    Parameters
    ----------
    nodes            : serialised node dicts from the frontend store.
    edges            : serialised edge dicts.
    exporter_node_id : ID of the CodeExporter node.
    output_dir       : directory where model artefacts will be saved.

    Returns
    -------
    str : the generated Python source code.
    """
    # Find the regressor directly connected to the exporter
    reg_node = _find_connected_regressor(nodes, edges, exporter_node_id)
    if reg_node is None:
        # Fall back: collect all nodes, generate classic script
        return _generate_classic_script(nodes, edges, exporter_node_id, output_dir)

    # Collect the full upstream subgraph (regressor + all its ancestors)
    subgraph = _collect_upstream(nodes, edges, reg_node["id"])
    # Also include the exporter node so it can be stripped later if needed
    # (classic path strips it; gff path ignores it)

    reg_data = _node_data(reg_node)
    model    = reg_data.get("model", "")

    if model in ("GraphFlowForecaster", "GFF", "graph_flow_forecaster"):
        return _generate_gff_script(subgraph, edges, reg_node["id"], output_dir)
    else:
        # classic: include exporter in node list so it gets filtered inside
        return _generate_classic_script(
            subgraph + [{"id": exporter_node_id, "data": {"category": "code_exporter"}}],
            edges,
            exporter_node_id,
            output_dir,
        )
