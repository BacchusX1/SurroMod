"""
Hierarchical Graph Builder
==========================
Feature engineering node that constructs a two-level spatial graph on
top of a point cloud for use with the hierarchical U-Net GNN inside
``GraphFlowForecaster``.

Level 0 — Fine graph
    k-NN graph on the full N-point mesh (default k=8).

Level 1 — Coarse graph
    Random subsample of K = N × coarse_ratio points, then k-NN graph
    on those K points (default k=8).

Unpool edges
    For each fine node, the k_unpool=3 nearest coarse nodes (by
    position) with inverse-distance weights.  Used to scatter coarse
    latent features back to fine resolution.

Per-sim extras cached alongside graphs
    dist_to_af : (N,) float32 — Euclidean distance from each node to
                 the nearest airfoil-surface node.  Used by GFF to
                 compute proximity-weighted training loss.

All artefacts are cached to a temporary directory (one .npy file per
sim) so that temporal windows sharing the same mesh reuse the same
graph without recomputation.

Ports
-----
Input : ``pos``               (N, 3)
Output: ``edge_index_fine``   (2, E_f)
        ``edge_attr_fine``    (E_f, D_e)
        ``edge_index_coarse`` (2, E_c)
        ``edge_attr_coarse``  (E_c, D_e)
        ``coarse_indices``    (K,)         — global fine-node indices
        ``unpool_ftc_idx``    (N, k_unpool) — per fine node: local coarse indices
        ``unpool_ftc_w``      (N, k_unpool) — inverse-distance weights
        ``dist_to_af``        (N,)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_knn_graph(
    pos: np.ndarray,
    k: int,
    include_displacement: bool = True,
    include_distance: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (edge_index, edge_attr) for a k-NN graph on *pos*."""
    from sklearn.neighbors import NearestNeighbors

    N = pos.shape[0]
    k_actual = min(k + 1, N)  # +1 because the point itself is returned

    nn = NearestNeighbors(n_neighbors=k_actual, algorithm="auto")
    nn.fit(pos)
    _, indices = nn.kneighbors(pos)

    sources, targets = [], []
    for i in range(N):
        for j in range(k_actual):
            nb = indices[i, j]
            if nb == i:
                continue
            sources.append(i)
            targets.append(nb)

    edge_index = np.array([sources, targets], dtype=np.int64)  # (2, E)
    E = edge_index.shape[1]

    if not (include_displacement or include_distance) or E == 0:
        return edge_index, None

    src, tgt = edge_index[0], edge_index[1]
    diff = pos[tgt] - pos[src]  # (E, 3)

    parts: list[np.ndarray] = []
    if include_displacement:
        parts.append(diff)
    if include_distance:
        parts.append(np.linalg.norm(diff, axis=1, keepdims=True).astype(np.float32))

    attr = np.concatenate(parts, axis=1).astype(np.float32)

    if normalize and attr.shape[0] > 0 and include_displacement and include_distance:
        dist_col = attr[:, 3:4]
        median_len = float(np.median(dist_col))
        if median_len > 1e-12:
            attr[:, :3] /= median_len
            attr[:, 3:4] /= float(dist_col.max() + 1e-12)

    return edge_index, attr


def _build_unpool_edges(
    pos_fine: np.ndarray,
    pos_coarse: np.ndarray,
    k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each fine node find *k* nearest coarse nodes.

    Returns
    -------
    ftc_idx : (N, k)  int32  — local coarse indices (0..K-1)
    ftc_w   : (N, k)  float32 — inverse-distance weights summing to 1
    """
    from sklearn.neighbors import NearestNeighbors

    K = pos_coarse.shape[0]
    k_actual = min(k, K)

    nn = NearestNeighbors(n_neighbors=k_actual, algorithm="ball_tree")
    nn.fit(pos_coarse)
    dists, idxs = nn.kneighbors(pos_fine)  # (N, k_actual)

    inv_d = 1.0 / (dists.astype(np.float32) + 1e-8)
    w = inv_d / inv_d.sum(axis=1, keepdims=True)

    return idxs.astype(np.int32), w.astype(np.float32)


def _compute_dist_to_af(pos: np.ndarray, idcs_airfoil: np.ndarray) -> np.ndarray:
    """
    Euclidean distance (2-D xy plane) from each node to the nearest
    airfoil-surface node.  Returns (N,) float32.
    """
    from sklearn.neighbors import NearestNeighbors

    af_pos = pos[idcs_airfoil, :2].astype(np.float32)  # (M, 2)
    all_pos_2d = pos[:, :2].astype(np.float32)         # (N, 2)

    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(af_pos)
    dists, _ = nn.kneighbors(all_pos_2d)               # (N, 1)
    return dists[:, 0].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalGraphBuilder:
    """
    Build fine + coarse graphs and unpool edges for a point cloud.

    Expected ``hyperparams``:
        fine_k              : int   – k for fine-level kNN         (default 8)
        coarse_ratio        : float – fraction of nodes for coarse (default 0.08)
        coarse_k            : int   – k for coarse-level kNN       (default 8)
        k_unpool            : int   – nearest coarse per fine node  (default 3)
        normalize_edge_attr : bool  – normalise displacements       (default True)
    """

    # Sentinel flag so GFF can detect this node without circular imports
    IS_HIERARCHICAL_GRAPH_BUILDER: bool = True

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = hyperparams or {}
        self._fine_k: int = int(hp.get("fine_k", 8))
        self._coarse_ratio: float = float(hp.get("coarse_ratio", 0.08))
        self._coarse_k: int = int(hp.get("coarse_k", 8))
        self._k_unpool: int = int(hp.get("k_unpool", 3))
        self._normalize: bool = bool(hp.get("normalize_edge_attr", True))
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        samples = inputs.get("samples")
        if samples is not None and isinstance(samples, list) and len(samples) > 0:
            return self._execute_multi(inputs)
        return self._execute_single(inputs)

    def _execute_single(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Build hierarchical graph for a single pos array (legacy mode)."""
        pos = np.asarray(inputs["pos"], dtype=np.float32)
        idcs_af = inputs.get("idcs_airfoil")
        if idcs_af is not None:
            idcs_af = np.asarray(idcs_af, dtype=np.int64)

        artefacts = self._build_for_pos(pos, idcs_af)
        result = {**inputs, **artefacts}
        logger.info(
            "HierarchicalGraphBuilder[single]: N=%d  K=%d  E_f=%d  E_c=%d",
            pos.shape[0], artefacts["coarse_indices"].shape[0],
            artefacts["edge_index_fine"].shape[1],
            artefacts["edge_index_coarse"].shape[1],
        )
        return result

    def _execute_multi(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Build and cache graphs per unique sim_id (lazy mode)."""
        samples = inputs["samples"]
        lazy = "filepath" in samples[0] and "pos" not in samples[0]

        if not lazy:
            # Eager: build in-memory per sim
            cache: dict[str, dict] = {}
            for s in samples:
                sid = s.get("sim_id", "")
                if sid in cache:
                    s.update(cache[sid])
                    continue
                pos = np.asarray(s["pos"], dtype=np.float32)
                idcs_af = s.get("idcs_airfoil")
                if idcs_af is not None:
                    idcs_af = np.asarray(idcs_af, dtype=np.int64)
                artefacts = self._build_for_pos(pos, idcs_af)
                s.update(artefacts)
                if sid:
                    cache[sid] = artefacts
            return {**inputs}

        # ── lazy mode: precompute per unique sim_id, cache to disk ────────
        cache_dir = Path(tempfile.mkdtemp(prefix="hgb_cache_"))
        graph_cache: dict[str, dict[str, str]] = {}

        sim_first: dict[str, dict] = {}
        for s in samples:
            sid = s.get("sim_id", "")
            if sid not in sim_first:
                sim_first[sid] = s

        max_points = int(inputs.get("max_points", 0))

        for sid, sample in sim_first.items():
            from src.backend.data_digester.temporal_point_cloud_field_digester import (
                TemporalPointCloudFieldDigester,
            )
            arrays = TemporalPointCloudFieldDigester.load_sample(
                sample["filepath"],
                keys={"pos", "idcs_airfoil"},
                max_points=max_points,
            )
            pos = np.asarray(arrays["pos"], dtype=np.float32)
            idcs_af = arrays.get("idcs_airfoil")
            if idcs_af is not None:
                idcs_af = np.asarray(idcs_af, dtype=np.int64)
            del arrays

            artefacts = self._build_for_pos(pos, idcs_af)

            # Save every array to disk
            paths: dict[str, str] = {}
            for key, arr in artefacts.items():
                if arr is None:
                    paths[key] = ""
                    continue
                p = str(cache_dir / f"{sid}_{key}.npy")
                np.save(p, arr)
                paths[key] = p

            graph_cache[sid] = paths
            del pos, artefacts

            logger.info(
                "HierarchicalGraphBuilder: cached sim '%s'  E_f=%s  E_c=%s",
                sid,
                np.load(paths["edge_index_fine"]).shape[1]
                if paths.get("edge_index_fine") else "?",
                np.load(paths["edge_index_coarse"]).shape[1]
                if paths.get("edge_index_coarse") else "?",
            )

        # Merge into pipeline transport
        existing_gc = inputs.get("graph_cache", {})
        # Store under "hgb_cache" key to avoid colliding with SGB's "graph_cache"
        merged = {**inputs}
        merged["hgb_cache"] = graph_cache
        merged["hgb_cache_dir"] = str(cache_dir)
        if "fe_pipeline" not in merged:
            merged["fe_pipeline"] = []
        merged["fe_pipeline"].append(self)
        # Keep any existing graph_cache from SpatialGraphBuilder intact
        if existing_gc:
            merged["graph_cache"] = existing_gc

        logger.info(
            "HierarchicalGraphBuilder[lazy]: %d unique graphs cached to %s",
            len(graph_cache), cache_dir,
        )
        return merged

    # ── lazy per-sample processing (called by GFF) ───────────────────────

    def process_sample(
        self,
        sample_data: dict[str, Any],
        graph_cache: dict[str, Any] | None = None,
    ) -> None:
        """
        Load cached hierarchical graph artefacts into *sample_data* in-place.

        GFF passes ``hgb_cache`` (not the generic ``graph_cache``) here.
        """
        hgb_cache: dict[str, dict[str, str]] | None = graph_cache
        sim_id = sample_data.get("sim_id", "")

        if hgb_cache and sim_id in hgb_cache:
            paths = hgb_cache[sim_id]
            for key, p in paths.items():
                if p:
                    sample_data[key] = np.load(p)
                else:
                    sample_data[key] = None
            return

        # Fallback: build on-the-fly
        pos = np.asarray(sample_data["pos"], dtype=np.float32)
        idcs_af = sample_data.get("idcs_airfoil")
        if idcs_af is not None:
            idcs_af = np.asarray(idcs_af, dtype=np.int64)
        artefacts = self._build_for_pos(pos, idcs_af)
        sample_data.update(artefacts)

    # ── core construction ────────────────────────────────────────────────

    def _build_for_pos(
        self, pos: np.ndarray, idcs_airfoil: np.ndarray | None
    ) -> dict[str, Any]:
        """
        Build all hierarchical graph artefacts for *pos*.

        Returns a dict with keys:
            edge_index_fine, edge_attr_fine,
            edge_index_coarse, edge_attr_coarse,
            coarse_indices,
            unpool_ftc_idx, unpool_ftc_w,
            dist_to_af   (or None when idcs_airfoil is unavailable)
        """
        N = pos.shape[0]

        # ── Fine graph ───────────────────────────────────────────────────
        ei_fine, ea_fine = _build_knn_graph(
            pos, self._fine_k,
            include_displacement=True,
            include_distance=True,
            normalize=self._normalize,
        )

        # ── Coarse subsample (random) ─────────────────────────────────────
        K = max(1, int(round(N * self._coarse_ratio)))
        rng = self._rng if self._seed is not None else np.random.RandomState()
        coarse_indices = np.sort(
            rng.choice(N, size=K, replace=False)
        ).astype(np.int64)
        pos_coarse = pos[coarse_indices]

        # ── Coarse graph ─────────────────────────────────────────────────
        ei_coarse, ea_coarse = _build_knn_graph(
            pos_coarse, self._coarse_k,
            include_displacement=True,
            include_distance=True,
            normalize=self._normalize,
        )

        # ── Unpool edges (fine → k nearest coarse) ───────────────────────
        ftc_idx, ftc_w = _build_unpool_edges(pos, pos_coarse, k=self._k_unpool)

        # ── Distance to airfoil ──────────────────────────────────────────
        dist_to_af: np.ndarray | None = None
        if idcs_airfoil is not None and len(idcs_airfoil) > 0:
            dist_to_af = _compute_dist_to_af(pos, idcs_airfoil)

        return {
            "edge_index_fine": ei_fine,
            "edge_attr_fine": ea_fine,
            "edge_index_coarse": ei_coarse,
            "edge_attr_coarse": ea_coarse,
            "coarse_indices": coarse_indices,
            "unpool_ftc_idx": ftc_idx,
            "unpool_ftc_w": ftc_w,
            "dist_to_af": dist_to_af,
        }
