"""
Spatial Graph Builder
=====================
Feature engineering node that constructs a graph on top of point cloud
positions using k-nearest-neighbours or radius-based connectivity.

The resulting ``edge_index`` and optional ``edge_attr`` can be passed
directly into graph neural network nodes for surrogate modelling.

Supports two modes:
    * **knn**    – connect each point to its k nearest neighbours.
    * **radius** – connect all point pairs within a given radius,
                   capped at ``max_neighbors`` per point.

Uses numpy / scipy / sklearn for a robust CPU implementation that
does not require torch_geometric.

In **lazy multi-sample** mode (samples carry only ``filepath``), graphs
are precomputed per unique ``sim_id`` and cached to disk.  The
``process_sample`` method loads a cached graph on demand – called by
the downstream GFF during training.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SpatialGraphBuilder:
    """
    Build spatial graphs from point cloud positions.

    Expected ``hyperparams``:
        - graph_mode                    : str   – ``"knn"`` or ``"radius"``
        - k                             : int   – neighbours for knn mode
        - radius                        : float – radius for radius mode
        - max_neighbors                 : int   – cap per point in radius mode
        - include_relative_displacement : bool  – include (dx, dy, dz) in edge_attr
        - include_distance              : bool  – include euclidean distance in edge_attr
        - self_loops                    : bool  – include self-edges
        - normalize_edge_attr           : bool  – normalise displacements by median edge
                                                  length and distance by max distance
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = hyperparams or {}
        self._graph_mode: str = str(hp.get("graph_mode", "knn"))
        self._k: int = int(hp.get("k", 16))
        self._radius: float = float(hp.get("radius", 0.1))
        self._max_neighbors: int = int(hp.get("max_neighbors", 32))
        self._include_displacement: bool = bool(hp.get("include_relative_displacement", True))
        self._include_distance: bool = bool(hp.get("include_distance", True))
        self._self_loops: bool = bool(hp.get("self_loops", False))
        self._normalize_edge_attr: bool = bool(hp.get("normalize_edge_attr", False))
        self._seed = seed

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Build graphs from upstream ``pos`` arrays.

        In **multi-sample** mode (``samples`` key present), builds a
        graph for each sample and caches per ``sim_id`` so that
        temporal windows sharing the same mesh reuse the graph.
        """
        samples = inputs.get("samples")
        if samples is not None and isinstance(samples, list) and len(samples) > 0:
            return self._execute_multi(inputs)
        return self._execute_single(inputs)

    def _execute_single(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Build graph for a single pos array (legacy mode)."""
        pos = np.asarray(inputs["pos"], dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"pos must have shape (N, 3), got {pos.shape}")

        N = pos.shape[0]

        if self._graph_mode == "knn":
            edge_index = self._build_knn(pos)
        elif self._graph_mode == "radius":
            edge_index = self._build_radius(pos)
        else:
            raise ValueError(f"Unknown graph_mode: {self._graph_mode}")

        # Add self-loops if requested
        if self._self_loops:
            self_edges = np.stack([np.arange(N), np.arange(N)], axis=0).astype(np.int64)
            edge_index = np.concatenate([edge_index, self_edges], axis=1)

        E = edge_index.shape[1]

        # Build edge attributes
        edge_attr = self._compute_edge_attr(pos, edge_index)

        result: dict[str, Any] = {**inputs}
        result["edge_index"] = edge_index
        if edge_attr is not None:
            result["edge_attr"] = edge_attr
        result["graph_meta"] = {
            "num_nodes": int(N),
            "num_edges": int(E),
            "graph_mode": self._graph_mode,
            "k": self._k if self._graph_mode == "knn" else None,
            "radius": self._radius if self._graph_mode == "radius" else None,
        }

        logger.info(
            "SpatialGraphBuilder: mode=%s  N=%d  E=%d",
            self._graph_mode, N, E,
        )
        return result

    def _execute_multi(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Build graphs per unique sim_id, cache to disk (lazy mode)."""
        samples = inputs["samples"]

        # Detect lazy mode: samples carry filepath but no pos array
        lazy = "filepath" in samples[0] and "pos" not in samples[0]

        if not lazy:
            # Eager mode (small datasets / unit tests) — keep old behaviour
            return self._execute_multi_eager(inputs)

        # ── lazy mode: precompute graphs to temporary .npy files ─────────
        cache_dir = Path(tempfile.mkdtemp(prefix="sgb_cache_"))
        graph_cache: dict[str, dict[str, str | None]] = {}

        # Group by sim_id → pick first sample per sim for pos loading
        sim_first: dict[str, dict[str, Any]] = {}
        for s in samples:
            sid = s.get("sim_id", "")
            if sid not in sim_first:
                sim_first[sid] = s

        max_points = int(inputs.get("max_points", 0))

        for sid, sample in sim_first.items():
            # Load only pos from disk
            from src.backend.data_digester.temporal_point_cloud_field_digester import (
                TemporalPointCloudFieldDigester,
            )
            arrays = TemporalPointCloudFieldDigester.load_sample(
                sample["filepath"], keys={"pos"}, max_points=max_points,
            )
            pos = np.asarray(arrays["pos"], dtype=np.float32)
            del arrays

            if self._graph_mode == "knn":
                edge_index = self._build_knn(pos)
            elif self._graph_mode == "radius":
                edge_index = self._build_radius(pos)
            else:
                raise ValueError(f"Unknown graph_mode: {self._graph_mode}")

            if self._self_loops:
                N = pos.shape[0]
                self_edges = np.stack([np.arange(N), np.arange(N)], axis=0).astype(np.int64)
                edge_index = np.concatenate([edge_index, self_edges], axis=1)

            edge_attr = self._compute_edge_attr(pos, edge_index)

            # Save to disk
            ei_path = str(cache_dir / f"{sid}_edge_index.npy")
            np.save(ei_path, edge_index)
            ea_path: str | None = None
            if edge_attr is not None:
                ea_path = str(cache_dir / f"{sid}_edge_attr.npy")
                np.save(ea_path, edge_attr)

            graph_cache[sid] = {"edge_index": ei_path, "edge_attr": ea_path}

            logger.info(
                "SpatialGraphBuilder: cached graph for sim '%s'  E=%d",
                sid, edge_index.shape[1],
            )
            del pos, edge_index, edge_attr

        # Register self + cache in the pipeline transport dict
        inputs["graph_cache"] = graph_cache
        inputs["graph_cache_dir"] = str(cache_dir)
        if "fe_pipeline" not in inputs:
            inputs["fe_pipeline"] = []
        inputs["fe_pipeline"].append(self)

        logger.info(
            "SpatialGraphBuilder[lazy]: %d unique graphs cached to %s",
            len(graph_cache), cache_dir,
        )
        return {**inputs}

    def _execute_multi_eager(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Build graphs in-memory for each sample (small datasets)."""
        samples = inputs["samples"]
        cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

        for sample in samples:
            sim_id = sample.get("sim_id", "")
            if sim_id in cache:
                edge_index, edge_attr = cache[sim_id]
            else:
                pos = np.asarray(sample["pos"], dtype=np.float32)
                if self._graph_mode == "knn":
                    edge_index = self._build_knn(pos)
                elif self._graph_mode == "radius":
                    edge_index = self._build_radius(pos)
                else:
                    raise ValueError(f"Unknown graph_mode: {self._graph_mode}")
                if self._self_loops:
                    N = pos.shape[0]
                    self_edges = np.stack([np.arange(N), np.arange(N)], axis=0).astype(np.int64)
                    edge_index = np.concatenate([edge_index, self_edges], axis=1)
                edge_attr = self._compute_edge_attr(pos, edge_index)
                if sim_id:
                    cache[sim_id] = (edge_index, edge_attr)

            sample["edge_index"] = edge_index
            if edge_attr is not None:
                sample["edge_attr"] = edge_attr

        E0 = samples[0]["edge_index"].shape[1] if samples else 0
        logger.info(
            "SpatialGraphBuilder[multi]: %d samples, %d cached graphs, E=%d",
            len(samples), len(cache), E0,
        )
        return {**inputs}

    # ── lazy per-sample processing (called by GFF) ───────────────────────

    def process_sample(
        self,
        sample_data: dict[str, Any],
        graph_cache: dict[str, dict[str, str | None]] | None = None,
    ) -> None:
        """
        Load or build graph for a single sample and add edge_index /
        edge_attr to *sample_data* in-place.
        """
        sim_id = sample_data.get("sim_id", "")
        if graph_cache and sim_id in graph_cache:
            paths = graph_cache[sim_id]
            sample_data["edge_index"] = np.load(paths["edge_index"])
            if paths["edge_attr"] is not None:
                sample_data["edge_attr"] = np.load(paths["edge_attr"])
            return

        # Fallback: build on-the-fly (unit-test / single sample)
        pos = np.asarray(sample_data["pos"], dtype=np.float32)
        if self._graph_mode == "knn":
            edge_index = self._build_knn(pos)
        elif self._graph_mode == "radius":
            edge_index = self._build_radius(pos)
        else:
            raise ValueError(f"Unknown graph_mode: {self._graph_mode}")

        if self._self_loops:
            N = pos.shape[0]
            self_edges = np.stack([np.arange(N), np.arange(N)], axis=0).astype(np.int64)
            edge_index = np.concatenate([edge_index, self_edges], axis=1)

        sample_data["edge_index"] = edge_index
        edge_attr = self._compute_edge_attr(pos, edge_index)
        if edge_attr is not None:
            sample_data["edge_attr"] = edge_attr

    # ── graph construction ───────────────────────────────────────────────

    def _build_knn(self, pos: np.ndarray) -> np.ndarray:
        """Build k-nearest-neighbour graph. Returns (2, E) edge_index."""
        from sklearn.neighbors import NearestNeighbors

        N = pos.shape[0]
        # k+1 because the point itself is included as nearest neighbour
        k_actual = min(self._k + 1, N)

        nn = NearestNeighbors(n_neighbors=k_actual, algorithm="auto")
        nn.fit(pos)
        _, indices = nn.kneighbors(pos)

        # indices shape: (N, k_actual); first column is the point itself
        sources = []
        targets = []
        for i in range(N):
            for j in range(k_actual):
                neighbour = indices[i, j]
                if neighbour == i and not self._self_loops:
                    continue
                sources.append(i)
                targets.append(neighbour)

        edge_index = np.array([sources, targets], dtype=np.int64)
        return edge_index

    def _build_radius(self, pos: np.ndarray) -> np.ndarray:
        """Build radius-based graph. Returns (2, E) edge_index."""
        from sklearn.neighbors import NearestNeighbors

        N = pos.shape[0]
        max_n = min(self._max_neighbors + 1, N)

        nn = NearestNeighbors(radius=self._radius, algorithm="auto")
        nn.fit(pos)
        distances, indices = nn.radius_neighbors(pos, sort_results=True)

        sources = []
        targets = []
        for i in range(N):
            count = 0
            for j, neighbour in enumerate(indices[i]):
                if neighbour == i and not self._self_loops:
                    continue
                if count >= self._max_neighbors:
                    break
                sources.append(i)
                targets.append(neighbour)
                count += 1

        if not sources:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            edge_index = np.array([sources, targets], dtype=np.int64)
        return edge_index

    def _compute_edge_attr(
        self, pos: np.ndarray, edge_index: np.ndarray
    ) -> np.ndarray | None:
        """Compute optional edge attributes."""
        if not self._include_displacement and not self._include_distance:
            return None

        src = edge_index[0]
        tgt = edge_index[1]
        diff = pos[tgt] - pos[src]  # (E, 3)

        parts: list[np.ndarray] = []
        if self._include_displacement:
            parts.append(diff)
        if self._include_distance:
            dist = np.linalg.norm(diff, axis=1, keepdims=True)
            parts.append(dist.astype(np.float32))

        if not parts:
            return None

        attr = np.concatenate(parts, axis=1).astype(np.float32)

        if self._normalize_edge_attr and attr.shape[0] > 0:
            # Displacement columns: divide by median edge length (robust scale)
            if self._include_displacement and self._include_distance:
                dist_col = attr[:, 3:4]
                median_len = float(np.median(dist_col))
                if median_len > 1e-12:
                    attr[:, :3] /= median_len  # normalise displacement
                    attr[:, 3:4] /= (dist_col.max() + 1e-12)  # normalise distance to [0,1]
            elif self._include_displacement:
                lengths = np.linalg.norm(attr, axis=1)
                median_len = float(np.median(lengths))
                if median_len > 1e-12:
                    attr /= median_len
            elif self._include_distance:
                max_dist = float(attr.max())
                if max_dist > 1e-12:
                    attr /= max_dist

        return attr
