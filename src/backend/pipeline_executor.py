"""
Pipeline Executor
=================
Topologically traverses the node graph from the frontend and executes
each node in order, passing outputs from upstream to downstream.

The executor is backend-framework agnostic – it routes each node's
``category`` / ``model`` / ``method`` to the correct Python class.

Data-flow contract
------------------
Every node's ``execute()`` receives a merged dict of upstream outputs and
returns a dict that is stored and forwarded to downstream nodes.

Special merging rules:

*  **General nodes** (input, feature_engineering, …):
   When multiple upstream nodes feed into a single node, the outputs are
   *smart-merged*:
     - ``X`` arrays → horizontally stacked (feature concatenation).
     - ``feature_names`` → lists concatenated.
     - All other keys → last writer wins.

*  **Validator nodes** (multi-model comparison):
   Instead of merging, each upstream *regressor* is bundled as a separate
   entry with its **own** ``X``, ``y``, ``y_scaler``, ``label_names`` and
   ``feature_names``.  This guarantees every model is evaluated with the
   exact preprocessing context from its own graph path.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections import defaultdict, deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Seed helpers ─────────────────────────────────────────────────────────────

def _set_global_seeds(seed: int) -> None:
    """
    Set the random seed for all relevant libraries to ensure reproducibility.

    Seeds: Python ``random``, ``numpy``, ``torch`` (CPU + CUDA).
    Also sets ``PYTHONHASHSEED`` and enables PyTorch deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms where possible
        torch.use_deterministic_algorithms(False)  # True can raise on some ops
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not installed — skip

    logger.info("Global random seed set to %d", seed)


# ── Graph helpers ────────────────────────────────────────────────────────────

def _topological_sort(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> list[str]:
    """Return node IDs in topological order."""
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    children: dict[str, list[str]] = defaultdict(list)

    for e in edges:
        src = e["source"]
        tgt = e["target"]
        children[src].append(tgt)
        in_degree.setdefault(tgt, 0)
        in_degree[tgt] += 1

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    order: list[str] = []

    while queue:
        nid = queue.popleft()
        order.append(nid)
        for child in children.get(nid, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(nodes):
        raise ValueError("Pipeline contains a cycle!")
    return order


# ── Edge handle resolution ───────────────────────────────────────────────────

def _resolve_target_handle(edge: dict[str, Any]) -> str:
    """Return the target handle for an edge, with edge-ID fallback for legacy data."""
    th = edge.get("targetHandle") or edge.get("target_handle") or ""
    if th:
        return th
    # Fallback: parse from React Flow edge ID (e.g.
    # "xy-edge__srcId-tgtIdrepresentations")
    # Fallback: parse from React Flow edge ID (legacy data)
    eid = edge.get("id", "")
    tid = edge.get("target", "")
    if tid and tid in eid:
        suffix = eid.split(tid)[-1]
        if suffix in ("representations",):
            logger.warning(
                "Edge '%s' has no explicit targetHandle — resolved to '%s' "
                "via edge-ID parsing (legacy fallback).",
                eid, suffix,
            )
            return suffix
    logger.warning(
        "Edge '%s' (source=%s → target=%s) has no targetHandle — "
        "falling back to 'default'. If this is an RBL representations "
        "edge, it will be misrouted.",
        eid, edge.get("source", "?"), tid,
    )
    return "default"


# ── Smart upstream merging ───────────────────────────────────────────────────

def _merge_upstream_outputs(
    upstream_dicts: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Intelligently merge outputs from multiple upstream nodes.

    Rules
    -----
    * ``X`` (np.ndarray) – horizontally stacked when multiple upstreams
      provide it (feature concatenation).  Arrays **must** have the same
      number of rows (samples).
    * ``feature_names`` (list[str]) – extended in upstream order.
    * All other keys – last writer wins (standard ``dict.update``).
    """
    if not upstream_dicts:
        return {}
    if len(upstream_dicts) == 1:
        return dict(upstream_dicts[0])

    merged: dict[str, Any] = {}
    x_arrays: list[np.ndarray] = []
    feature_name_lists: list[list[str]] = []

    for up in upstream_dicts:
        for key, value in up.items():
            if key == "X" and isinstance(value, np.ndarray):
                x_arrays.append(value)
            elif key == "feature_names" and isinstance(value, list):
                feature_name_lists.append(value)
            else:
                merged[key] = value

    # ── X concatenation ──────────────────────────────────────────────────
    if len(x_arrays) > 1:
        n_samples = x_arrays[0].shape[0]
        for xa in x_arrays[1:]:
            if xa.shape[0] != n_samples:
                raise ValueError(
                    f"Cannot merge X arrays with different sample counts "
                    f"({[xa.shape for xa in x_arrays]}).  "
                    f"All upstream data sources must have the same number of "
                    f"samples when feeding into a single node."
                )
        merged["X"] = np.hstack(x_arrays)
        logger.info(
            "Merged %d X arrays → shape %s",
            len(x_arrays),
            merged["X"].shape,
        )
    elif len(x_arrays) == 1:
        merged["X"] = x_arrays[0]

    # ── feature_names concatenation ──────────────────────────────────────
    if feature_name_lists:
        combined: list[str] = []
        for fn in feature_name_lists:
            combined.extend(fn)
        merged["feature_names"] = combined

    return merged


# ── Validator-specific: per-model context ────────────────────────────────────

_MODEL_CONTEXT_KEYS = (
    "X", "y", "y_pred", "y_scaler", "label_names", "feature_names",
    "X_holdout", "y_holdout", "y_pred_holdout",
)
"""Keys that vary per model and must NOT be shared across models.
``y_pred`` is used by RBL-Aggregator outputs: the full-chain prediction
cannot be reproduced by a single model.predict() call, so the
pre-computed predictions are forwarded to validators directly.
``X_holdout`` / ``y_holdout`` carry the holdout set from TrainTestSplitter.
``y_pred_holdout`` carries pre-computed holdout predictions (e.g. RBL)."""


def _collect_model_entries(
    upstream_ids: list[str],
    outputs: dict[str, Any],
    node_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    For a validator node, bundle each upstream regressor with its own
    complete context (X, y, y_scaler, label_names, feature_names).

    Returns
    -------
    merged_inputs : dict
        Either ``{"model": ..., "X": ..., ...}`` for a single model, or
        ``{"models": [...]}`` for multi-model comparison.
    models_list : list[dict]
        The raw model entries (length 0 if no models found upstream).
    """
    models_list: list[dict[str, Any]] = []

    for uid in upstream_ids:
        up_out = outputs.get(uid, {})
        if not isinstance(up_out, dict):
            continue
        if "model" not in up_out:
            continue

        up_data = node_map[uid].get("data", node_map[uid])
        model_name = up_data.get("label") or up_data.get("model") or uid

        entry: dict[str, Any] = {
            "name": model_name,
            "model": up_out["model"],
        }
        # Attach per-model context
        for ctx_key in _MODEL_CONTEXT_KEYS:
            if ctx_key in up_out:
                entry[ctx_key] = up_out[ctx_key]

        models_list.append(entry)

    # Build merged_inputs
    if len(models_list) == 1:
        entry = models_list[0]
        merged: dict[str, Any] = {"model": entry["model"]}
        for ctx_key in _MODEL_CONTEXT_KEYS:
            if ctx_key in entry:
                merged[ctx_key] = entry[ctx_key]
        return merged, models_list

    if len(models_list) > 1:
        return {"models": models_list}, models_list

    return {}, models_list


def _build_executor(node_data: dict[str, Any], seed: int | None = None) -> Any:
    """Instantiate the correct backend handler for a node."""
    category = node_data.get("category")

    if category == "input":
        input_kind = node_data.get("inputKind", "scalar")
        if input_kind == "scalar":
            from src.backend.data_digester.scalar_data_digester import ScalarDataDigester
            return ScalarDataDigester(node_data)
        if input_kind == "2d_field":
            from src.backend.data_digester.two_d_data_digester import TwoDFieldDigester
            return TwoDFieldDigester(node_data)
        if input_kind == "2d_geometry":
            from src.backend.data_digester.geometry_2d_digester import Geometry2DDigester
            return Geometry2DDigester(node_data)
        # Future digesters:
        # if input_kind == "time_series":
        #     from src.backend.data_digester.time_series_digester import TimeSeriesDigester
        #     return TimeSeriesDigester(node_data)
        # if input_kind == "3d_field":
        #     from src.backend.data_digester.three_d_data_digester import ThreeDFieldDigester
        #     return ThreeDFieldDigester(node_data)
        # if input_kind == "3d_geometry":
        #     from src.backend.data_digester.geometry_3d_digester import Geometry3DDigester
        #     return Geometry3DDigester(node_data)
        raise ValueError(f"Unsupported input kind: {input_kind}")

    if category == "feature_engineering":
        method = node_data.get("method")
        hp = node_data.get("hyperparams", {})

        if method == "Scaler":
            from src.backend.feature_engineering.scaler import Scaler
            return Scaler(hp)

        if method == "PCA":
            from src.backend.feature_engineering.pca import PCATransformer
            return PCATransformer(hp, seed=seed)

        if method == "Autoencoder":
            from src.backend.feature_engineering.autoencoder import Autoencoder
            return Autoencoder(hp, seed=seed)

        if method == "GeometrySampler":
            from src.backend.feature_engineering.geometry_sampler import GeometrySampler
            return GeometrySampler(hp, seed=seed)

        if method == "TrainTestSplit":
            from src.backend.feature_engineering.train_test_splitter import TrainTestSplitter
            return TrainTestSplitter(hp, seed=seed)

        # Other FE methods can be added here
        raise ValueError(f"Unsupported FE method: {method}")

    if category == "regressor":
        from src.backend.predictors.model_base import ModelBase
        # Ensure all regressor subclasses are registered
        import src.backend.predictors.regressors.mlp  # noqa: F401
        import src.backend.predictors.regressors.krr  # noqa: F401
        instance = ModelBase.from_node_data(node_data)
        instance._seed = seed
        return instance

    if category == "rbl":
        from src.backend.feature_engineering.rbl import RBLNode
        return RBLNode(node_data)

    if category == "rbl_aggregator":
        from src.backend.feature_engineering.rbl import RBLAggregatorNode
        return RBLAggregatorNode(node_data)

    if category == "validator":
        kind = node_data.get("validatorKind", "")
        if kind == "regressor_validator":
            from src.backend.analyzers.regressor_validator import RegressorValidator
            return RegressorValidator(node_data)
        raise ValueError(f"Unsupported validator kind: {kind}")

    raise ValueError(f"Unsupported node category: {category}")


def run_pipeline(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]],
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Execute a full pipeline.

    The pipeline runs in three phases:

    1.  **Data & FE phase** — execute non-regressor nodes in topological
        order (input digesters, feature engineering).
    2.  **Branch training phase** — detect regressor branches and train
        them using the :class:`BranchTrainer` (end-to-end for pure
        differentiable chains, alternating residual for mixed chains).
    3.  **Validator phase** — execute validator nodes with trained models.

    Parameters
    ----------
    nodes : list of serialised node dicts (from the frontend store).
    edges : list of edge dicts with ``source`` and ``target`` keys.
    seed  : optional global random seed for reproducibility.

    Returns
    -------
    dict mapping node IDs to their execution results.
    """
    t0 = time.perf_counter()
    logger.info("Pipeline: starting execution (%d nodes, %d edges)", len(nodes), len(edges))

    # ── Set global seeds for reproducibility ─────────────────────────────
    if seed is not None:
        _set_global_seeds(seed)

    order = _topological_sort(nodes, edges)
    node_map: dict[str, dict[str, Any]] = {n["id"]: n for n in nodes}

    # Build adjacency: for each node, which upstream node feeds it?
    upstream: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        upstream[e["target"]].append(e["source"])

    outputs: dict[str, Any] = {}       # node_id → execution output
    results: dict[str, Any] = {}       # node_id → serialisable result for the frontend
    executors: dict[str, Any] = {}     # node_id → executor instance (for branch trainer)

    # ── Phase 1: Execute non-regressor, non-validator nodes ──────────────
    for nid in order:
        nd = node_map[nid]
        data = nd.get("data", nd)
        category = data.get("category")

        if category in ("regressor", "validator", "rbl", "rbl_aggregator", "hp_tuner"):
            continue  # handled in later phases or externally (hp_tuner)

        logger.info("Pipeline: executing node '%s' (%s)", nid, category)

        up_ids = upstream.get(nid, [])
        upstream_dicts = [
            outputs[uid]
            for uid in up_ids
            if uid in outputs and isinstance(outputs[uid], dict)
        ]
        merged_inputs = _merge_upstream_outputs(upstream_dicts)

        executor = _build_executor(data, seed=seed)
        out = executor.execute(merged_inputs)
        outputs[nid] = out
        results[nid] = {"status": "ok"}

    # ── Phase 2: Build regressor executors and detect branches ───────────
    from src.backend.branch_trainer import BranchTrainer

    for nid in order:
        nd = node_map[nid]
        data = nd.get("data", nd)
        cat = data.get("category")
        if cat == "regressor":
            executor = _build_executor(data, seed=seed)
            executors[nid] = executor
        elif cat in ("rbl", "rbl_aggregator"):
            executor = _build_executor(data, seed=seed)
            executors[nid] = executor

    trainer = BranchTrainer(nodes, edges, node_map, outputs, executors)
    branches = trainer.detect_branches()

    # Track which nodes are part of a branch (to avoid double-processing)
    branched_nodes: set[str] = set()
    for branch in branches:
        branched_nodes.update(branch.node_chain)

    # Train each branch
    for branch in branches:
        # Gather merged upstream data for the branch's data sources
        first_nid = branch.regressor_chain[0]
        up_ids = upstream.get(first_nid, [])

        upstream_dicts = [
            outputs[uid]
            for uid in up_ids
            if uid in outputs and isinstance(outputs[uid], dict)
        ]
        merged_data = _merge_upstream_outputs(upstream_dicts)

        X = merged_data.get("X")
        y = merged_data.get("y")
        if X is None or y is None:
            raise ValueError(
                f"Branch starting at '{first_nid}' got no X or y from upstream."
            )

        branch_result = trainer.train_branch(branch, X, y, seed=seed)

        # Store outputs for each node in the branch (regressors + RBL + RBL-Agg)
        # Use dict.fromkeys to preserve order while deduplicating
        seen_chain: list[str] = list(dict.fromkeys(branch.node_chain))
        for nid in seen_chain:
            node_data_entry = node_map[nid].get("data", node_map[nid])
            cat = node_data_entry.get("category")

            if cat == "regressor":
                executor = executors[nid]
                role = node_data_entry.get("role", "final")

                if role == "transform":
                    X_in = merged_data.get("X")
                    X_out = executor.predict(np.asarray(X_in, dtype=np.float32))
                    out_dim = X_out.shape[1] if X_out.ndim > 1 else 1
                    out = {
                        **merged_data,
                        "X": X_out,
                        "feature_names": [f"emb_{i}" for i in range(out_dim)],
                    }
                    # Also transform holdout features through the same model
                    if "X_holdout" in merged_data:
                        X_ho_in = np.asarray(merged_data["X_holdout"], dtype=np.float32)
                        out["X_holdout"] = executor.predict(X_ho_in)
                    outputs[nid] = out
                    results[nid] = {"status": "ok", "is_trained": True}

                    # Refresh merged_data for the next node
                    next_idx = seen_chain.index(nid) + 1
                    if next_idx < len(seen_chain):
                        next_nid = seen_chain[next_idx]
                        # Determine which upstream sources connect via non-
                        # representation handles (representation inputs for
                        # RBL nodes must not be merged into X).
                        rep_sources: set[str] = set()
                        for e in edges:
                            if e["target"] == next_nid and _resolve_target_handle(e) == "representations":
                                rep_sources.add(e["source"])
                        next_up_ids = upstream.get(next_nid, [])
                        next_upstream_dicts = [
                            outputs[uid]
                            for uid in next_up_ids
                            if uid in outputs and isinstance(outputs[uid], dict)
                            and uid not in rep_sources
                        ]
                        merged_data = _merge_upstream_outputs(next_upstream_dicts)

                else:
                    out = {
                        **merged_data,
                        "model": executor,
                        "metrics": branch_result.get("metrics"),
                    }
                    # For RBL branches the final regressor must produce
                    # the actual prediction (residual r) so that the
                    # downstream RBL-Aggregator can compute ŷ = z + r.
                    if branch.has_rbl:
                        X_in = np.asarray(merged_data.get("X"), dtype=np.float32)
                        X_pred = executor.predict(X_in)
                        if X_pred.ndim == 1:
                            X_pred = X_pred.reshape(-1, 1)
                        out["X"] = X_pred
                        # Also predict residual on holdout
                        if "X_holdout" in merged_data:
                            X_ho_in = np.asarray(merged_data["X_holdout"], dtype=np.float32)
                            X_ho_pred = executor.predict(X_ho_in)
                            if X_ho_pred.ndim == 1:
                                X_ho_pred = X_ho_pred.reshape(-1, 1)
                            out["X_holdout"] = X_ho_pred
                    outputs[nid] = out
                    results[nid] = {
                        "metrics": branch_result.get("metrics"),
                        "is_trained": True,
                    }
                    # Refresh merged_data
                    merged_data = out

            elif cat == "rbl":
                # Execute RBL node to transform data flow
                rbl_executor = executors[nid]

                # Gather representation inputs from top-handle upstream
                rbl_up_ids = upstream.get(nid, [])
                # Find which edges target the 'representations' handle
                rep_arrays: list[np.ndarray] = []
                rep_ho_arrays: list[np.ndarray] = []
                for e in edges:
                    if e["target"] != nid or e["source"] not in outputs:
                        continue
                    if _resolve_target_handle(e) == "representations":
                        rep_out = outputs[e["source"]]
                        if isinstance(rep_out, dict) and "X" in rep_out:
                            rep_arrays.append(np.asarray(rep_out["X"], dtype=np.float32))
                        if isinstance(rep_out, dict) and "X_holdout" in rep_out:
                            rep_ho_arrays.append(np.asarray(rep_out["X_holdout"], dtype=np.float32))

                rbl_inputs = {**merged_data, "representations": rep_arrays}
                rbl_out = rbl_executor.execute(rbl_inputs)

                # Propagate holdout through the same RBL logic
                if "X_holdout" in merged_data and "y_holdout" in merged_data:
                    z_ho = np.asarray(merged_data["X_holdout"], dtype=np.float32)
                    y_ho = np.asarray(merged_data["y_holdout"], dtype=np.float32)
                    if z_ho.ndim == 1:
                        z_ho = z_ho.reshape(-1, 1)
                    if y_ho.ndim == 1:
                        y_ho = y_ho.reshape(-1, 1)
                    ho_parts: list[np.ndarray] = [z_ho]
                    for h_ho in rep_ho_arrays:
                        if h_ho.ndim == 1:
                            h_ho = h_ho.reshape(-1, 1)
                        ho_parts.append(h_ho)
                    X_ho_rbl = np.hstack(ho_parts) if len(ho_parts) > 1 else z_ho
                    rbl_out["X_holdout"] = X_ho_rbl
                    rbl_out["y_holdout"] = y_ho - z_ho
                    rbl_out["z_original_holdout"] = z_ho
                    rbl_out["y_original_holdout"] = y_ho

                outputs[nid] = rbl_out
                results[nid] = {"status": "ok"}
                merged_data = rbl_out

            elif cat == "rbl_aggregator":
                # Execute RBL-Aggregator
                agg_executor = executors[nid]
                agg_out = agg_executor.execute(merged_data)

                # Compute holdout aggregation: ŷ_ho = z_ho + r_ho
                if "X_holdout" in merged_data and "z_original_holdout" in merged_data:
                    r_ho = np.asarray(merged_data["X_holdout"], dtype=np.float32)
                    z_ho = np.asarray(merged_data["z_original_holdout"], dtype=np.float32)
                    if r_ho.ndim == 1:
                        r_ho = r_ho.reshape(-1, 1)
                    if z_ho.ndim == 1:
                        z_ho = z_ho.reshape(-1, 1)
                    y_hat_ho = z_ho + r_ho
                    agg_out["y_pred_holdout"] = y_hat_ho
                    agg_out["X_holdout"] = y_hat_ho
                    if "y_original_holdout" in merged_data:
                        agg_out["y_holdout"] = np.asarray(
                            merged_data["y_original_holdout"], dtype=np.float32,
                        )

                outputs[nid] = agg_out
                results[nid] = {
                    "metrics": agg_out.get("metrics"),
                    "status": "ok",
                }
                merged_data = agg_out

    # Handle any standalone regressors not in a branch (shouldn't happen
    # normally, but defensive)
    for nid in order:
        nd = node_map[nid]
        data = nd.get("data", nd)
        if data.get("category") == "regressor" and nid not in branched_nodes:
            logger.warning("Regressor '%s' is not part of any branch — training standalone.", nid)
            executor = executors.get(nid) or _build_executor(data, seed=seed)

            up_ids = upstream.get(nid, [])
            upstream_dicts = [
                outputs[uid]
                for uid in up_ids
                if uid in outputs and isinstance(outputs[uid], dict)
            ]
            merged_inputs = _merge_upstream_outputs(upstream_dicts)

            X = merged_inputs.get("X")
            y = merged_inputs.get("y")
            if X is None or y is None:
                raise ValueError(f"Regressor node '{nid}' got no X or y from upstream.")

            X_arr = np.asarray(X, dtype=np.float32)
            y_arr = np.asarray(y, dtype=np.float32)
            executor._input_dim = X_arr.shape[1] if X_arr.ndim > 1 else 1
            executor._output_dim = y_arr.shape[1] if y_arr.ndim > 1 else 1
            executor.build()
            executor.compile()
            executor.train(X_arr, y_arr)

            out = {
                **merged_inputs,
                "model": executor,
                "metrics": executor.score(X, y),
            }
            outputs[nid] = out
            results[nid] = {
                "metrics": out.get("metrics"),
                "is_trained": True,
            }

    # ── Phase 3: Execute validator nodes ─────────────────────────────────
    for nid in order:
        nd = node_map[nid]
        data = nd.get("data", nd)
        category = data.get("category")

        if category != "validator":
            continue

        logger.info("Pipeline: executing node '%s' (%s)", nid, category)

        up_ids = upstream.get(nid, [])
        merged_inputs, _ = _collect_model_entries(
            up_ids, outputs, node_map,
        )
        if not merged_inputs:
            raise ValueError(
                f"Validator node '{nid}' has no upstream model outputs."
            )

        executor = _build_executor(data, seed=seed)
        out = executor.execute(merged_inputs)
        outputs[nid] = out
        results[nid] = out

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info("Pipeline: completed in %.3fs", elapsed)

    return {"node_results": results, "elapsed_s": elapsed}
