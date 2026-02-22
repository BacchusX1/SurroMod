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

_MODEL_CONTEXT_KEYS = ("X", "y", "y_scaler", "label_names", "feature_names")
"""Keys that vary per model and must NOT be shared across models."""


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
        # Future digesters:
        # if input_kind == "time_series":
        #     from src.backend.data_digester.time_series_digester import TimeSeriesDigester
        #     return TimeSeriesDigester(node_data)
        # if input_kind == "2d_field":
        #     from src.backend.data_digester.2d_data_digester import TwoDDataDigester
        #     return TwoDDataDigester(node_data)
        # if input_kind == "3d_field":
        #     from src.backend.data_digester.3d_data_digester import ThreeDDataDigester
        #     return ThreeDDataDigester(node_data)
        # if input_kind == "step":
        #     from src.backend.data_digester.step_data_digester import StepDataDigester
        #     return StepDataDigester(node_data)
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

    for nid in order:
        nd = node_map[nid]
        data = nd.get("data", nd)  # frontend sends {id, type, data, ...}
        category = data.get("category")

        logger.info("Pipeline: executing node '%s' (%s)", nid, category)

        # ── Build merged inputs from upstream ────────────────────────────
        up_ids = upstream.get(nid, [])

        if category == "validator":
            # Per-model context: each regressor carries its OWN X/y/y_scaler
            merged_inputs, _ = _collect_model_entries(
                up_ids, outputs, node_map,
            )
            if not merged_inputs:
                raise ValueError(
                    f"Validator node '{nid}' has no upstream model outputs."
                )
        else:
            # Smart merge: concatenates X arrays, extends feature_names
            upstream_dicts = [
                outputs[uid]
                for uid in up_ids
                if uid in outputs and isinstance(outputs[uid], dict)
            ]
            merged_inputs = _merge_upstream_outputs(upstream_dicts)

        executor = _build_executor(data, seed=seed)

        # Execute (regressor needs special handling for build→compile→train)
        if category == "regressor":
            from src.backend.predictors.model_base import ModelBase
            assert isinstance(executor, ModelBase)

            X = merged_inputs.get("X")
            y = merged_inputs.get("y")
            if X is None or y is None:
                raise ValueError(f"Regressor node '{nid}' got no X or y from upstream.")

            # Let train() handle build/compile (it auto-detects dimensions)
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
        else:
            out = executor.execute(merged_inputs)

        outputs[nid] = out

        # Build a JSON-serialisable result snapshot for this node
        if category == "validator":
            # Validator results go directly to the frontend
            results[nid] = out
        elif category == "regressor":
            results[nid] = {
                "metrics": out.get("metrics"),
                "is_trained": True,
            }
        else:
            results[nid] = {"status": "ok"}

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info("Pipeline: completed in %.3fs", elapsed)

    return {"node_results": results, "elapsed_s": elapsed}
