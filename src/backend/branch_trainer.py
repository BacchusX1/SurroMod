"""
Branch Trainer
==============
Detects regressor branches in the pipeline graph and orchestrates training
using differentiability-aware strategies.

Since all regressors are now PyTorch-based (including KRR), every branch
is fully differentiable and trained end-to-end with a single optimiser.

Two branch layouts are supported:

*   **Simple chains** (e.g. ``MLP -> MLP -> MLP``) are composed into a
    single ``nn.Module`` and trained with MSE.

*   **RBL chains** (e.g. ``MLP -> KRR -> RBL -> MLP -> RBL-Agg``) use an
    explicit forward pass with a composite loss:
    ``loss = MSE(y_hat, y) + lam1*MSE(z, y) + lam2*mean(r**2)``

Key data-flow invariant
-----------------------
*   ``role='transform'`` regressors pass their predictions downstream as the
    new ``X``, keeping ``y`` untouched.
*   ``role='final'`` regressors train on ``y`` and emit trained model + metrics
    for the validator.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class Branch:
    """A linear chain of regressors (and optional RBL nodes) from data
    sources to a final regressor or RBL-Aggregator."""
    #: Ordered list of node IDs in the chain (regressors + rbl + rbl_aggregator).
    node_chain: list[str] = field(default_factory=list)
    #: Ordered list of regressor-only node IDs (subset of node_chain).
    regressor_chain: list[str] = field(default_factory=list)
    #: The terminal regressor node ID.
    final_regressor: str = ""
    #: Node IDs of non-regressor upstream nodes (input / FE).
    data_sources: list[str] = field(default_factory=list)
    #: Whether this branch contains RBL / RBL-Aggregator nodes.
    has_rbl: bool = False
    #: The RBL node ID (if any).
    rbl_node: str = ""
    #: The RBL-Aggregator node ID (if any).
    rbl_aggregator: str = ""
    #: Node IDs that feed into the RBL representation (top) handle.
    rbl_representation_sources: list[str] = field(default_factory=list)


# =====================================================================
# ComposedPipeline
# =====================================================================

if nn is not None:
    class ComposedPipeline(nn.Module):
        """Sequentially chains multiple ``nn.Module`` instances."""

        def __init__(self, modules: list[Any]) -> None:
            super().__init__()
            self.stages = nn.ModuleList(modules)

        def forward(self, x: Any) -> Any:
            for stage in self.stages:
                x = stage(x)
            return x


# =====================================================================
# BranchTrainer
# =====================================================================

class BranchTrainer:
    """
    Orchestrates training of regressor branches.

    Usage::

        trainer = BranchTrainer(nodes, edges, node_map, outputs, executors)
        branches = trainer.detect_branches()
        for branch in branches:
            result = trainer.train_branch(branch, seed=42)
    """

    def __init__(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        node_map: dict[str, dict[str, Any]],
        outputs: dict[str, Any],
        executors: dict[str, Any],
    ) -> None:
        self._nodes = nodes
        self._edges = edges
        self._node_map = node_map
        self._outputs = outputs
        self._executors = executors

        # Build adjacency helpers
        self._upstream: dict[str, list[str]] = defaultdict(list)
        self._downstream: dict[str, list[str]] = defaultdict(list)
        # Handle-aware upstream for RBL nodes
        self._upstream_by_handle: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for e in edges:
            self._upstream[e["target"]].append(e["source"])
            self._downstream[e["source"]].append(e["target"])
            target_handle = e.get("targetHandle") or e.get("target_handle")
            if not target_handle:
                # Fallback: parse from React Flow edge ID
                eid = e.get("id", "")
                tid = e["target"]
                suffix = eid.split(tid)[-1] if tid in eid else ""
                target_handle = suffix if suffix in ("representations",) else "default"
            self._upstream_by_handle[e["target"]][target_handle].append(e["source"])

    # -----------------------------------------------------------------
    # Branch detection
    # -----------------------------------------------------------------

    def detect_branches(self) -> list[Branch]:
        """
        Find all regressor branches in the graph.

        A branch terminates at either:
        - An ``rbl_aggregator`` node.
        - A ``role='final'`` regressor with no downstream regressors.
        """
        branches: list[Branch] = []
        visited: set[str] = set()

        # First, look for RBL-Aggregator terminals
        for nid, nd in self._node_map.items():
            data = nd.get("data", nd)
            if data.get("category") == "rbl_aggregator":
                branch = self._build_branch_from_terminal(nid, visited)
                if branch:
                    branches.append(branch)

        # Then, look for final regressors not covered by RBL branches
        for nid, nd in self._node_map.items():
            data = nd.get("data", nd)
            if data.get("category") != "regressor":
                continue
            if nid in visited:
                continue
            role = data.get("role", "final")
            if role != "final":
                continue
            branch = self._build_branch_from_terminal(nid, visited)
            if branch:
                branches.append(branch)

        logger.info(
            "BranchTrainer: detected %d branch(es): %s",
            len(branches),
            [b.regressor_chain for b in branches],
        )
        return branches

    def _build_branch_from_terminal(
        self, terminal_nid: str, visited: set[str],
    ) -> "Branch | None":
        """Build a branch by tracing backwards from a terminal node."""
        chain: list[str] = []
        data_sources: list[str] = []
        rbl_rep_sources: list[str] = []

        self._trace_back(terminal_nid, chain, data_sources, rbl_rep_sources)
        chain.reverse()

        if not chain:
            return None

        visited.update(chain)

        regressor_chain = [
            nid for nid in chain
            if self._node_category(nid) == "regressor"
        ]
        rbl_node = ""
        rbl_agg = ""
        has_rbl = False

        for nid in chain:
            cat = self._node_category(nid)
            if cat == "rbl":
                rbl_node = nid
                has_rbl = True
            elif cat == "rbl_aggregator":
                rbl_agg = nid
                has_rbl = True

        final_reg = regressor_chain[-1] if regressor_chain else ""

        return Branch(
            node_chain=chain,
            regressor_chain=regressor_chain,
            final_regressor=final_reg,
            data_sources=data_sources,
            has_rbl=has_rbl,
            rbl_node=rbl_node,
            rbl_aggregator=rbl_agg,
            rbl_representation_sources=rbl_rep_sources,
        )

    def _trace_back(
        self,
        nid: str,
        chain: list[str],
        data_sources: list[str],
        rbl_rep_sources: list[str],
        _visited: set[str] | None = None,
    ) -> None:
        """Recursively trace upstream from *nid*, collecting relevant nodes."""
        if _visited is None:
            _visited = set()
        if nid in _visited:
            return
        _visited.add(nid)

        chain.append(nid)
        cat = self._node_category(nid)

        for uid in self._upstream.get(nid, []):
            u_cat = self._node_category(uid)
            if u_cat in ("regressor", "rbl", "rbl_aggregator"):
                # Check if this is a representation input to RBL (top handle)
                if cat == "rbl":
                    top_sources = self._upstream_by_handle[nid].get(
                        "representations", []
                    )
                    if uid in top_sources:
                        if uid not in rbl_rep_sources:
                            rbl_rep_sources.append(uid)
                        continue
                self._trace_back(uid, chain, data_sources, rbl_rep_sources, _visited)
            else:
                if uid not in data_sources:
                    data_sources.append(uid)

    def _node_category(self, nid: str) -> str:
        nd = self._node_map.get(nid, {})
        data = nd.get("data", nd)
        return data.get("category", "")

    # -----------------------------------------------------------------
    # Branch training dispatcher
    # -----------------------------------------------------------------

    def train_branch(
        self,
        branch: Branch,
        X: np.ndarray,
        y: np.ndarray,
        seed: "int | None" = None,
    ) -> dict[str, Any]:
        """Train all regressors in a branch."""
        t0 = time.perf_counter()

        single_node = len(branch.regressor_chain) == 1 and not branch.has_rbl

        if single_node:
            result = self._train_single(branch, X, y, seed)
        elif branch.has_rbl:
            result = self._train_with_rbl(branch, X, y, seed)
        else:
            result = self._train_composed(branch, X, y, seed)

        elapsed = time.perf_counter() - t0
        result["elapsed_s"] = round(elapsed, 4)
        logger.info(
            "BranchTrainer: branch %s trained in %.2fs",
            branch.regressor_chain, elapsed,
        )
        return result

    # -----------------------------------------------------------------
    # Strategy: single node
    # -----------------------------------------------------------------

    def _train_single(
        self,
        branch: Branch,
        X: np.ndarray,
        y: np.ndarray,
        seed: "int | None",
    ) -> dict[str, Any]:
        """Train a single regressor (no chaining)."""
        nid = branch.regressor_chain[0]
        executor = self._executors[nid]

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        executor._input_dim = X_arr.shape[1] if X_arr.ndim > 1 else 1
        executor._output_dim = y_arr.shape[1] if y_arr.ndim > 1 else 1

        hp_output_dim = int(executor.get_hyperparam("output_dim", 0))
        if hp_output_dim > 0:
            executor._output_dim = hp_output_dim

        executor.build()
        executor.compile()
        executor.train(X_arr, y_arr)

        return {
            "model": executor,
            "metrics": executor.score(X_arr, y_arr),
            "node_outputs": {nid: executor},
        }

    # -----------------------------------------------------------------
    # Strategy: Pure composed end-to-end (no RBL)
    # -----------------------------------------------------------------

    def _train_composed(
        self,
        branch: Branch,
        X: np.ndarray,
        y: np.ndarray,
        seed: "int | None",
    ) -> dict[str, Any]:
        """Compose all differentiable models and train end-to-end."""
        if torch is None:
            raise RuntimeError("Composed training requires PyTorch.")

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        models: list[Any] = []
        modules: list[Any] = []

        prev_dim = X_arr.shape[1] if X_arr.ndim > 1 else 1
        final_dim = y_arr.shape[1] if y_arr.ndim > 1 else 1

        for i, nid in enumerate(branch.regressor_chain):
            executor = self._executors[nid]
            is_final = (i == len(branch.regressor_chain) - 1)

            executor._input_dim = prev_dim
            if is_final:
                executor._output_dim = final_dim
            else:
                hp_out = int(executor.get_hyperparam("output_dim", 0))
                executor._output_dim = hp_out if hp_out > 0 else prev_dim

            executor.build()
            module = executor.get_torch_module()
            models.append(executor)
            modules.append(module)
            prev_dim = executor._output_dim

        composed = ComposedPipeline(modules)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        composed.to(device)

        final_executor = models[-1]
        lr = float(final_executor.get_hyperparam("learning_rate", 1e-3))
        epochs = int(final_executor.get_hyperparam("epochs", 100))
        batch_size = int(final_executor.get_hyperparam("batch_size", 32))
        wd = float(final_executor.get_hyperparam("weight_decay", 0.0))
        clip_val = float(final_executor.get_hyperparam("gradient_clipping", 0.0))

        optimizer = torch.optim.Adam(composed.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss()

        X_t = torch.as_tensor(X_arr).to(device)
        y_t = torch.as_tensor(y_arr).to(device)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(-1)

        dataset = TensorDataset(X_t, y_t)
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=generator)

        # Full-batch when chain contains KRR (kernel matrix needs all data)
        has_krr = any(
            self._executors[nid].model_type == "KRR"
            for nid in branch.regressor_chain
        )

        composed.train()
        for epoch in range(1, epochs + 1):
            if has_krr:
                optimizer.zero_grad()
                y_hat = self._forward_composed_with_krr(
                    modules, branch.regressor_chain, X_t, y_t,
                )
                loss = criterion(y_hat, y_t)
                loss.backward()
                if clip_val > 0.0:
                    nn.utils.clip_grad_norm_(composed.parameters(), clip_val)
                optimizer.step()
                epoch_loss = loss.item()
            else:
                epoch_loss = 0.0
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    y_hat = composed(X_batch)
                    loss = criterion(y_hat, y_batch)
                    loss.backward()
                    if clip_val > 0.0:
                        nn.utils.clip_grad_norm_(composed.parameters(), clip_val)
                    optimizer.step()
                    epoch_loss += loss.item() * X_batch.size(0)
                epoch_loss /= len(dataset)

            if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                logger.debug(
                    "BranchTrainer [composed]: epoch %d/%d  loss=%.6f",
                    epoch, epochs, epoch_loss,
                )

        for executor in models:
            executor._is_trained = True

        composed.eval()
        with torch.no_grad():
            if has_krr:
                preds = self._forward_composed_with_krr(
                    modules, branch.regressor_chain, X_t, y_t,
                ).cpu().numpy()
            else:
                preds = composed(X_t).cpu().numpy()
        y_np = y_arr
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.ravel()
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np.ravel()

        metrics = _compute_metrics(y_np, preds)

        return {
            "model": final_executor,
            "metrics": metrics,
            "node_outputs": {nid: exc for nid, exc in
                             zip(branch.regressor_chain, models)},
        }

    def _forward_composed_with_krr(
        self,
        modules: list[Any],
        nids: list[str],
        X: Any,
        y: Any,
    ) -> Any:
        """Forward through a composed chain, passing y to KRR modules."""
        h = X
        for module, nid in zip(modules, nids):
            executor = self._executors[nid]
            if executor.model_type == "KRR":
                h = module(h, y)
            else:
                h = module(h)
        return h

    # -----------------------------------------------------------------
    # Strategy: RBL-aware end-to-end training
    # -----------------------------------------------------------------

    def _train_with_rbl(
        self,
        branch: Branch,
        X: np.ndarray,
        y: np.ndarray,
        seed: "int | None",
    ) -> dict[str, Any]:
        """
        End-to-end training for branches with RBL / RBL-Aggregator nodes.

        Explicit forward each epoch:
        1. Pre-RBL regressors produce z (KRR gets y).
        2. RBL concatenates features [z, h_i...] and sets residual target.
        3. Post-RBL regressors produce residual r.
        4. RBL-Aggregator: y_hat = z + r.
        5. loss = MSE(y_hat, y) + lam1*MSE(z, y) + lam2*mean(r**2).
        """
        if torch is None:
            raise RuntimeError("RBL training requires PyTorch.")

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -- Identify pre/post-RBL regressors -------------------------
        rbl_nid = branch.rbl_node
        pre_rbl_nids: list[str] = []
        post_rbl_nids: list[str] = []
        before_rbl = True
        _seen: set[str] = set()

        for nid in branch.node_chain:
            if nid in _seen:
                continue
            _seen.add(nid)
            cat = self._node_category(nid)
            if nid == rbl_nid:
                before_rbl = False
                continue
            if cat == "rbl_aggregator":
                continue
            if cat == "regressor":
                if before_rbl:
                    pre_rbl_nids.append(nid)
                else:
                    post_rbl_nids.append(nid)

        # -- Build all models -----------------------------------------
        prev_dim = X_arr.shape[1] if X_arr.ndim > 1 else 1
        final_dim = y_arr.shape[1] if y_arr.ndim > 1 else 1

        all_modules: dict[str, Any] = {}
        all_executors: dict[str, Any] = {}

        # Pre-RBL
        pre_out_dims: dict[str, int] = {}
        cur_dim = prev_dim
        for i, nid in enumerate(pre_rbl_nids):
            executor = self._executors[nid]
            all_executors[nid] = executor
            executor._input_dim = cur_dim
            is_last_pre = (i == len(pre_rbl_nids) - 1)
            hp_out = int(executor.get_hyperparam("output_dim", 0))
            if hp_out > 0:
                executor._output_dim = hp_out
            elif is_last_pre:
                # Last pre-RBL model produces z which is compared to y
                executor._output_dim = final_dim
            else:
                # Intermediate models preserve dimension by default
                executor._output_dim = cur_dim
            executor.build()
            all_modules[nid] = executor.get_torch_module()
            pre_out_dims[nid] = executor._output_dim
            cur_dim = executor._output_dim

        z_dim = cur_dim

        # Representation dimensions for RBL
        rep_dims: list[int] = []
        for rep_nid in branch.rbl_representation_sources:
            if rep_nid in pre_out_dims:
                rep_dims.append(pre_out_dims[rep_nid])
            elif rep_nid in self._outputs:
                rep_x = self._outputs[rep_nid].get("X")
                if rep_x is not None:
                    rd = rep_x.shape[1] if np.asarray(rep_x).ndim > 1 else 1
                    rep_dims.append(rd)

        rbl_output_dim = z_dim + sum(rep_dims)

        # Post-RBL
        cur_dim = rbl_output_dim
        for i, nid in enumerate(post_rbl_nids):
            executor = self._executors[nid]
            all_executors[nid] = executor
            executor._input_dim = cur_dim
            is_final = (i == len(post_rbl_nids) - 1)
            if is_final:
                executor._output_dim = final_dim
            else:
                hp_out = int(executor.get_hyperparam("output_dim", 0))
                executor._output_dim = hp_out if hp_out > 0 else cur_dim
            executor.build()
            all_modules[nid] = executor.get_torch_module()
            cur_dim = executor._output_dim

        for m in all_modules.values():
            m.to(device)

        # -- RBL config -----------------------------------------------
        rbl_data = self._node_map[rbl_nid].get("data", self._node_map[rbl_nid])
        lambda_kernel = float(rbl_data.get("lambda_kernel", 1.0))
        lambda_residual = float(rbl_data.get("lambda_residual", 0.01))

        # -- Optimiser -------------------------------------------------
        all_params: list[Any] = []
        for m in all_modules.values():
            all_params.extend(m.parameters())

        if not all_params:
            raise RuntimeError("No trainable parameters in the RBL branch.")

        last_reg_nid = post_rbl_nids[-1] if post_rbl_nids else pre_rbl_nids[-1]
        cfg_executor = self._executors[last_reg_nid]
        lr = float(cfg_executor.get_hyperparam("learning_rate", 1e-3))
        epochs = int(cfg_executor.get_hyperparam("epochs", 100))
        wd = float(cfg_executor.get_hyperparam("weight_decay", 0.0))
        clip_val = float(cfg_executor.get_hyperparam("gradient_clipping", 0.0))

        optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=wd)
        criterion = nn.MSELoss()

        X_t = torch.as_tensor(X_arr).to(device)
        y_t = torch.as_tensor(y_arr).to(device)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(-1)

        # Pre-compute external representations (from data phase)
        external_reps: list[torch.Tensor] = []
        for rep_nid in branch.rbl_representation_sources:
            if rep_nid not in all_modules:
                rep_x = self._outputs.get(rep_nid, {}).get("X")
                if rep_x is not None:
                    ext_t = torch.as_tensor(
                        np.asarray(rep_x, dtype=np.float32)
                    ).to(device)
                    external_reps.append(ext_t)

        # -- Training loop ---------------------------------------------
        for m in all_modules.values():
            m.train()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # Forward: pre-RBL
            h = X_t
            node_outputs: dict[str, Any] = {}
            for nid in pre_rbl_nids:
                module = all_modules[nid]
                executor = self._executors[nid]
                if executor.model_type == "KRR":
                    h = module(h, y_t)
                else:
                    h = module(h)
                node_outputs[nid] = h

            z = h  # primary prediction

            # RBL: concatenate [z, h_rep1, h_rep2, ...]
            rbl_parts: list[Any] = [z]
            for rep_nid in branch.rbl_representation_sources:
                if rep_nid in node_outputs:
                    rbl_parts.append(node_outputs[rep_nid])
            rbl_parts.extend(external_reps)
            rbl_input = torch.cat(rbl_parts, dim=1) if len(rbl_parts) > 1 else z

            # Forward: post-RBL (residual learning)
            h = rbl_input
            for nid in post_rbl_nids:
                module = all_modules[nid]
                executor = self._executors[nid]
                if executor.model_type == "KRR":
                    y_residual = y_t - z
                    h = module(h, y_residual)
                else:
                    h = module(h)

            r = h  # residual prediction

            # RBL-Aggregator: y_hat = z + r
            y_hat = z + r

            # Composite loss
            loss_final = criterion(y_hat, y_t)
            loss_kernel = criterion(z, y_t)
            loss_residual = torch.mean(r ** 2)

            loss = (
                loss_final
                + lambda_kernel * loss_kernel
                + lambda_residual * loss_residual
            )

            loss.backward()
            if clip_val > 0.0:
                nn.utils.clip_grad_norm_(all_params, clip_val)
            optimizer.step()

            if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                logger.debug(
                    "BranchTrainer [rbl]: epoch %d/%d  "
                    "loss=%.6f  final=%.6f  kernel=%.6f  residual=%.6f",
                    epoch, epochs,
                    loss.item(), loss_final.item(),
                    loss_kernel.item(), loss_residual.item(),
                )

        # -- Mark trained ----------------------------------------------
        for executor in all_executors.values():
            executor._is_trained = True

        # -- Score -----------------------------------------------------
        for m in all_modules.values():
            m.eval()

        with torch.no_grad():
            h = X_t
            for nid in pre_rbl_nids:
                mod = all_modules[nid]
                exc = self._executors[nid]
                h = mod(h, y_t) if exc.model_type == "KRR" else mod(h)
            z_eval = h

            rbl_parts_eval: list[Any] = [z_eval]
            for rep_nid in branch.rbl_representation_sources:
                if rep_nid in node_outputs:
                    rbl_parts_eval.append(node_outputs[rep_nid].detach())
            rbl_parts_eval.extend(external_reps)
            rbl_eval = (
                torch.cat(rbl_parts_eval, dim=1)
                if len(rbl_parts_eval) > 1
                else z_eval
            )

            h = rbl_eval
            for nid in post_rbl_nids:
                mod = all_modules[nid]
                exc = self._executors[nid]
                if exc.model_type == "KRR":
                    h = mod(h, y_t - z_eval)
                else:
                    h = mod(h)
            y_hat_eval = (z_eval + h).cpu().numpy()

        y_np = y_arr
        if y_hat_eval.ndim == 2 and y_hat_eval.shape[1] == 1:
            y_hat_eval = y_hat_eval.ravel()
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np.ravel()

        metrics = _compute_metrics(y_np, y_hat_eval)

        return {
            "model": all_executors.get(branch.final_regressor, cfg_executor),
            "metrics": metrics,
            "node_outputs": all_executors,
        }


# =====================================================================
# Helpers
# =====================================================================

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute R-squared, RMSE, MAE for regression evaluation."""
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    return {"r2": round(r2, 6), "rmse": round(rmse, 6), "mae": round(mae, 6)}
