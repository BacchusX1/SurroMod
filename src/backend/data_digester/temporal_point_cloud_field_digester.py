"""
Temporal Point Cloud Field Digester
====================================
Loads temporal point cloud field data from .npz files (e.g. GRaM
challenge format).

Supports two modes:

*  **single** (default) — load one .npz file.
*  **batch** — load all .npz files from a directory.  Returns a list of
   sample dicts keyed under ``"samples"``, with simulation-level metadata
   for downstream splitting and training.

Each sample contains:
    - pos:          (N, 3)        – spatial coordinates
    - velocity_in:  (T_in, N, 3)  – input velocity field over time
    - velocity_out: (T_out, N, 3) – output velocity field over time
    - t:            (T,)          – time steps (optional)
    - idcs_airfoil: (#surf,)      – surface point indices (optional)
    - pressure:     (T, N)        – pressure field (optional)

The digester validates the sample format and returns a structured dict
suitable for downstream feature engineering and graph-based modelling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.backend.data_digester import DataDigester
from src.backend.data_digester.validators import (
    validate_temporal_point_cloud_sample,
)

logger = logging.getLogger(__name__)


def _parse_filename(name: str) -> tuple[str, str, str]:
    """Parse ``{geometry}_{sample}-{subset}`` → (geometry, sample, subset)."""
    stem = name.replace(".npz", "")
    parts = stem.split("_")
    geo_id = parts[0]
    rest = "_".join(parts[1:])
    sample_id, subset = rest.rsplit("-", 1)
    return geo_id, sample_id, subset


def _subsample_arrays(
    raw: dict[str, np.ndarray],
    max_points: int,
    rng: np.random.RandomState,
) -> dict[str, np.ndarray]:
    """Subsample a single sample to *max_points* while keeping airfoil."""
    pos = raw["pos"]
    N = pos.shape[0]
    if N <= max_points:
        return raw

    idcs_airfoil = raw.get("idcs_airfoil")
    if idcs_airfoil is not None and len(idcs_airfoil) > 0:
        # Keep as many airfoil points as feasible, fill rest with field
        n_airfoil_budget = min(len(idcs_airfoil), max_points // 2)
        airfoil_keep = rng.choice(idcs_airfoil, n_airfoil_budget, replace=False)

        field_mask = np.ones(N, dtype=bool)
        field_mask[idcs_airfoil] = False
        field_indices = np.where(field_mask)[0]
        n_field = max_points - n_airfoil_budget
        field_keep = rng.choice(field_indices, min(n_field, len(field_indices)), replace=False)

        keep = np.sort(np.concatenate([airfoil_keep, field_keep]))
    else:
        keep = np.sort(rng.choice(N, max_points, replace=False))

    out = dict(raw)
    out["pos"] = raw["pos"][keep]
    if "velocity_in" in raw:
        out["velocity_in"] = raw["velocity_in"][:, keep, :]
    if "velocity_out" in raw:
        out["velocity_out"] = raw["velocity_out"][:, keep, :]
    if "pressure" in raw:
        out["pressure"] = raw["pressure"][:, keep]

    # Remap airfoil indices
    if idcs_airfoil is not None and len(idcs_airfoil) > 0:
        old_to_new = {int(old): new for new, old in enumerate(keep)}
        new_airfoil = np.array(
            [old_to_new[int(i)] for i in airfoil_keep if int(i) in old_to_new],
            dtype=np.int32,
        )
        out["idcs_airfoil"] = new_airfoil
    return out


class TemporalPointCloudFieldDigester(DataDigester):
    """
    Pipeline-executable digester for Temporal Point Cloud Field data.

    Expected ``node_data`` keys (single mode):
        - source       : str  – path / upload ID of an .npz file
        - format_mode  : str  – ``"Temporal Point Cloud Field"``

    Additional ``hyperparams`` for batch mode:
        - batch_dir         : str  – directory of .npz files
        - max_files         : int  – limit files loaded (0 = all)
        - max_simulations   : int  – limit unique simulations (0 = all)
        - geometry_filter   : list[str] – whitelist of geometry IDs
        - max_points        : int  – subsample each mesh (0 = no limit)
    """

    def __init__(self, node_data: dict[str, Any]) -> None:
        super().__init__(node_data)
        hp = node_data.get("hyperparams", {})
        self._include_velocity_in: bool = bool(hp.get("field_select_velocity_in", True))
        self._include_velocity_out: bool = bool(hp.get("field_select_velocity_out", True))
        self._include_pos: bool = bool(hp.get("field_select_pos", True))
        self._include_t: bool = bool(hp.get("field_select_t", True))
        self._include_pressure: bool = bool(hp.get("field_select_pressure", False))

        # Batch mode
        self._batch_dir: str = str(hp.get("batch_dir", ""))
        self._max_files: int = int(hp.get("max_files", 0))
        self._max_simulations: int = int(hp.get("max_simulations", 0))
        # geometry_filter can be a list or a comma-separated string from the frontend
        raw_filter = hp.get("geometry_filter", [])
        if isinstance(raw_filter, str):
            self._geometry_filter: list[str] = [
                s.strip() for s in raw_filter.split(",") if s.strip()
            ]
        else:
            self._geometry_filter = list(raw_filter)
        self._max_points: int = int(hp.get("max_points", 0))

    # ── introspection ────────────────────────────────────────────────────

    @staticmethod
    def read_structure(path: str) -> dict[str, Any]:
        """
        Return the structure of an .npz file.

        Returns a dict with ``"format": "npz"`` and the array names
        with their shapes and dtypes.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"NPZ file not found: {path}")
        with np.load(str(p), allow_pickle=False) as npz:
            datasets: dict[str, dict[str, Any]] = {}
            for name in npz.files:
                arr = npz[name]
                datasets[name] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }
        return {"format": "npz", "datasets": datasets}

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, _inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._batch_dir:
            return self._execute_batch()
        return self._execute_single()

    def _execute_single(self) -> dict[str, Any]:
        """
        Load a single Temporal Point Cloud Field sample from an .npz file.
        """
        path = self._resolve_source()
        if not path.exists():
            raise FileNotFoundError(f"NPZ file not found: {self._source}")

        with np.load(str(path), allow_pickle=False) as npz:
            raw: dict[str, np.ndarray] = {k: npz[k] for k in npz.files}

        # Validate required keys and shapes
        validate_temporal_point_cloud_sample(raw)

        pos = raw["pos"].astype(np.float32)
        velocity_in = raw["velocity_in"].astype(np.float32)
        velocity_out = raw["velocity_out"].astype(np.float32)

        result: dict[str, Any] = {}

        if self._include_pos:
            result["pos"] = pos
        if self._include_velocity_in:
            result["velocity_in"] = velocity_in
        if self._include_velocity_out:
            result["velocity_out"] = velocity_out
        if self._include_t and "t" in raw:
            result["t"] = raw["t"].astype(np.float32)

        N = pos.shape[0]
        T_in = velocity_in.shape[0]
        T_out = velocity_out.shape[0]

        result["meta"] = {
            "format": "Temporal Point Cloud Field",
            "num_points": int(N),
            "num_input_steps": int(T_in),
            "num_output_steps": int(T_out),
        }

        logger.info(
            "TemporalPointCloudFieldDigester: loaded %s  N=%d  T_in=%d  T_out=%d",
            path.name, N, T_in, T_out,
        )
        return result

    # ── batch mode ───────────────────────────────────────────────────────

    @staticmethod
    def load_sample(
        filepath: str,
        keys: set[str] | None = None,
        max_points: int = 0,
        rng: np.random.RandomState | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Load arrays from a single ``.npz`` file on demand.

        Parameters
        ----------
        filepath   : path to the .npz file
        keys       : optional whitelist of array names to load (default: all)
        max_points : subsample mesh if > 0
        rng        : random state for subsampling reproducibility
        """
        with np.load(filepath, allow_pickle=False) as npz:
            if keys is not None:
                raw = {k: npz[k] for k in npz.files if k in keys}
            else:
                raw = {k: npz[k] for k in npz.files}

        # Cast float arrays to float32 to halve memory
        out: dict[str, np.ndarray] = {}
        for k, v in raw.items():
            if v.dtype.kind == "f":
                out[k] = np.asarray(v, dtype=np.float32)
            else:
                out[k] = v
        del raw

        if max_points > 0:
            if rng is None:
                rng = np.random.RandomState(42)
            out = _subsample_arrays(out, max_points, rng)

        return out

    def _execute_batch(self) -> dict[str, Any]:
        """
        Scan all ``.npz`` files from ``batch_dir`` and return a list of
        lightweight sample descriptors (no arrays loaded into memory).

        Downstream FE nodes and the GFF use ``load_sample()`` to read
        arrays on demand, keeping peak RAM well below data-set size.

        Returns
        -------
        dict with keys:
            ``samples`` – list of metadata-only dicts with ``filepath``
            ``num_samples`` – total count
            ``sim_ids``     – unique simulation identifiers
        """
        batch_path = Path(self._batch_dir)
        if not batch_path.is_absolute():
            # Resolve relative to project root
            batch_path = Path(__file__).resolve().parent.parent.parent.parent / self._batch_dir
        if not batch_path.is_dir():
            raise FileNotFoundError(f"Batch directory not found: {batch_path}")

        # Collect and sort .npz files
        npz_files = sorted(batch_path.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {batch_path}")

        # Apply geometry filter
        if self._geometry_filter:
            geo_set = set(self._geometry_filter)
            npz_files = [
                f for f in npz_files
                if _parse_filename(f.name)[0] in geo_set
            ]

        # Apply max_files limit
        if self._max_files > 0:
            npz_files = npz_files[: self._max_files]

        # Build lightweight sample descriptors (NO array loading)
        samples: list[dict[str, Any]] = []
        for fpath in npz_files:
            geo_id, sample_id, subset = _parse_filename(fpath.name)
            sim_id = f"{geo_id}_{sample_id}"
            samples.append({
                "sim_id": sim_id,
                "window_id": int(subset),
                "geometry_id": geo_id,
                "filename": fpath.name,
                "filepath": str(fpath),
            })

        # Apply max_simulations limit (keep all windows of selected sims)
        if self._max_simulations > 0:
            unique_sims_ordered: list[str] = []
            seen_sims: set[str] = set()
            for s in samples:
                if s["sim_id"] not in seen_sims:
                    seen_sims.add(s["sim_id"])
                    unique_sims_ordered.append(s["sim_id"])
            kept_sims = set(unique_sims_ordered[: self._max_simulations])
            samples = [s for s in samples if s["sim_id"] in kept_sims]
            logger.info(
                "TemporalPointCloudFieldDigester: max_simulations=%d → "
                "kept %d sims (%d windows)",
                self._max_simulations, len(kept_sims), len(samples),
            )

        # Validate first file and extract mesh size
        first = self.load_sample(
            samples[0]["filepath"],
            keys={"pos", "velocity_in", "velocity_out"},
        )
        validate_temporal_point_cloud_sample(first)
        N_pts = first["pos"].shape[0]
        del first

        unique_sims = sorted(set(s["sim_id"] for s in samples))

        logger.info(
            "TemporalPointCloudFieldDigester[batch]: found %d windows "
            "from %d simulations (N=%d pts%s) — lazy mode, no arrays loaded",
            len(samples),
            len(unique_sims),
            N_pts,
            f", will subsample to {self._max_points}" if self._max_points > 0 else "",
        )
        return {
            "samples": samples,
            "num_samples": len(samples),
            "sim_ids": unique_sims,
            "max_points": self._max_points,
            "meta": {
                "format": "Temporal Point Cloud Field (batch)",
                "num_windows": len(samples),
                "num_simulations": len(unique_sims),
                "num_points": N_pts,
            },
        }
