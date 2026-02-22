"""
2-D Field Data Digester
=======================
Loads 2-D mesh / field data from HDF5 files.

A typical HDF5 layout for CFD results::

    /flow/
        rho    (N, Ny, Nx)  – density field per sample
        rho_u  (N, Ny, Nx)  – x-momentum
        rho_v  (N, Ny, Nx)  – y-momentum
        e      (N, Ny, Nx)  – energy
        omega  (N, Ny, Nx)  – specific dissipation rate

The user selects which datasets to load as features (X) and/or
labels (y) via the frontend Inspector.  Each selected dataset is
read as-is (preserving its natural shape) so downstream nodes
(regressors, data splitters, etc.) can decide how to reshape.

For scalar regressors that need flat input, a :class:`DataSplitter`
or :class:`PCA` node can be inserted in between.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.backend.data_digester import DataDigester

logger = logging.getLogger(__name__)


class TwoDFieldDigester(DataDigester):
    """
    Pipeline-executable digester for 2-D spatial field data stored in HDF5.

    Expected ``node_data`` keys (from the frontend input node):
        - source      : str        – path / upload ID of the HDF5 file
        - features    : list[str]  – full dataset paths to load as X
                                     (e.g. ``["/flow/rho", "/flow/rho_u"]``)
        - labels      : list[str]  – full dataset paths to load as y
    """

    def __init__(self, node_data: dict[str, Any]) -> None:
        super().__init__(node_data)

    # ── introspection ────────────────────────────────────────────────────

    @staticmethod
    def read_structure(path: str) -> dict[str, Any]:
        """Delegate to :class:`H5Loader` for HDF5 introspection."""
        from src.backend.data_digester.utils.h5_loader import H5Loader
        return H5Loader.read_structure(path)

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, _inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Load selected 2-D field datasets from an HDF5 file.

        Returns
        -------
        dict with keys:
            - ``X``             : ndarray – features.  If multiple datasets are
              selected they are stacked along a new *channel* axis giving
              shape ``(N, C, Ny, Nx)`` where C = number of selected fields.
              A single field yields ``(N, Ny, Nx)``.
            - ``feature_names`` : list[str] – dataset paths used as features.
            - ``y``             : ndarray (optional) – labels, same stacking.
            - ``label_names``   : list[str] (optional).
        """
        from src.backend.data_digester.utils.h5_loader import H5Loader

        path = self._resolve_source()
        if not path.exists():
            raise FileNotFoundError(f"HDF5 not found: {self._source}")

        if not self._features and not self._labels:
            raise ValueError("No feature or label datasets selected for 2D field.")

        result: dict[str, Any] = {}

        with H5Loader(path) as h5:
            # ── Features ─────────────────────────────────────────────────
            if self._features:
                arrays = [
                    h5.read_dataset(ds).astype(np.float32)
                    for ds in self._features
                ]
                X = self._stack_fields(arrays)
                result["X"] = X
                result["feature_names"] = list(self._features)

            # ── Labels ───────────────────────────────────────────────────
            if self._labels:
                arrays = [
                    h5.read_dataset(ds).astype(np.float32)
                    for ds in self._labels
                ]
                y = self._stack_fields(arrays)
                result["y"] = y
                result["label_names"] = list(self._labels)

        x_shape = result["X"].shape if "X" in result else None
        y_shape = result["y"].shape if "y" in result else None
        logger.info(
            "TwoDFieldDigester: loaded %s  X=%s  y=%s",
            path.name, x_shape, y_shape,
        )
        return result

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _stack_fields(arrays: list[np.ndarray]) -> np.ndarray:
        """
        Stack multiple field arrays along a channel axis.

        Single field  → returned as-is  ``(N, Ny, Nx)``
        Multiple      → stacked to      ``(N, C, Ny, Nx)``
        """
        if len(arrays) == 1:
            return arrays[0]
        # Ensure all arrays have the same shape
        ref_shape = arrays[0].shape
        for i, a in enumerate(arrays[1:], 1):
            if a.shape != ref_shape:
                raise ValueError(
                    f"Field shape mismatch: field 0 has shape {ref_shape} "
                    f"but field {i} has shape {a.shape}"
                )
        # Stack along a new axis=1 → (N, C, Ny, Nx)
        return np.stack(arrays, axis=1)
