"""
2-D Geometry Data Digester
==========================
Loads 2-D shape representations from HDF5 files.

Supported geometry representations (stored under a ``/shape/`` group
or similar):

* **Landmarks** – ``(N, n_pts, 2)``  dense point clouds on the shape
  surface.  Each point has (x, y) coordinates.
* **CST** – ``(N, n_cst)``  Class-Shape-Transformation coefficients
  (typically 19 parameters for upper + lower surface).
* **Bezier** – ``(N, n_ctrl)``  Bezier control-point weights
  (typically 15 parameters).

The digester outputs standard ``X`` / ``feature_names`` (and optionally
``y`` / ``label_names``) so the data can flow directly into a regressor
or through an optional :class:`GeometrySampler` feature-engineering node.

CST and Bezier representations are already compact feature vectors and
work directly with flat regressors (MLP, KRR, Polynomial, …).
Landmarks ``(N, n_pts, 2)`` preserve their natural 3-D tensor shape;
a :class:`GeometrySampler` or :class:`DataSplitter` can flatten or
resample them if needed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.backend.data_digester import DataDigester

logger = logging.getLogger(__name__)


class Geometry2DDigester(DataDigester):
    """
    Pipeline-executable digester for 2-D geometry data stored in HDF5.

    Expected ``node_data`` keys:
        - source       : str        – path / upload ID of the HDF5 file
        - features     : list[str]  – dataset paths to load as X
                                      (e.g. ``["/shape/cst"]``)
        - labels       : list[str]  – dataset paths to load as y (optional)
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
        Load geometry datasets from HDF5.

        Returns
        -------
        dict with keys:
            ``X``             – ndarray.  Shape depends on representation:
                                CST ``(N, 19)``, Bezier ``(N, 15)``,
                                Landmarks ``(N, n_pts, 2)``.
            ``feature_names`` – list[str] of dataset paths used.
            ``y``             – ndarray (optional).
            ``label_names``   – list[str] (optional).
        """
        from src.backend.data_digester.utils.h5_loader import H5Loader

        path = self._resolve_source()
        if not path.exists():
            raise FileNotFoundError(f"HDF5 not found: {self._source}")

        if not self._features and not self._labels:
            raise ValueError("No feature or label datasets selected for 2D geometry.")

        result: dict[str, Any] = {}

        with H5Loader(path) as h5:
            # ── Features ─────────────────────────────────────────────────
            if self._features:
                if len(self._features) == 1:
                    X = h5.read_dataset(self._features[0]).astype(np.float32)
                else:
                    # Multiple geometry datasets → try to hstack if shapes
                    # are compatible (same N, all 2-D), otherwise stack
                    arrays = [
                        h5.read_dataset(ds).astype(np.float32)
                        for ds in self._features
                    ]
                    X = self._combine_arrays(arrays)
                result["X"] = X
                result["feature_names"] = list(self._features)

            # ── Labels ───────────────────────────────────────────────────
            if self._labels:
                if len(self._labels) == 1:
                    y = h5.read_dataset(self._labels[0]).astype(np.float32)
                else:
                    arrays = [
                        h5.read_dataset(ds).astype(np.float32)
                        for ds in self._labels
                    ]
                    y = self._combine_arrays(arrays)
                if y.ndim == 2 and y.shape[1] == 1:
                    y = y.ravel()
                result["y"] = y
                result["label_names"] = list(self._labels)

        x_shape = result["X"].shape if "X" in result else None
        y_shape = result["y"].shape if "y" in result else None
        logger.info(
            "Geometry2DDigester: loaded %s  X=%s  y=%s",
            path.name, x_shape, y_shape,
        )
        return result

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _combine_arrays(arrays: list[np.ndarray]) -> np.ndarray:
        """
        Combine multiple geometry arrays intelligently.

        * All 2-D with same N → ``np.hstack``  (feature concatenation).
        * All same shape        → ``np.stack``  (channel axis).
        * Otherwise             → raise.
        """
        if len(arrays) == 1:
            return arrays[0]

        n_samples = arrays[0].shape[0]
        for i, a in enumerate(arrays):
            if a.shape[0] != n_samples:
                raise ValueError(
                    f"Sample count mismatch: array 0 has {n_samples} samples "
                    f"but array {i} has {a.shape[0]}"
                )

        # All 2-D → hstack (simple feature concatenation)
        if all(a.ndim == 2 for a in arrays):
            return np.hstack(arrays)

        # All same shape → stack along new channel axis
        ref_shape = arrays[0].shape
        if all(a.shape == ref_shape for a in arrays):
            return np.stack(arrays, axis=1)

        raise ValueError(
            f"Cannot automatically combine geometry arrays with shapes "
            f"{[a.shape for a in arrays]}.  "
            f"Select a single geometry representation or use compatible shapes."
        )
