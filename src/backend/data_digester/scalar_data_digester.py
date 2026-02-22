"""
Scalar Data Digester
====================
Handles flat tabular data where every feature and label cell is a
single scalar value (e.g. the concrete-strength dataset).

Supports two file formats:
    * **CSV**  – classic tabular data read via pandas.
    * **HDF5** – scalar datasets (1-D arrays sharing the same first
      dimension) read via :class:`H5Loader`.

This is the most common data format for classical surrogate modelling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backend.data_digester import DataDigester

logger = logging.getLogger(__name__)


class ScalarDataDigester(DataDigester):
    """
    Pipeline-executable digester for scalar tabular data.

    Expected ``node_data`` keys (from the frontend input node):
        - source   : str        – path to a CSV or HDF5 file
        - features : list[str]  – column / dataset names to use as X
        - labels   : list[str]  – column / dataset names to use as y
    """

    def __init__(self, node_data: dict[str, Any]) -> None:
        super().__init__(node_data)
        self._df: pd.DataFrame | None = None

    # ── format detection ─────────────────────────────────────────────────

    def _is_h5(self) -> bool:
        """``True`` when the source file is HDF5."""
        return self.detect_format(self._source) == "h5"

    # ── introspection (static) ───────────────────────────────────────────

    @staticmethod
    def read_structure(path: str) -> dict[str, Any]:
        """
        Return the structure of a CSV or HDF5 scalar file.

        * CSV  → ``{"format": "csv", "columns": [...]}``
        * HDF5 → delegates to :meth:`H5Loader.read_structure`.
        """
        fmt = DataDigester.detect_format(path)
        if fmt == "h5":
            from src.backend.data_digester.utils.h5_loader import H5Loader
            return H5Loader.read_structure(path)

        # CSV
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        df = pd.read_csv(p, nrows=0)
        return {"format": "csv", "columns": list(df.columns)}

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, _inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Load CSV or HDF5 scalar data and split into X (features) and
        optionally y (labels).

        Returns
        -------
        dict  with keys ``X``, ``feature_names``, ``columns``, and
              optionally ``y`` and ``label_names``.
        """
        if self._is_h5():
            return self._execute_h5()
        return self._execute_csv()

    # ── CSV path ─────────────────────────────────────────────────────────

    def _execute_csv(self) -> dict[str, Any]:
        path = self._resolve_source()
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {self._source}")

        self._df = pd.read_csv(path)
        columns = list(self._df.columns)

        if not self._features and not self._labels:
            raise ValueError("No feature or label columns selected.")

        result: dict[str, Any] = {"columns": columns}

        # ── Features (optional) ──────────────────────────────────────────
        if self._features:
            missing_feat = [c for c in self._features if c not in columns]
            if missing_feat:
                raise ValueError(f"Feature column(s) not in CSV: {missing_feat}")
            X = self._df[self._features].to_numpy(dtype=np.float32)
            result["X"] = X
            result["feature_names"] = list(self._features)

        # ── Labels (optional) ────────────────────────────────────────────
        if self._labels:
            missing_lab = [c for c in self._labels if c not in columns]
            if missing_lab:
                raise ValueError(f"Label column(s) not in CSV: {missing_lab}")
            y = self._df[self._labels].to_numpy(dtype=np.float32)
            if y.shape[1] == 1:
                y = y.ravel()
            result["y"] = y
            result["label_names"] = list(self._labels)

        x_shape = result.get("X", np.empty(0)).shape if "X" in result else None
        y_shape = result.get("y", np.empty(0)).shape if "y" in result else None
        logger.info(
            "ScalarDataDigester(CSV): loaded %s  X=%s  y=%s",
            path.name, x_shape, y_shape,
        )
        return result

    # ── HDF5 path ────────────────────────────────────────────────────────

    def _execute_h5(self) -> dict[str, Any]:
        from src.backend.data_digester.utils.h5_loader import H5Loader

        path = self._resolve_source()
        if not path.exists():
            raise FileNotFoundError(f"HDF5 not found: {self._source}")

        if not self._features and not self._labels:
            raise ValueError("No feature or label datasets selected.")

        result: dict[str, Any] = {}

        with H5Loader(path) as h5:
            # ── Features ─────────────────────────────────────────────────
            if self._features:
                feat_arrays = []
                for ds_path in self._features:
                    arr = h5.read_dataset(ds_path).astype(np.float32)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    feat_arrays.append(arr)
                X = np.hstack(feat_arrays)
                result["X"] = X
                result["feature_names"] = list(self._features)

            # ── Labels ───────────────────────────────────────────────────
            if self._labels:
                lab_arrays = []
                for ds_path in self._labels:
                    arr = h5.read_dataset(ds_path).astype(np.float32)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    lab_arrays.append(arr)
                y = np.hstack(lab_arrays)
                if y.shape[1] == 1:
                    y = y.ravel()
                result["y"] = y
                result["label_names"] = list(self._labels)

        x_shape = result["X"].shape if "X" in result else None
        y_shape = result["y"].shape if "y" in result else None
        logger.info(
            "ScalarDataDigester(H5): loaded %s  X=%s  y=%s",
            path.name, x_shape, y_shape,
        )
        return result
