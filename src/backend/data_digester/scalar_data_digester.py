"""
Scalar Data Digester
====================
Handles flat tabular CSV data where every feature and label cell is a
single scalar value (e.g. the concrete-strength dataset).

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
        - source   : str        – path to a CSV file
        - features : list[str]  – column names to use as X
        - labels   : list[str]  – column names to use as y
    """

    def __init__(self, node_data: dict[str, Any]) -> None:
        super().__init__(node_data)
        self._df: pd.DataFrame | None = None

    # ── static helper ────────────────────────────────────────────────────

    @staticmethod
    def read_columns(path: str) -> list[str]:
        """Return column names from a CSV file (header-only read)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        df = pd.read_csv(p, nrows=0)
        return list(df.columns)

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, _inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Load the CSV and split into X (features) and y (labels).

        Returns
        -------
        dict  with keys ``X``, ``feature_names``, ``columns``, and
              optionally ``y`` and ``label_names`` (when label columns
              are selected).
        """
        path = self._resolve_source()
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {self._source}")

        self._df = pd.read_csv(path)
        columns = list(self._df.columns)

        if not self._features:
            raise ValueError("No feature columns selected.")

        missing_feat = [c for c in self._features if c not in columns]
        if missing_feat:
            raise ValueError(f"Feature column(s) not in CSV: {missing_feat}")

        X = self._df[self._features].to_numpy(dtype=np.float32)

        result: dict[str, Any] = {
            "X": X,
            "feature_names": self._features,
            "columns": columns,
        }

        # Labels are optional — a purely-features input node is valid when
        # another upstream node provides the labels.
        if self._labels:
            missing_lab = [c for c in self._labels if c not in columns]
            if missing_lab:
                raise ValueError(f"Label column(s) not in CSV: {missing_lab}")
            y = self._df[self._labels].to_numpy(dtype=np.float32)
            if y.shape[1] == 1:
                y = y.ravel()
            result["y"] = y
            result["label_names"] = self._labels
            logger.info(
                "ScalarDataDigester: loaded %s  X=%s  y=%s",
                path.name, X.shape, y.shape,
            )
        else:
            logger.info(
                "ScalarDataDigester: loaded %s  X=%s  (features only, no labels)",
                path.name, X.shape,
            )

        return result
