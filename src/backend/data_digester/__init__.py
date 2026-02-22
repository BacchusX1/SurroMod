"""
Data Digester Package
=====================
Each digester handles loading & preparing a specific data modality
(scalar CSV, time-series, 2-D / 3-D field, geometry) and acts
as a pipeline-executable unit.

The abstract ``DataDigester`` base class lives here so every
concrete digester shares the same interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DataDigester(ABC):
    """
    Abstract base class for all data digesters.

    A digester is the backend counterpart of a frontend *input* node.
    It knows how to:
        1. Discover the structure of a data source (``read_structure``).
        2. Load the data and split it into X/y arrays (``execute``).

    Subclass contract
    -----------------
    * ``__init__(node_data)`` – store any source path / user selections.
    * ``execute(inputs) -> dict`` – return at minimum ``X`` and
      ``feature_names``.  ``y`` and ``label_names`` are optional (a node
      may provide features only, labels only, or both).
    * ``read_structure(path) -> dict`` (static) – lightweight
      introspection of the file at *path*.  Returns a dict whose shape
      depends on the file format:

      **CSV**::

          {"format": "csv", "columns": ["col1", "col2", ...]}

      **HDF5**::

          {
              "format": "h5",
              "groups": {
                  "/group": {
                      "datasets": {
                          "name": {"shape": [...], "dtype": "float64"},
                      }
                  }
              }
          }
    """

    def __init__(self, node_data: dict[str, Any]) -> None:
        self._source: str = node_data.get("source", "")
        self._features: list[str] = node_data.get("features", [])
        self._labels: list[str] = node_data.get("labels", [])

    def _resolve_source(self) -> Path:
        """Resolve the source to an actual file path (supports upload IDs)."""
        # Check uploads directory first
        root = Path(__file__).resolve().parent.parent.parent.parent  # → SurroMod/
        uploads_dir = root / "uploads"
        candidate = uploads_dir / self._source
        if candidate.exists():
            return candidate
        # Fall back to raw path
        return Path(self._source)

    # ── pipeline interface ───────────────────────────────────────────────

    @abstractmethod
    def execute(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Load the data, returning at least ``X`` and ``feature_names``."""
        ...

    @staticmethod
    @abstractmethod
    def read_structure(path: str) -> dict[str, Any]:
        """
        Return a structure dict describing the contents of *path*.

        The returned dict always contains a ``"format"`` key (e.g.
        ``"csv"`` or ``"h5"``) plus format-specific metadata that the
        frontend uses to populate column/dataset selectors.
        """
        ...

    # ── convenience helpers ──────────────────────────────────────────────

    @staticmethod
    def detect_format(path: str) -> str:
        """Return ``'csv'`` or ``'h5'`` based on file extension."""
        suffix = Path(path).suffix.lower()
        if suffix in (".h5", ".hdf5", ".he5"):
            return "h5"
        return "csv"

    @classmethod
    def read_columns(cls, path: str) -> list[str]:
        """
        Convenience wrapper: return a flat list of column / dataset names.

        For CSV files this is the header row.  For HDF5 files this is the
        list of all dataset full-paths (e.g. ``"/flow/rho"``).
        """
        struct = cls.read_structure(path)
        fmt = struct.get("format", "csv")
        if fmt == "csv":
            return struct.get("columns", [])
        # H5 → flatten all dataset paths
        names: list[str] = []
        for grp_path, grp_info in struct.get("groups", {}).items():
            for ds_name in grp_info.get("datasets", {}):
                full = f"{grp_path}/{ds_name}" if grp_path != "/" else f"/{ds_name}"
                names.append(full)
        return names
