"""
Data Digester Package
=====================
Each digester handles loading & preparing a specific data modality
(scalar CSV, time-series, 2-D / 3-D field, STEP geometry) and acts
as a pipeline-executable unit.

The abstract ``DataDigester`` base class lives here so every
concrete digester shares the same interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class DataDigester(ABC):
    """
    Abstract base class for all data digesters.

    A digester is the backend counterpart of a frontend *input* node.
    It knows how to:
        1. Discover the structure of a data source (``read_columns``).
        2. Load the data and split it into X/y arrays (``execute``).

    Subclass contract
    -----------------
    * ``__init__(node_data)`` – store any source path / user selections.
    * ``execute(inputs) -> dict`` – return at minimum ``X``, ``y``,
      ``feature_names``, ``label_names``.
    * ``read_columns(path) -> list[str]`` (static) – lightweight
      header-only introspection of the file at *path*.
    """

    def __init__(self, node_data: dict[str, Any]) -> None:
        self._source: str = node_data.get("source", "")
        self._features: list[str] = node_data.get("features", [])
        self._labels: list[str] = node_data.get("labels", [])

    def _resolve_source(self) -> Path:
        """Resolve the source to an actual file path (supports upload IDs)."""
        from pathlib import Path as _P

        # Check uploads directory first
        root = _P(__file__).resolve().parent.parent.parent.parent  # → SurroMod/
        uploads_dir = root / "uploads"
        candidate = uploads_dir / self._source
        if candidate.exists():
            return candidate
        # Fall back to raw path
        return _P(self._source)

    # ── pipeline interface ───────────────────────────────────────────────

    @abstractmethod
    def execute(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Load the data, returning at least ``X`` and ``y``."""
        ...

    @staticmethod
    @abstractmethod
    def read_columns(path: str) -> list[str]:
        """Return column / field names from *path* without loading data."""
        ...
