"""
HDF5 Loader Utility
===================
Shared helper for opening, inspecting and reading HDF5 files.
Used by all HDF5-backed data digesters (2D field, scalar, geometry).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import h5py as _h5py

# h5py is an optional dependency — imported lazily so the rest of the
# codebase works even when only CSV pipelines are used.
try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]


def _require_h5py() -> None:
    """Raise a clear error if h5py is not installed."""
    if h5py is None:
        raise ImportError(
            "h5py is required for HDF5 support.  "
            "Install it with:  pip install h5py"
        )


class H5Loader:
    """
    Context-manager wrapper around an HDF5 file.

    Usage::

        with H5Loader("data.h5") as h5:
            struct = h5.structure()
            arr = h5.read_dataset("/flow/rho")

    All public methods can also be called as module-level statics that
    open and close the file automatically (convenience for one-shot reads).
    """

    def __init__(self, path: str | Path) -> None:
        _require_h5py()
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self._path}")
        self._file: h5py.File | None = None

    # ── context manager ──────────────────────────────────────────────────

    def __enter__(self) -> H5Loader:
        self._file = h5py.File(self._path, "r")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError(
                "H5Loader must be used as a context manager: "
                "'with H5Loader(path) as h5: ...'"
            )
        return self._file

    # ── introspection ────────────────────────────────────────────────────

    def structure(self) -> dict[str, Any]:
        """
        Return a nested dict describing every group and dataset.

        Returns
        -------
        dict of the form::

            {
                "format": "h5",
                "groups": {
                    "/group_name": {
                        "datasets": {
                            "dataset_name": {
                                "shape": (N, ...),
                                "dtype": "float64",
                            },
                        }
                    },
                    ...
                }
            }

        Top-level datasets (not inside any group) are placed under the
        ``"/"`` key.
        """
        result: dict[str, Any] = {"format": "h5", "groups": {}}

        def _visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                # Determine the parent group path
                parts = name.rsplit("/", 1)
                if len(parts) == 2:
                    group_path = "/" + parts[0]
                    ds_name = parts[1]
                else:
                    group_path = "/"
                    ds_name = parts[0]

                grp = result["groups"].setdefault(group_path, {"datasets": {}})
                grp["datasets"][ds_name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }

        self.file.visititems(_visit)

        # If nothing was found under groups, check root-level datasets
        if not result["groups"]:
            for key in self.file:
                obj = self.file[key]
                if isinstance(obj, h5py.Dataset):
                    grp = result["groups"].setdefault("/", {"datasets": {}})
                    grp["datasets"][key] = {
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                    }

        return result

    def list_datasets(self, group: str = "/") -> list[str]:
        """Return dataset names within *group*."""
        grp = self.file[group]
        if not isinstance(grp, h5py.Group):
            raise ValueError(f"'{group}' is not a group")
        return [k for k in grp if isinstance(grp[k], h5py.Dataset)]

    def list_groups(self) -> list[str]:
        """Return all group paths in the file."""
        groups: list[str] = ["/"]

        def _visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group):
                groups.append("/" + name)

        self.file.visititems(_visitor)
        return groups

    # ── reading ──────────────────────────────────────────────────────────

    def read_dataset(self, path: str) -> np.ndarray:
        """
        Read a single dataset by its full path (e.g. ``"/flow/rho"``).

        Returns the data as a NumPy array.
        """
        ds = self.file[path]
        if not isinstance(ds, h5py.Dataset):
            raise ValueError(f"'{path}' is not a dataset")
        return ds[()]

    def read_datasets(self, paths: list[str]) -> dict[str, np.ndarray]:
        """
        Read multiple datasets at once.

        Parameters
        ----------
        paths : list of full dataset paths.

        Returns
        -------
        dict mapping each path to its NumPy array.
        """
        return {p: self.read_dataset(p) for p in paths}

    def dataset_shape(self, path: str) -> tuple[int, ...]:
        """Return the shape of a dataset without reading its data."""
        ds = self.file[path]
        if not isinstance(ds, h5py.Dataset):
            raise ValueError(f"'{path}' is not a dataset")
        return ds.shape

    def dataset_dtype(self, path: str) -> np.dtype:
        """Return the dtype of a dataset without reading its data."""
        ds = self.file[path]
        if not isinstance(ds, h5py.Dataset):
            raise ValueError(f"'{path}' is not a dataset")
        return ds.dtype

    # ── static convenience ───────────────────────────────────────────────

    @staticmethod
    def read_structure(path: str | Path) -> dict[str, Any]:
        """
        One-shot: open a file, return its structure dict, then close.
        """
        with H5Loader(path) as h5:
            return h5.structure()

    @staticmethod
    def read_one(path: str | Path, dataset: str) -> np.ndarray:
        """
        One-shot: open a file, read one dataset, close.
        """
        with H5Loader(path) as h5:
            return h5.read_dataset(dataset)
