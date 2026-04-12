"""
Field Slice Plot
================
Generate 2-D slice visualisations of 3-D velocity fields for
ground truth, prediction, and absolute error comparison.

Uses a mesh-faithful rendering approach:
- High-resolution contourf (not imshow on a coarse grid)
- Solid body detection via DBSCAN + ConvexHull on airfoil surface points
- White fill + black outline per detected body
- Proper aspect ratio (equal)

Supports three modes:
- **single**: one comparison plot for the entire domain
- **train_test_comparison**: separate plots for train/test point subsets
- **video_comparison**: animated GIFs per sample over all T_out timesteps
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Output directory for generated plots
_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "proj_dir" / "outputs"


def _ensure_output_dir() -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


# ── Solid-body detection (ported from analyze_data.ipynb) ────────────────────

def _detect_body_polygons(
    pts_2d: np.ndarray,
    airfoil_local_mask: np.ndarray,
) -> list[tuple[Any, np.ndarray]]:
    """
    Detect individual solid bodies from surface points in a 2-D slice.

    Parameters
    ----------
    pts_2d            : (M, 2) — all in-slice point positions (2-D projection)
    airfoil_local_mask: (M,) bool — which of those points are airfoil/surface

    Returns
    -------
    List of (MplPath, hull_pts_closed) — one entry per detected body.
    hull_pts_closed is (K+1, 2) with the first point repeated at the end.
    """
    from matplotlib.path import Path as MplPath
    from scipy.spatial import ConvexHull
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    af_pts = pts_2d[airfoil_local_mask]
    if len(af_pts) < 10:
        return []

    # Adaptive DBSCAN: eps ~ 15× median nearest-neighbour distance
    nn = NearestNeighbors(n_neighbors=2).fit(af_pts)
    d1 = nn.kneighbors(af_pts)[0][:, 1]
    eps = float(np.median(d1) * 15)
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(af_pts)

    bodies: list[tuple[Any, np.ndarray]] = []
    for lab in sorted(set(labels) - {-1}):
        cluster = af_pts[labels == lab]
        if len(cluster) < 10:
            continue
        try:
            hull = ConvexHull(cluster)
        except Exception:
            continue
        hull_pts = cluster[hull.vertices]
        hull_pts_closed = np.vstack([hull_pts, hull_pts[:1]])
        bodies.append((MplPath(hull_pts_closed), hull_pts_closed))

    return bodies


def _apply_body_mask(
    grid_h: np.ndarray,
    grid_v: np.ndarray,
    bodies: list[tuple[Any, np.ndarray]],
) -> np.ndarray:
    """Return a boolean mask (same shape as grid_h) — True inside any body."""
    from matplotlib.path import Path as MplPath  # noqa: F401 (already imported in detect)

    grid_pts = np.column_stack([grid_h.ravel(), grid_v.ravel()])
    mask = np.zeros(grid_h.shape, dtype=bool)
    for body_path, _ in bodies:
        mask |= body_path.contains_points(grid_pts).reshape(grid_h.shape)
    return mask


# ── Shared mesh-faithful rendering primitive ──────────────────────────────────

def _render_mesh_faithful(
    ax: Any,
    pts_2d: np.ndarray,
    values: np.ndarray,
    bodies: list[tuple[Any, np.ndarray]],
    body_mask: np.ndarray,
    grid_h: np.ndarray,
    grid_v: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str,
    contour_levels: int,
    body_outline_lw: float,
    show_mesh_points: bool,
) -> Any:
    """
    Render a scalar field on a 2-D slice with mesh-faithful contourf and
    solid body masking.

    Parameters
    ----------
    ax              : matplotlib Axes
    pts_2d          : (M, 2) scattered point positions in the slice plane
    values          : (M,) scalar field at those points
    bodies          : list of (MplPath, hull_pts_closed) from _detect_body_polygons
    body_mask       : (grid_h.shape) bool — pre-computed interior mask
    grid_h, grid_v  : meshgrid arrays (shape R×R)
    vmin, vmax      : colour scale limits
    cmap            : matplotlib colormap name
    contour_levels  : number of contourf bands
    body_outline_lw : linewidth for black body outline
    show_mesh_points: if True, overlay scatter of raw mesh points

    Returns
    -------
    mappable suitable for colorbar
    """
    from scipy.interpolate import griddata

    grid_vals = griddata(
        pts_2d, values, (grid_h, grid_v), method="linear",
    )
    # Apply solid-body NaN mask
    grid_vals[body_mask] = np.nan

    # Fill any remaining NaNs at boundary via nearest interpolation
    if np.any(np.isnan(grid_vals)):
        nearest = griddata(pts_2d, values, (grid_h, grid_v), method="nearest")
        grid_vals = np.where(np.isnan(grid_vals), nearest, grid_vals)
        # Re-apply body mask after nearest fill (don't colour inside bodies)
        grid_vals[body_mask] = np.nan

    levels = np.linspace(vmin, vmax, max(contour_levels, 2))
    mappable = ax.contourf(
        grid_h, grid_v, grid_vals,
        levels=levels, cmap=cmap, extend="both",
    )

    # White fill + black border for each body
    for _, hull_pts_closed in bodies:
        ax.fill(hull_pts_closed[:, 0], hull_pts_closed[:, 1],
                color="white", zorder=3)
        ax.plot(hull_pts_closed[:, 0], hull_pts_closed[:, 1],
                "k-", linewidth=body_outline_lw, zorder=4)

    # Optional: overlay raw mesh points
    if show_mesh_points:
        ax.scatter(pts_2d[:, 0], pts_2d[:, 1],
                   s=0.4, c="black", alpha=0.25, linewidths=0, zorder=2)

    ax.set_aspect("equal")
    return mappable


class FieldSlicePlot:
    """
    Create 2-D slice visualisations of 3-D velocity fields.

    Hyperparams
    -----------
    slice_plane         : ``"xy"`` | ``"yz"`` | ``"xz"``        (default ``"xz"``)
    slice_quantile      : float — quantile along slice axis       (default 0.5)
    slice_position      : float — absolute position (used when slice_quantile <= 0)
    field_component     : ``"magnitude"`` | ``"vx"`` | ``"vy"`` | ``"vz"``
    timestep            : int — which output time step to visualise (default 0)
    iso_grid_res        : int — contourf grid resolution          (default 400)
    grid_resolution     : int — legacy alias for iso_grid_res
    colormap            : str — matplotlib colormap for field     (default ``"turbo"``)
    error_colormap      : str — colormap for absolute error panel (default ``"hot"``)
    contour_levels      : int — number of contourf level bands    (default 20)
    body_outline_lw     : float — linewidth of solid body outline (default 1.0)
    show_mesh_points    : bool — overlay raw mesh scatter         (default False)
    mode                : ``"single"`` | ``"train_test_comparison"`` | ``"video_comparison"``
    num_samples_per_set : int — max random samples per train/test (default 2)
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = hyperparams or {}
        self._slice_plane: str = str(hp.get("slice_plane", "xz"))
        self._slice_position: float = float(hp.get("slice_position", 0.0))
        self._slice_quantile: float = float(hp.get("slice_quantile", 0.5))
        self._field_component: str = str(hp.get("field_component", "magnitude"))
        self._timestep: int = int(hp.get("timestep", 0))
        # iso_grid_res supersedes legacy grid_resolution
        self._iso_grid_res: int = int(
            hp.get("iso_grid_res", hp.get("grid_resolution", 400))
        )
        self._colormap: str = str(hp.get("colormap", "turbo"))
        self._error_colormap: str = str(hp.get("error_colormap", "hot"))
        self._contour_levels: int = int(hp.get("contour_levels", 20))
        self._body_outline_lw: float = float(hp.get("body_outline_lw", 1.0))
        self._show_mesh_points: bool = bool(hp.get("show_mesh_points", False))
        self._mode: str = str(hp.get("mode", "single"))
        self._num_samples_per_set: int = int(hp.get("num_samples_per_set", 2))
        self._seed = seed

    # ── public pipeline interface ────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._mode == "video_comparison":
            return self._execute_video(inputs)
        if self._mode == "train_test_comparison":
            return self._execute_train_test(inputs)
        return self._execute_single(inputs)

    # ── axis / slice helpers ─────────────────────────────────────────────

    def _get_axes(self) -> tuple[tuple[int, int], int]:
        """Return (plane_axis_0, plane_axis_1), slice_axis."""
        if self._slice_plane == "xy":
            return (0, 1), 2
        if self._slice_plane == "yz":
            return (1, 2), 0
        if self._slice_plane == "xz":
            return (0, 2), 1
        return (0, 2), 1

    def _get_slice_coord(self, pos: np.ndarray, axis: int) -> float:
        if self._slice_quantile > 0:
            return float(np.quantile(pos[:, axis], self._slice_quantile))
        return self._slice_position

    def _compute_tolerance(self, pos: np.ndarray, axis: int) -> float:
        vals = pos[:, axis]
        span = vals.max() - vals.min()
        return max(span * 0.02, 1e-6)

    def _extract_component(self, vel: np.ndarray) -> np.ndarray:
        """Extract a scalar field from (N, 3) velocity."""
        if self._field_component == "vx":
            return vel[:, 0]
        if self._field_component == "vy":
            return vel[:, 1]
        if self._field_component == "vz":
            return vel[:, 2]
        return np.linalg.norm(vel, axis=-1)

    def _get_slice_mask(
        self, pos: np.ndarray, axis: int, coord: float | None = None,
    ) -> np.ndarray:
        if coord is None:
            coord = self._get_slice_coord(pos, axis)
        tol = self._compute_tolerance(pos, axis)
        mask = np.abs(pos[:, axis] - coord) < tol
        if mask.sum() < 50:
            mask = np.abs(pos[:, axis] - coord) < tol * 3
        return mask

    def _build_grid(
        self, pts_2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a meshgrid over the bounding box of pts_2d."""
        res = self._iso_grid_res
        xi = np.linspace(pts_2d[:, 0].min(), pts_2d[:, 0].max(), res)
        yi = np.linspace(pts_2d[:, 1].min(), pts_2d[:, 1].max(), res)
        return np.meshgrid(xi, yi)

    # ── shared per-panel render helper ───────────────────────────────────

    def _render_panel(
        self,
        ax: Any,
        pts_2d: np.ndarray,
        values: np.ndarray,
        bodies: list,
        body_mask: np.ndarray,
        grid_h: np.ndarray,
        grid_v: np.ndarray,
        vmin: float,
        vmax: float,
        cmap: str,
        title: str,
        ax_labels: tuple[str, str],
    ) -> Any:
        import matplotlib.pyplot as plt  # noqa: F401

        mappable = _render_mesh_faithful(
            ax=ax,
            pts_2d=pts_2d,
            values=values,
            bodies=bodies,
            body_mask=body_mask,
            grid_h=grid_h,
            grid_v=grid_v,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            contour_levels=self._contour_levels,
            body_outline_lw=self._body_outline_lw,
            show_mesh_points=self._show_mesh_points,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
        return mappable

    # ── single-plot mode ────────────────────────────────────────────────

    def _execute_single(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pos = np.asarray(inputs["pos"], dtype=np.float32)
        pred = np.asarray(inputs["predicted_velocity_out"], dtype=np.float32)
        target = np.asarray(inputs["velocity_out"], dtype=np.float32)
        idcs_airfoil = inputs.get("idcs_airfoil")

        t = min(self._timestep, pred.shape[0] - 1, target.shape[0] - 1)
        pred_t, tgt_t = pred[t], target[t]

        plane_axes, slice_axis = self._get_axes()
        ax0, ax1 = plane_axes
        slice_coord = self._get_slice_coord(pos, slice_axis)
        mask = self._get_slice_mask(pos, slice_axis, slice_coord)

        if mask.sum() < 10:
            logger.warning("FieldSlicePlot: insufficient points in slice.")
            return {**inputs, "slice_plot_paths": [], "slice_plot_meta": {"error": "insufficient points"}}

        pts_2d = pos[mask][:, [ax0, ax1]]
        pred_vals = self._extract_component(pred_t[mask])
        tgt_vals = self._extract_component(tgt_t[mask])
        err_vals = np.abs(pred_vals - tgt_vals)

        # Determine which slice points are airfoil points
        if idcs_airfoil is not None and len(idcs_airfoil) > 0:
            af_set = set(int(i) for i in idcs_airfoil)
            local_idcs = np.where(mask)[0]
            airfoil_local = np.array([i in af_set for i in local_idcs])
        else:
            airfoil_local = np.zeros(mask.sum(), dtype=bool)

        bodies = _detect_body_polygons(pts_2d, airfoil_local)
        grid_h, grid_v = self._build_grid(pts_2d)
        body_mask = _apply_body_mask(grid_h, grid_v, bodies)

        vmin = float(min(np.nanmin(tgt_vals), np.nanmin(pred_vals)))
        vmax = float(max(np.nanmax(tgt_vals), np.nanmax(pred_vals)))
        emax = float(np.nanmax(err_vals)) or 1e-6

        ax_labels = {"xy": ("x", "y"), "yz": ("y", "z"), "xz": ("x", "z")}.get(
            self._slice_plane, ("h", "v")
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, vals, title, vm, vx, cmap in zip(
            axes,
            [tgt_vals, pred_vals, err_vals],
            ["Ground Truth", "Prediction", "Absolute Error"],
            [vmin, vmin, 0.0],
            [vmax, vmax, emax],
            [self._colormap, self._colormap, self._error_colormap],
        ):
            mp = self._render_panel(ax, pts_2d, vals, bodies, body_mask,
                                    grid_h, grid_v, vm, vx, cmap, title, ax_labels)
            plt.colorbar(mp, ax=ax, shrink=0.8)

        fig.suptitle(
            f"t={t}  |  {self._field_component}  |  {self._slice_plane}-plane "
            f"@ {slice_coord:.3f}  ({mask.sum()} pts)",
            fontsize=12,
        )
        plt.tight_layout()

        out_dir = _ensure_output_dir()
        uid = uuid.uuid4().hex[:8]
        path = out_dir / f"slice_{self._slice_plane}_{uid}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        result: dict[str, Any] = {**inputs}
        result["slice_plot_paths"] = [str(path)]
        result["slice_plot_meta"] = {
            "slice_plane": self._slice_plane,
            "slice_position": float(slice_coord),
            "field_component": self._field_component,
            "timestep": t,
            "num_points_in_slice": int(mask.sum()),
            "num_bodies": len(bodies),
            "iso_grid_res": self._iso_grid_res,
        }
        logger.info("FieldSlicePlot[single]: saved %s (%d bodies)", path, len(bodies))
        return result

    # ── train/test comparison mode ───────────────────────────────────────

    def _execute_train_test(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pos = np.asarray(inputs["pos"], dtype=np.float32)
        pred = np.asarray(inputs["predicted_velocity_out"], dtype=np.float32)
        target = np.asarray(inputs["velocity_out"], dtype=np.float32)
        idcs_airfoil = inputs.get("idcs_airfoil")

        if not inputs.get("train_ids") and not inputs.get("test_ids"):
            return self._execute_single(inputs)

        t = min(self._timestep, pred.shape[0] - 1, target.shape[0] - 1)
        pred_t, tgt_t = pred[t], target[t]

        plane_axes, slice_axis = self._get_axes()
        ax0, ax1 = plane_axes
        slice_coord = self._get_slice_coord(pos, slice_axis)
        mask = self._get_slice_mask(pos, slice_axis, slice_coord)

        pts_2d = pos[mask][:, [ax0, ax1]]
        if idcs_airfoil is not None and len(idcs_airfoil) > 0:
            af_set = set(int(i) for i in idcs_airfoil)
            local_idcs = np.where(mask)[0]
            airfoil_local = np.array([i in af_set for i in local_idcs])
        else:
            airfoil_local = np.zeros(mask.sum(), dtype=bool)

        bodies = _detect_body_polygons(pts_2d, airfoil_local)
        grid_h, grid_v = self._build_grid(pts_2d)
        body_mask_grid = _apply_body_mask(grid_h, grid_v, bodies)

        ax_labels = {"xy": ("x", "y"), "yz": ("y", "z"), "xz": ("x", "z")}.get(
            self._slice_plane, ("h", "v")
        )

        out_dir = _ensure_output_dir()
        all_paths: list[str] = []

        for set_name, ids_key in [("train", "train_ids"), ("test", "test_ids")]:
            ids = np.asarray(inputs.get(ids_key, []))
            if len(ids) == 0:
                continue
            # Intersect with slice mask
            local_idcs = np.where(mask)[0]
            sub_mask = np.isin(local_idcs, ids)
            if sub_mask.sum() < 5:
                continue

            sub_pts = pts_2d[sub_mask]
            pred_vals = self._extract_component(pred_t[mask][sub_mask])
            tgt_vals = self._extract_component(tgt_t[mask][sub_mask])
            err_vals = np.abs(pred_vals - tgt_vals)
            r2 = 1.0 - float(np.sum((pred_vals - tgt_vals)**2)) / max(
                float(np.sum((tgt_vals - tgt_vals.mean())**2)), 1e-12
            )

            vmin = float(min(np.nanmin(tgt_vals), np.nanmin(pred_vals)))
            vmax = float(max(np.nanmax(tgt_vals), np.nanmax(pred_vals)))
            emax = float(np.nanmax(err_vals)) or 1e-6

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"{set_name.upper()} — R²={r2:.4f}  ({sub_mask.sum()} pts)", fontsize=13)

            for ax, vals, title, vm, vx, cmap in zip(
                axes,
                [tgt_vals, pred_vals, err_vals],
                ["Ground Truth", "Prediction", "Absolute Error"],
                [vmin, vmin, 0.0],
                [vmax, vmax, emax],
                [self._colormap, self._colormap, self._error_colormap],
            ):
                mp = self._render_panel(ax, sub_pts, vals, bodies, body_mask_grid,
                                        grid_h, grid_v, vm, vx, cmap, title, ax_labels)
                plt.colorbar(mp, ax=ax, shrink=0.8)

            plt.tight_layout()
            uid = uuid.uuid4().hex[:8]
            path = out_dir / f"slice_{set_name}_{uid}.png"
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            all_paths.append(str(path))
            logger.info("FieldSlicePlot[train_test]: %s R²=%.4f → %s", set_name, r2, path)

        result: dict[str, Any] = {**inputs}
        result["slice_plot_paths"] = all_paths
        result["slice_plot_meta"] = {
            "mode": "train_test_comparison",
            "timestep": t,
            "field_component": self._field_component,
            "slice_plane": self._slice_plane,
        }
        return result

    # ── video comparison mode ────────────────────────────────────────────

    def _execute_video(self, inputs: dict[str, Any]) -> dict[str, Any]:
        samples = inputs.get("samples")
        if samples is None or not isinstance(samples, list):
            logger.warning("FieldSlicePlot[video]: no samples list — falling back to single.")
            return self._execute_single(inputs)

        train_indices = np.asarray(inputs.get("train_indices", []), dtype=np.int64)
        test_indices = np.asarray(inputs.get("test_indices", []), dtype=np.int64)

        rng = np.random.RandomState(self._seed or 42)
        out_dir = _ensure_output_dir()
        all_paths: list[str] = []
        video_meta: list[dict[str, Any]] = []

        for set_name, idxs in [("train", train_indices), ("test", test_indices)]:
            if len(idxs) == 0:
                continue
            k = min(self._num_samples_per_set, len(idxs))
            chosen = rng.choice(idxs, size=k, replace=False)

            for si in chosen:
                sample = samples[int(si)]

                # Lazy load pos, velocity_out, idcs_airfoil if not present
                if "pos" not in sample and "filepath" in sample:
                    from src.backend.data_digester.temporal_point_cloud_field_digester import (
                        TemporalPointCloudFieldDigester,
                    )
                    max_points = int(inputs.get("max_points", 0))
                    arrays = TemporalPointCloudFieldDigester.load_sample(
                        sample["filepath"],
                        keys={"pos", "velocity_out", "idcs_airfoil"},
                        max_points=max_points,
                    )
                    sample.update(arrays)
                    del arrays

                # Get prediction (generate if not stored)
                if "predicted_velocity_out" not in sample:
                    model = inputs.get("model_artifact")
                    if model is not None and hasattr(model, "_predict_sample"):
                        pred = model._predict_sample(
                            sample,
                            fe_pipeline=inputs.get("fe_pipeline"),
                            graph_cache=inputs.get("graph_cache"),
                            max_points=int(inputs.get("max_points", 0)),
                        )
                        sample["predicted_velocity_out"] = pred
                    else:
                        logger.warning(
                            "FieldSlicePlot[video]: sample %d has no prediction, skipping.", si
                        )
                        continue

                pos = np.asarray(sample["pos"], dtype=np.float32)
                pred_vel = np.asarray(sample["predicted_velocity_out"], dtype=np.float32)
                tgt_vel = np.asarray(sample["velocity_out"], dtype=np.float32)
                idcs_airfoil = sample.get("idcs_airfoil")
                sim_id = sample.get("sim_id", f"sample_{si}")
                window_id = sample.get("window_id", "?")

                uid = uuid.uuid4().hex[:8]
                gif_path = self._render_temporal_gif(
                    pos=pos,
                    pred_vel=pred_vel,
                    tgt_vel=tgt_vel,
                    idcs_airfoil=idcs_airfoil,
                    title_prefix=f"{set_name.upper()} sim={sim_id} win={window_id}",
                    out_dir=out_dir,
                    uid=uid,
                )
                if gif_path:
                    all_paths.append(gif_path)
                    video_meta.append({
                        "set": set_name,
                        "sample_index": int(si),
                        "sim_id": sim_id,
                        "window_id": window_id,
                        "path": gif_path,
                    })

        result: dict[str, Any] = {**inputs}
        result["video_paths"] = all_paths
        result["video_meta"] = video_meta
        result["slice_plot_paths"] = all_paths
        result["slice_plot_meta"] = {
            "mode": "video_comparison",
            "field_component": self._field_component,
            "slice_plane": self._slice_plane,
            "num_videos": len(all_paths),
        }
        logger.info("FieldSlicePlot[video]: generated %d GIFs", len(all_paths))
        return result

    def _render_temporal_gif(
        self,
        pos: np.ndarray,
        pred_vel: np.ndarray,
        tgt_vel: np.ndarray,
        idcs_airfoil: np.ndarray | None,
        title_prefix: str,
        out_dir: Path,
        uid: str,
    ) -> str | None:
        """Render animated GIF with mesh-faithful frames for all T_out steps."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        T_out = pred_vel.shape[0]
        plane_axes, slice_axis = self._get_axes()
        ax0, ax1 = plane_axes

        slice_coord = self._get_slice_coord(pos, slice_axis)
        mask = self._get_slice_mask(pos, slice_axis, slice_coord)
        if mask.sum() < 10:
            logger.warning("FieldSlicePlot[video]: too few pts for %s", title_prefix)
            return None

        pts_2d = pos[mask][:, [ax0, ax1]]

        # Detect bodies and build body mask once (geometry is static)
        if idcs_airfoil is not None and len(idcs_airfoil) > 0:
            af_set = set(int(i) for i in idcs_airfoil)
            local_idcs = np.where(mask)[0]
            airfoil_local = np.array([i in af_set for i in local_idcs])
        else:
            airfoil_local = np.zeros(mask.sum(), dtype=bool)

        bodies = _detect_body_polygons(pts_2d, airfoil_local)
        grid_h, grid_v = self._build_grid(pts_2d)
        body_mask_grid = _apply_body_mask(grid_h, grid_v, bodies)

        # Global colour range over all timesteps
        all_gt = np.concatenate([self._extract_component(tgt_vel[t][mask]) for t in range(T_out)])
        all_pr = np.concatenate([self._extract_component(pred_vel[t][mask]) for t in range(T_out)])
        all_err = np.abs(all_gt - all_pr)
        vmin = float(np.nanmin([np.nanmin(all_gt), np.nanmin(all_pr)]))
        vmax = float(np.nanmax([np.nanmax(all_gt), np.nanmax(all_pr)]))
        emax = float(np.nanmax(all_err)) or 1e-6

        ax_labels = {"xy": ("x", "y"), "yz": ("y", "z"), "xz": ("x", "z")}.get(
            self._slice_plane, ("h", "v")
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colorbars_added = [False]

        def _render_frame(frame: int) -> list:
            for ax in axes:
                ax.cla()
            gt_vals = self._extract_component(tgt_vel[frame][mask])
            pr_vals = self._extract_component(pred_vel[frame][mask])
            err_vals = np.abs(gt_vals - pr_vals)

            for ax, vals, title, vm, vx, cmap in zip(
                axes,
                [gt_vals, pr_vals, err_vals],
                ["Ground Truth", "Prediction", "Absolute Error"],
                [vmin, vmin, 0.0],
                [vmax, vmax, emax],
                [self._colormap, self._colormap, self._error_colormap],
            ):
                mp = self._render_panel(ax, pts_2d, vals, bodies, body_mask_grid,
                                        grid_h, grid_v, vm, vx, cmap, title, ax_labels)
                if not colorbars_added[0]:
                    plt.colorbar(mp, ax=ax, shrink=0.8)

            colorbars_added[0] = True

            # Phase indicator
            phase = "INPUT" if frame == 0 else "FORECAST"
            phase_color = "steelblue" if frame == 0 else "firebrick"
            fig.suptitle(
                f"{title_prefix}  |  t={frame+1}/{T_out}  [{phase}]",
                fontsize=12, color=phase_color,
            )
            plt.tight_layout()
            return []

        # Render first frame to initialise colorbars
        _render_frame(0)

        anim = FuncAnimation(fig, _render_frame, frames=T_out, interval=500, blit=False)
        gif_path = out_dir / f"video_{uid}.gif"
        anim.save(str(gif_path), writer=PillowWriter(fps=2))
        plt.close(fig)

        logger.info("FieldSlicePlot[video]: saved %s (%d frames, %d bodies)",
                    gif_path, T_out, len(bodies))
        return str(gif_path)
