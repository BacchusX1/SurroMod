"""
Flow Forecast Loss Utilities
=============================
Loss functions and metric helpers for velocity-field prediction tasks.
"""

from __future__ import annotations

import numpy as np


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, target)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    """Relative L2 error: ||pred - target||_2 / ||target||_2."""
    denom = np.linalg.norm(target.ravel())
    if denom < 1e-12:
        return 0.0
    return float(np.linalg.norm((pred - target).ravel()) / denom)


def max_abs_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.abs(pred - target)))


def per_timestep_metrics(
    pred: np.ndarray, target: np.ndarray,
) -> dict[str, list[float]]:
    """
    Compute per-timestep MSE, MAE, RMSE for arrays of shape (T, N, 3).
    """
    T = pred.shape[0]
    mse_list, mae_list, rmse_list = [], [], []
    for t in range(T):
        p, g = pred[t], target[t]
        m = float(np.mean((p - g) ** 2))
        mse_list.append(m)
        rmse_list.append(float(np.sqrt(m)))
        mae_list.append(float(np.mean(np.abs(p - g))))
    return {"mse": mse_list, "mae": mae_list, "rmse": rmse_list}


def per_component_metrics(
    pred: np.ndarray, target: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Compute per-velocity-component (vx, vy, vz) metrics.
    Arrays should have last dim == 3.
    """
    labels = ["vx", "vy", "vz"]
    result: dict[str, dict[str, float]] = {}
    for i, lbl in enumerate(labels):
        p = pred[..., i]
        g = target[..., i]
        result[lbl] = {
            "mse": float(np.mean((p - g) ** 2)),
            "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
            "mae": float(np.mean(np.abs(p - g))),
        }
    return result


def r_squared(pred: np.ndarray, target: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = float(np.sum((pred - target) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def compute_all_metrics(
    pred: np.ndarray, target: np.ndarray,
) -> dict[str, object]:
    """Compute a complete metrics dictionary for velocity field prediction."""
    return {
        "mse": mse(pred, target),
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
        "r2": r_squared(pred, target),
        "relative_l2": relative_l2(pred, target),
        "max_abs_error": max_abs_error(pred, target),
        "per_timestep": per_timestep_metrics(pred, target),
        "per_component": per_component_metrics(pred, target),
    }
