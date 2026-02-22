"""
Optimiser-Based HP Tuner
========================
Black-box optimisation over the hyperparameter space (TPE, CMA-ES, Bayesian, …).

Designed to integrate with Optuna, but keeps the interface general enough to
plug in any sampler/pruner backend.

Hyperparameters (frontend-settable):
    n_trials         – number of optimisation trials
    algorithm        – sampler algorithm (tpe, cma_es, random, bayesian)
    scoring_metric   – metric to optimise
    timeout          – wall-clock budget in seconds
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class OptimiserBasedTuner:
    """Optimiser-based hyperparameter tuner.

    Parameters
    ----------
    hyperparams : dict
        Configuration from the frontend.
    """

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        self._hyperparams: dict[str, Any] = {
            "n_trials": 100,
            "algorithm": "tpe",
            "scoring_metric": "r2",
            "timeout": 600,
            **(hyperparams or {}),
        }
        self._results: dict[str, Any] = {}

    def run(
        self,
        model: Any,
        search_space: dict[str, Any],
        X: Any,
        y: Any,
    ) -> dict[str, Any]:
        """
        Execute optimiser-based hyperparameter search.

        Parameters
        ----------
        model : ModelBase
            The surrogate model to tune.
        search_space : dict
            Defines ranges / distributions for each hyperparameter.
        X, y : array-like
            Training data.

        Returns
        -------
        dict
            Best hyperparameters, best score, and trial history.
        """
        # TODO: implement optimiser-based HP search logic
        raise NotImplementedError("OptimiserBasedTuner.run() is not yet implemented.")
