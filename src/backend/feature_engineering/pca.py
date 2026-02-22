"""
PCA (Principal Component Analysis)
===================================
Dimensionality reduction via PCA for surrogate modelling pipelines.

Learns a compression on the feature columns of the upstream scalar data
and outputs PCA-transformed features for downstream blocks.

Hyperparameters:
    n_components   – number of principal components to keep
    whiten         – whether to whiten the output (unit variance per component)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PCATransformer:
    """
    PCA for feature engineering in surrogate modelling pipelines.

    Wraps ``sklearn.decomposition.PCA`` and exposes the standard pipeline
    ``execute`` interface.  The node learns a compression on the selected
    feature columns and outputs the PCA-weighted data downstream.

    Expected ``hyperparams``:
        - n_components : int   – number of components to retain
        - whiten       : bool  – normalise components to unit variance
    """

    def __init__(self, hyperparams: dict[str, Any] | None = None, seed: int | None = None) -> None:
        hp = hyperparams or {}
        self._n_components: int = int(hp.get("n_components", 2))
        self._whiten: bool = bool(hp.get("whiten", False))
        self._seed: int | None = seed

        self._pca: Any = None          # fitted sklearn PCA object
        self._fitted: bool = False

    # ── core API ─────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "PCATransformer":
        """Fit PCA on *X* (n_samples, n_features)."""
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise RuntimeError(
                "PCA requires scikit-learn.  Install with `pip install scikit-learn`."
            ) from exc

        X = np.asarray(X, dtype=np.float32)

        # Clamp n_components to available features
        n_comp = min(self._n_components, X.shape[1], X.shape[0])

        self._pca = PCA(n_components=n_comp, whiten=self._whiten,
                        random_state=self._seed)
        self._pca.fit(X)
        self._fitted = True

        logger.info(
            "PCA: fitted  n_components=%d  explained_variance=%.4f  input_shape=%s",
            n_comp,
            float(self._pca.explained_variance_ratio_.sum()),
            X.shape,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted PCA projection to *X*."""
        if not self._fitted:
            raise RuntimeError("PCA not fitted – call fit() first.")
        X = np.asarray(X, dtype=np.float32)
        return self._pca.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """Reconstruct from PCA space back to original feature space."""
        if not self._fitted:
            raise RuntimeError("PCA not fitted.")
        return self._pca.inverse_transform(np.asarray(X_pca, dtype=np.float32)).astype(
            np.float32
        )

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Pipeline node entry point.

        Receives ``X`` (and optionally ``y``) from upstream, fits PCA on X,
        transforms X, and passes everything downstream.  The PCA weights
        (components), explained variance, and the fitted transformer itself
        are included in the output so downstream nodes or validators can
        inspect them.
        """
        X: np.ndarray = inputs["X"]
        y = inputs.get("y")

        X_pca = self.fit_transform(X)

        # Build downstream output
        outputs: dict[str, Any] = {
            **inputs,
            "X": X_pca,
            "pca": self,
            "pca_components": self._pca.components_.tolist(),
            "pca_explained_variance": self._pca.explained_variance_ratio_.tolist(),
            "pca_n_components": int(self._pca.n_components_),
        }

        # Generate PCA feature names for downstream
        outputs["feature_names"] = [
            f"PC{i + 1}" for i in range(X_pca.shape[1])
        ]

        logger.info(
            "PCA: transformed X %s → %s  explained_var=%.4f",
            X.shape,
            X_pca.shape,
            float(self._pca.explained_variance_ratio_.sum()),
        )
        return outputs
