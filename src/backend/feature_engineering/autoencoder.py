"""
Autoencoder
===========
Learns a compressed latent-space representation of the input data.
The encoder maps high-dimensional input to a low-dimensional latent vector;
the decoder reconstructs the original.  Only the latent variables are
passed downstream (e.g. into an MLP regressor).

Hyperparameters:
    latent_dim, hidden_layers, neurons_per_layer, activation,
    learning_rate, epochs, batch_size
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ── Activation look-up ───────────────────────────────────────────────────────

_ACTIVATIONS: dict[str, type] = {}
if nn is not None:
    _ACTIVATIONS = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid,
        "GELU": nn.GELU,
        "LeakyReLU": nn.LeakyReLU,
        "ELU": nn.ELU,
        "SiLU": nn.SiLU,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Inner nn.Module
# ═════════════════════════════════════════════════════════════════════════════

class _AutoencoderNetwork(nn.Module):
    """
    Symmetric encoder–decoder network.

    The encoder compresses ``input_dim`` → ``latent_dim`` through
    progressively narrower hidden layers.  The decoder mirrors the
    encoder in reverse.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_layers: int = 2,
        neurons_per_layer: int = 64,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()

        act_cls = _ACTIVATIONS.get(activation, nn.ReLU)

        # ── Encoder ──────────────────────────────────────────────────────
        enc_layers: list[Any] = []
        prev = input_dim
        # Build progressively smaller layers towards latent_dim
        layer_sizes = self._make_layer_sizes(
            input_dim, latent_dim, hidden_layers, neurons_per_layer
        )
        for width in layer_sizes:
            enc_layers.append(nn.Linear(prev, width))
            enc_layers.append(act_cls())
            prev = width
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder (mirror) ─────────────────────────────────────────────
        dec_layers: list[Any] = []
        prev = latent_dim
        for width in reversed(layer_sizes):
            dec_layers.append(nn.Linear(prev, width))
            dec_layers.append(act_cls())
            prev = width
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    @staticmethod
    def _make_layer_sizes(
        input_dim: int,
        latent_dim: int,
        n_hidden: int,
        neurons: int,
    ) -> list[int]:
        """
        Generate hidden-layer widths that taper from *neurons* down
        towards *latent_dim*.
        """
        if n_hidden == 0:
            return []
        if n_hidden == 1:
            return [neurons]
        # Linearly interpolate widths
        sizes = []
        for i in range(n_hidden):
            frac = i / max(n_hidden - 1, 1)
            w = int(neurons * (1 - frac) + latent_dim * frac)
            w = max(w, latent_dim)
            sizes.append(w)
        return sizes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, latent)."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the latent representation."""
        return self.encoder(x)


# ═════════════════════════════════════════════════════════════════════════════
# Autoencoder  (public pipeline node)
# ═════════════════════════════════════════════════════════════════════════════

class Autoencoder:
    """
    Autoencoder for latent-space feature extraction in surrogate pipelines.

    Trains an encoder–decoder network to reconstruct the input features,
    then passes only the latent-space variables downstream.

    Expected ``hyperparams``:
        - latent_dim         : int   – dimensionality of the bottleneck
        - hidden_layers      : int   – number of hidden layers per side
        - neurons_per_layer  : int   – width of the first hidden layer
        - activation         : str   – activation function name
        - learning_rate      : float – optimiser learning rate
        - epochs             : int   – training epochs
        - batch_size         : int   – mini-batch size
    """

    def __init__(self, hyperparams: dict[str, Any] | None = None, seed: int | None = None) -> None:
        hp = hyperparams or {}
        self._latent_dim: int = int(hp.get("latent_dim", 16))
        self._hidden_layers: int = int(hp.get("hidden_layers", 2))
        self._neurons_per_layer: int = int(hp.get("neurons_per_layer", 64))
        self._activation: str = str(hp.get("activation", "ReLU"))
        self._learning_rate: float = float(hp.get("learning_rate", 1e-3))
        self._epochs: int = int(hp.get("epochs", 100))
        self._batch_size: int = int(hp.get("batch_size", 32))
        self._seed: int | None = seed

        self._model: _AutoencoderNetwork | None = None
        self._fitted: bool = False

    # ── core API ─────────────────────────────────────────────────────────

    def _build(self, input_dim: int) -> None:
        """Construct the autoencoder network."""
        if torch is None:
            raise RuntimeError(
                "Autoencoder requires PyTorch.  Install with `pip install torch`."
            )
        self._model = _AutoencoderNetwork(
            input_dim=input_dim,
            latent_dim=self._latent_dim,
            hidden_layers=self._hidden_layers,
            neurons_per_layer=self._neurons_per_layer,
            activation=self._activation,
        )

    @staticmethod
    def _device() -> Any:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def fit(self, X: np.ndarray) -> "Autoencoder":
        """Train the autoencoder to reconstruct *X*."""
        X = np.asarray(X, dtype=np.float32)
        input_dim = X.shape[1]

        self._build(input_dim)
        device = self._device()
        self._model.to(device)

        X_t = torch.as_tensor(X).to(device)
        dataset = TensorDataset(X_t)

        # Seed the DataLoader shuffle for reproducibility
        generator = None
        if self._seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self._seed)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True,
                            generator=generator)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._learning_rate
        )
        loss_fn = nn.MSELoss()

        self._model.train()
        for epoch in range(1, self._epochs + 1):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                x_hat, _ = self._model(batch)
                loss = loss_fn(x_hat, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)

            epoch_loss /= len(dataset)

            if epoch % max(1, self._epochs // 10) == 0 or epoch == self._epochs:
                logger.debug(
                    "Autoencoder: epoch %d/%d  recon_loss=%.6f",
                    epoch, self._epochs, epoch_loss,
                )

        self._fitted = True
        logger.info(
            "Autoencoder: training complete  input_dim=%d  latent_dim=%d  final_loss=%.6f",
            input_dim, self._latent_dim, epoch_loss,
        )
        return self

    @torch.no_grad()
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode *X* into the latent space."""
        if not self._fitted:
            raise RuntimeError("Autoencoder not fitted – call fit() first.")
        device = self._device()
        self._model.to(device)
        self._model.eval()

        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32)).to(device)
        z = self._model.encode(X_t)
        return z.cpu().numpy()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform."""
        return self.fit(X).transform(X)

    @torch.no_grad()
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Full encode→decode round-trip (for diagnostics)."""
        if not self._fitted:
            raise RuntimeError("Autoencoder not fitted.")
        device = self._device()
        self._model.to(device)
        self._model.eval()

        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32)).to(device)
        x_hat, _ = self._model(X_t)
        return x_hat.cpu().numpy()

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Pipeline node entry point.

        Receives ``X`` (and optionally ``y``) from upstream, trains the
        autoencoder on X, and replaces X with the latent representation
        for downstream consumption.
        """
        X: np.ndarray = inputs["X"]
        y = inputs.get("y")

        X_latent = self.fit_transform(X)

        outputs: dict[str, Any] = {
            **inputs,
            "X": X_latent,
            "autoencoder": self,
            "latent_dim": self._latent_dim,
        }

        # Encode holdout with the same fitted autoencoder (no re-fitting)
        if "X_holdout" in inputs:
            X_ho = np.asarray(inputs["X_holdout"], dtype=np.float32)
            outputs["X_holdout"] = self.transform(X_ho)

        # Generate latent feature names
        outputs["feature_names"] = [
            f"z{i + 1}" for i in range(X_latent.shape[1])
        ]

        logger.info(
            "Autoencoder: compressed X %s → %s (latent_dim=%d)",
            X.shape,
            X_latent.shape,
            self._latent_dim,
        )
        return outputs
