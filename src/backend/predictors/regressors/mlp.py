"""
MLP (Multi-Layer Perceptron) Regressor
======================================
Full PyTorch implementation for surrogate regression.

Basic hyperparameters:
    hidden_layers, neurons_per_layer, activation, learning_rate,
    epochs, batch_size

Advanced hyperparameters:
    optimizer, loss_function, weight_init, dropout, batch_norm,
    lr_scheduler, lr_scheduler_step_size, lr_scheduler_gamma,
    lr_scheduler_patience, gradient_clipping, early_stopping,
    early_stopping_patience, weight_decay, layer_sizes
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

if TYPE_CHECKING:
    pass  # torch types used via string annotations at runtime

from src.backend.predictors.model_base import ModelBase

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
        "Mish": nn.Mish,
        "Softplus": nn.Softplus,
        "PReLU": nn.PReLU,
    }

# ── Weight initialisation helpers ────────────────────────────────────────────

_INIT_FNS: dict[str, Any] = {}
if nn is not None:
    _INIT_FNS = {
        "xavier_uniform": nn.init.xavier_uniform_,
        "xavier_normal": nn.init.xavier_normal_,
        "kaiming_uniform": nn.init.kaiming_uniform_,
        "kaiming_normal": nn.init.kaiming_normal_,
        "orthogonal": nn.init.orthogonal_,
        "zeros": nn.init.zeros_,
        "ones": nn.init.ones_,
        "default": None,  # use PyTorch defaults
    }


def _apply_weight_init(module: Any, init_name: str) -> None:
    """Recursively apply *init_name* initialisation to Linear layers."""
    fn = _INIT_FNS.get(init_name)
    if fn is None:
        return  # 'default' or unknown → keep PyTorch defaults
    for m in module.modules():
        if isinstance(m, nn.Linear):
            fn(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ═════════════════════════════════════════════════════════════════════════════
# Inner nn.Module
# ═════════════════════════════════════════════════════════════════════════════

class _MLPNetwork(nn.Module):
    """
    Configurable feed-forward network.

    Supports:
    - Arbitrary layer widths (via *layer_sizes* list)
    - Dropout between hidden layers
    - Optional BatchNorm before activation
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_sizes: list[int],
        activation: str = "ReLU",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        act_cls = _ACTIVATIONS.get(activation, nn.ReLU)
        layers: list[Any] = []
        prev = input_dim

        for width in layer_sizes:
            layers.append(nn.Linear(prev, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev = width

        # Output head (no activation / norm / dropout)
        layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


# ═════════════════════════════════════════════════════════════════════════════
# MLPRegressor  (public)
# ═════════════════════════════════════════════════════════════════════════════

class MLPRegressor(ModelBase):
    """
    PyTorch Multi-Layer Perceptron regressor.

    All configuration is driven by hyperparameters that the frontend can set.
    Training does **not** perform any hyperparameter tuning itself — that is
    the responsibility of the ``hp_tuner`` node category.
    """

    model_name = "MLP"
    model_category = "regressor"
    model_type = "MLP"
    is_differentiable = True

    # ── Hyperparameter schema ────────────────────────────────────────────────

    @classmethod
    def default_hyperparams(cls) -> dict[str, Any]:
        return {
            # ─ basic ─────────────────────────────────────────────────────────
            "hidden_layers": 3,
            "neurons_per_layer": 64,
            "activation": "ReLU",
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32,
            # ─ advanced ──────────────────────────────────────────────────────
            "optimizer": "Adam",
            "loss_function": "MSE",
            "weight_init": "default",
            "dropout": 0.0,
            "batch_norm": False,
            "lr_scheduler": "none",
            "lr_scheduler_step_size": 10,
            "lr_scheduler_gamma": 0.1,
            "lr_scheduler_patience": 5,
            "gradient_clipping": 0.0,
            "early_stopping": False,
            "early_stopping_patience": 10,
            "weight_decay": 0.0,
            "layer_sizes": "",
        }

    # ── Cached internal state ────────────────────────────────────────────────

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        super().__init__(hyperparams)
        self._optimizer: Any = None
        self._scheduler: Any = None
        self._loss_fn: Any = None

    # ── Lifecycle: build ─────────────────────────────────────────────────────

    def build(self) -> None:
        """
        Construct the ``_MLPNetwork`` from current hyperparams.

        If ``layer_sizes`` is non-empty (comma-separated ints), it overrides
        ``hidden_layers`` / ``neurons_per_layer``.

        If ``output_dim`` hyperparam is set to a positive value, it overrides
        the automatically detected output dimension (used for transform mode).
        """
        if torch is None:
            raise RuntimeError(
                "MLP requires PyTorch.  Install with `pip install torch`."
            )

        layer_str: str = str(self.get_hyperparam("layer_sizes", "")).strip()
        if layer_str:
            layer_sizes = [int(s.strip()) for s in layer_str.split(",") if s.strip()]
        else:
            n_layers = int(self.get_hyperparam("hidden_layers", 3))
            width = int(self.get_hyperparam("neurons_per_layer", 64))
            layer_sizes = [width] * n_layers

        # Allow explicit output_dim override (for transform role)
        hp_output_dim = int(self.get_hyperparam("output_dim", 0))
        if hp_output_dim > 0:
            self._output_dim = hp_output_dim

        self._model = _MLPNetwork(
            input_dim=self._input_dim,
            output_dim=self._output_dim,
            layer_sizes=layer_sizes,
            activation=str(self.get_hyperparam("activation", "ReLU")),
            dropout=float(self.get_hyperparam("dropout", 0.0)),
            batch_norm=bool(self.get_hyperparam("batch_norm", False)),
        )

        # Weight initialisation
        init_name = str(self.get_hyperparam("weight_init", "default"))
        _apply_weight_init(self._model, init_name)

        logger.info(
            "%s: built network  layers=%s  params=%s",
            self.model_name,
            layer_sizes,
            sum(p.numel() for p in self._model.parameters()),
        )

    # ── Differentiable interface ───────────────────────────────────────────────

    def get_torch_module(self) -> "nn.Module":
        """
        Return the underlying ``_MLPNetwork`` nn.Module for use in
        composed differentiable training pipelines.
        """
        if self._model is None:
            raise RuntimeError("Call build() before get_torch_module().")
        return self._model

    # ── Lifecycle: compile ───────────────────────────────────────────────────

    def compile(self) -> None:
        """Set up optimiser, loss function and learning-rate scheduler."""
        if self._model is None:
            raise RuntimeError("Call build() before compile().")

        lr = float(self.get_hyperparam("learning_rate", 1e-3))
        wd = float(self.get_hyperparam("weight_decay", 0.0))

        # ── Loss ──────────────────────────────────────────────────────────
        loss_name = str(self.get_hyperparam("loss_function", "MSE"))
        loss_map = {
            "MSE": nn.MSELoss,
            "MAE": nn.L1Loss,
            "Huber": nn.HuberLoss,
            "SmoothL1": nn.SmoothL1Loss,
        }
        self._loss_fn = loss_map.get(loss_name, nn.MSELoss)()

        # ── Optimiser ─────────────────────────────────────────────────────
        opt_name = str(self.get_hyperparam("optimizer", "Adam"))
        opt_map: dict[str, type] = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
            "RMSprop": torch.optim.RMSprop,
            "LBFGS": torch.optim.LBFGS,
        }
        opt_cls = opt_map.get(opt_name, torch.optim.Adam)
        opt_kwargs: dict[str, Any] = {"lr": lr, "weight_decay": wd}
        if opt_name == "SGD":
            opt_kwargs["momentum"] = 0.9
        self._optimizer = opt_cls(self._model.parameters(), **opt_kwargs)

        # ── LR Scheduler ─────────────────────────────────────────────────
        sched_name = str(self.get_hyperparam("lr_scheduler", "none"))
        step_size = int(self.get_hyperparam("lr_scheduler_step_size", 10))
        gamma = float(self.get_hyperparam("lr_scheduler_gamma", 0.1))
        patience = int(self.get_hyperparam("lr_scheduler_patience", 5))

        sched_map: dict[str, Any] = {
            "step": lambda: torch.optim.lr_scheduler.StepLR(
                self._optimizer, step_size=step_size, gamma=gamma
            ),
            "cosine": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer,
                T_max=int(self.get_hyperparam("epochs", 100)),
            ),
            "plateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer, patience=patience, factor=gamma
            ),
            "exponential": lambda: torch.optim.lr_scheduler.ExponentialLR(
                self._optimizer, gamma=gamma
            ),
        }
        factory = sched_map.get(sched_name)
        self._scheduler = factory() if factory else None

        logger.info(
            "%s: compiled  opt=%s  lr=%.2e  loss=%s  sched=%s",
            self.model_name,
            opt_name,
            lr,
            loss_name,
            sched_name,
        )

    # ── Lifecycle: train ─────────────────────────────────────────────────────

    def train(self, X: Any, y: Any, **kwargs: Any) -> None:
        """
        Train the MLP for the configured number of epochs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        """
        if self._model is None:
            raise RuntimeError("Call build() and compile() before train().")

        X_t, y_t = self._to_tensors(X, y)
        self._input_dim = X_t.shape[1]
        self._output_dim = y_t.shape[1] if y_t.ndim > 1 else 1

        # Rebuild if dims changed since last build() (first call scenario).
        if (
            self._model.net[0].in_features != self._input_dim
            or self._model.net[-1].out_features != self._output_dim
        ):
            self.build()
            self.compile()

        device = self._device()
        self._model.to(device)
        X_t, y_t = X_t.to(device), y_t.to(device)

        epochs = int(self.get_hyperparam("epochs", 100))
        batch_size = int(self.get_hyperparam("batch_size", 32))
        clip_val = float(self.get_hyperparam("gradient_clipping", 0.0))
        early_stop = bool(self.get_hyperparam("early_stopping", False))
        es_patience = int(self.get_hyperparam("early_stopping_patience", 10))

        dataset = TensorDataset(X_t, y_t)

        # Seed the DataLoader shuffle for reproducibility
        generator = None
        if self._seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self._seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=generator)

        best_loss = float("inf")
        patience_counter = 0

        self._model.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0

            for X_batch, y_batch in loader:
                self._optimizer.zero_grad()
                pred = self._model(X_batch)

                if y_batch.ndim == 1:
                    pred = pred.squeeze(-1)

                loss = self._loss_fn(pred, y_batch)
                loss.backward()

                if clip_val > 0.0:
                    nn.utils.clip_grad_norm_(self._model.parameters(), clip_val)

                self._optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= len(dataset)

            # LR scheduler step
            if self._scheduler is not None:
                if isinstance(
                    self._scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self._scheduler.step(epoch_loss)
                else:
                    self._scheduler.step()

            # Early stopping
            if early_stop:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                    best_state = copy.deepcopy(self._model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= es_patience:
                        logger.info(
                            "%s: early stopping at epoch %d/%d (loss=%.6f)",
                            self.model_name,
                            epoch,
                            epochs,
                            epoch_loss,
                        )
                        self._model.load_state_dict(best_state)
                        break

            if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                logger.debug(
                    "%s: epoch %d/%d  loss=%.6f",
                    self.model_name,
                    epoch,
                    epochs,
                    epoch_loss,
                )

        self._is_trained = True

    # ── Lifecycle: predict ───────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, X: Any) -> np.ndarray:
        """
        Return predictions as a numpy array.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
        """
        self._require_trained("predict")
        device = self._device()
        self._model.to(device)
        self._model.eval()

        X_t = self._to_tensor(X).to(device)
        preds = self._model(X_t).cpu().numpy()

        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.ravel()
        return preds

    # ── Lifecycle: score ─────────────────────────────────────────────────────

    def score(self, X: Any, y: Any) -> dict[str, float]:
        """
        Compute regression metrics: R², RMSE, MAE.

        Returns
        -------
        dict[str, float]
        """
        self._require_trained("score")
        preds = self.predict(X)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        preds = preds.ravel()

        ss_res = float(np.sum((y_arr - preds) ** 2))
        ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rmse = float(np.sqrt(np.mean((y_arr - preds) ** 2)))
        mae = float(np.mean(np.abs(y_arr - preds)))

        return {"r2": round(r2, 6), "rmse": round(rmse, 6), "mae": round(mae, 6)}

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save trained MLP to a directory (weights + hyperparams + metadata)."""
        self._require_trained("save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self._model.state_dict(), path / "weights.pt")

        meta = {
            "class": self.__class__.__qualname__,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "hyperparams": self._hyperparams,
            "metadata": self._metadata,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        logger.info("%s: saved → %s", self.model_name, path)

    @classmethod
    def load(cls, path: str | Path) -> "MLPRegressor":
        """Load a previously saved MLP from *path*."""
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())

        instance = cls(hyperparams=meta.get("hyperparams"))
        instance._metadata = meta.get("metadata", {})
        instance._input_dim = meta.get("input_dim", 0)
        instance._output_dim = meta.get("output_dim", 0)

        instance.build()

        state_dict = torch.load(path / "weights.pt", map_location="cpu")
        instance._model.load_state_dict(state_dict)
        instance._is_trained = True

        logger.info("%s: loaded ← %s", cls.model_name, path)
        return instance

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _device() -> Any:
        """Select CUDA if available, else CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _to_tensor(arr: Any) -> Any:
        """Convert array-like to a float32 tensor."""
        if isinstance(arr, torch.Tensor):
            return arr.float()
        return torch.as_tensor(np.asarray(arr, dtype=np.float32))

    @staticmethod
    def _to_tensors(X: Any, y: Any) -> tuple[Any, Any]:
        """Convert X and y to float32 tensors, ensuring ≥ 2-D for X."""
        X_t = MLPRegressor._to_tensor(X)
        y_t = MLPRegressor._to_tensor(y)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(-1)
        return X_t, y_t
