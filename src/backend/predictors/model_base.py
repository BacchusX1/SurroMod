"""
ModelBase
=========
Abstract base class for all surrogate model predictors (regressors and classifiers).

Every concrete predictor (MLP, LSTM, CNN, KRR, RandomForest, SVM, …) must subclass
this and implement the abstract interface below.  The base class owns the full
lifecycle:

    build → compile → train → validate → save/load → predict

as well as housekeeping helpers (hyperparameter management, logging, serialisation).
"""

from __future__ import annotations

import abc
import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ModelBase(abc.ABC):
    """
    Abstract base for all SurroMod predictors.

    Subclasses must implement the abstract methods marked with @abc.abstractmethod.
    All other methods provide shared default behaviour or extension hooks.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    #: Human-readable name shown in the UI (override in subclass).
    model_name: str = "ModelBase"

    #: One of "regressor" or "classifier" (override in subclass).
    model_category: str = "undefined"

    #: Frontend model discriminator — must match the TypeScript union value
    #: exactly (e.g. ``'MLP'``, ``'RandomForest'``, ``'KRR'``, …).
    #: Override in every concrete subclass.
    model_type: str = "undefined"

    #: Whether the model supports PyTorch autograd (end-to-end backprop).
    #: Override to ``True`` in differentiable subclasses (MLP, CNN, LSTM, …).
    is_differentiable: bool = False

    # ── Subclass registry ─────────────────────────────────────────────────────
    #: Maps ``model_type`` strings to concrete subclasses so the pipeline
    #: executor can instantiate models from frontend node data.
    _registry: dict[str, type["ModelBase"]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register every concrete subclass by its *model_type*."""
        super().__init_subclass__(**kwargs)
        if cls.model_type != "undefined":
            ModelBase._registry[cls.model_type] = cls

    @classmethod
    def create(cls, model_type: str, hyperparams: dict[str, Any] | None = None) -> "ModelBase":
        """
        Factory: instantiate a model by its frontend *model_type* string.

        Parameters
        ----------
        model_type:
            One of the values from the TypeScript ``RegressorModel`` or
            ``ClassifierModel`` unions (e.g. ``'MLP'``, ``'SVM'``).
        hyperparams:
            Optional hyperparameter overrides.

        Raises
        ------
        KeyError
            If *model_type* has no registered subclass.
        """
        if model_type not in cls._registry:
            registered = ", ".join(sorted(cls._registry)) or "(none)"
            raise KeyError(
                f"Unknown model_type '{model_type}'. "
                f"Registered types: {registered}"
            )
        return cls._registry[model_type](hyperparams=hyperparams)

    @classmethod
    def registered_types(cls) -> list[str]:
        """Return all registered *model_type* strings."""
        return sorted(cls._registry)

    # ─────────────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        """
        Initialise the model with an optional hyperparameter dict.

        Parameters
        ----------
        hyperparams:
            Key/value pairs that control architecture and training.  Missing keys
            fall back to :meth:`default_hyperparams`.
        """
        self._hyperparams: dict[str, Any] = {
            **self.default_hyperparams(),
            **(hyperparams or {}),
        }
        self._model: Any = None          # the underlying PyTorch / sklearn object
        self._is_trained: bool = False
        self._metadata: dict[str, Any] = {}
        self._seed: int | None = None    # global reproducibility seed
        self._role: str = "final"        # 'transform' or 'final'
        self._input_dim: int = 0
        self._output_dim: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Hyperparameter interface
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def default_hyperparams(cls) -> dict[str, Any]:
        """
        Return the default hyperparameter dict for this model type.

        Override in every subclass to expose the model's knobs to the UI.
        """
        return {}

    def get_hyperparam(self, key: str, fallback: Any = None) -> Any:
        """Return a single hyperparameter value, or *fallback* if not set."""
        return self._hyperparams.get(key, fallback)

    def set_hyperparam(self, key: str, value: Any) -> None:
        """Update a single hyperparameter.  Rebuilds the model if already built."""
        self._hyperparams[key] = value
        if self._model is not None:
            logger.info(
                "%s: hyperparam '%s' changed — rebuilding model.", self.model_name, key
            )
            self._is_trained = False
            self.build()

    def set_hyperparams(self, hyperparams: dict[str, Any]) -> None:
        """Bulk-update hyperparameters from a dict.  Rebuilds the model if already built."""
        self._hyperparams.update(hyperparams)
        if self._model is not None:
            logger.info("%s: hyperparams updated — rebuilding model.", self.model_name)
            self._is_trained = False
            self.build()

    def validate_hyperparams(self) -> list[str]:
        """
        Validate current hyperparameters.

        Returns
        -------
        list[str]
            A (possibly empty) list of human-readable error/warning strings.
        """
        errors: list[str] = []
        defaults = self.default_hyperparams()

        # Warn about unknown keys (typos).
        for key in self._hyperparams:
            if key not in defaults:
                errors.append(f"Unknown hyperparameter '{key}'.")

        # Type-check against defaults where a default exists.
        for key, default_val in defaults.items():
            if key not in self._hyperparams:
                continue
            current = self._hyperparams[key]
            if not isinstance(current, type(default_val)):
                errors.append(
                    f"'{key}' should be {type(default_val).__name__}, "
                    f"got {type(current).__name__}."
                )

        return errors

    # ─────────────────────────────────────────────────────────────────────────
    # Model lifecycle  (build → compile → train → predict)
    # ─────────────────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def build(self) -> None:
        """
        Instantiate the underlying model architecture.

        Called once before training.  Should populate ``self._model``.
        """
        pass

    def compile(self) -> None:
        """
        Configure loss function, optimiser and metrics.

        Default implementation is a no-op (relevant mainly for PyTorch models).
        Override for models that require explicit compilation.
        """
        pass

    @abc.abstractmethod
    def train(self, X: Any, y: Any, **kwargs: Any) -> None:
        """
        Fit the model to training data.

        Parameters
        ----------
        X:
            Input features / fields.  Shape and type depend on the concrete model.
        y:
            Target values or class labels.
        **kwargs:
            Optional overrides forwarded to the underlying framework fit call.
        """
        pass

    def train_with_validation(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """
        Fit the model and evaluate on a held-out validation set each epoch.

        Returns
        -------
        dict[str, list[float]]
            Training history keyed by metric name, e.g. ``{"loss": [...], "val_loss": [...]}``.
        """
        epochs = int(self.get_hyperparam("epochs", 1))
        history: dict[str, list[float]] = {"train_score": [], "val_score": []}

        for epoch in range(1, epochs + 1):
            self.train(X_train, y_train, **kwargs)
            train_metrics = self.score(X_train, y_train)
            val_metrics = self.score(X_val, y_val)

            # Track the first metric returned by score() (e.g. r2 or accuracy).
            first_metric = next(iter(train_metrics))
            history["train_score"].append(train_metrics[first_metric])
            history["val_score"].append(val_metrics[first_metric])

            logger.debug(
                "Epoch %d/%d  train_%s=%.4f  val_%s=%.4f",
                epoch,
                epochs,
                first_metric,
                train_metrics[first_metric],
                first_metric,
                val_metrics[first_metric],
            )

        return history

    @abc.abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Return class probability estimates.

        Only applicable to classifiers.  Raises :class:`NotImplementedError` by
        default; override in classifier subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support probability estimates."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation & validation
    # ─────────────────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def score(self, X: Any, y: Any) -> dict[str, float]:
        """
        Evaluate the trained model and return a metrics dict.

        Returns
        -------
        dict[str, float]
            E.g. ``{"r2": 0.97, "rmse": 0.04}`` for regressors or
            ``{"accuracy": 0.95, "f1": 0.94}`` for classifiers.
        """
        pass

    def cross_validate(self, X: Any, y: Any, n_splits: int = 5) -> dict[str, Any]:
        """
        Run k-fold cross-validation and aggregate metrics.

        Parameters
        ----------
        n_splits:
            Number of folds.

        Returns
        -------
        dict[str, Any]
            Mean and std of each metric across folds.
        """
        from sklearn.model_selection import KFold

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if self._seed is None:
            logger.warning(" Seed in model base set to 42 as no seed was passed")
        seed = self._seed if self._seed is not None else 42
        kf = KFold(n_splits=n_splits, shuffle=True,
                   random_state=seed)

        fold_metrics: dict[str, list[float]] = {}

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_arr), start=1):
            # Deep-copy so we train a clean model per fold.
            fold_model = copy.deepcopy(self)
            fold_model.build()
            fold_model.compile()
            fold_model.train(X_arr[train_idx], y_arr[train_idx])

            metrics = fold_model.score(X_arr[test_idx], y_arr[test_idx])
            for name, value in metrics.items():
                fold_metrics.setdefault(name, []).append(value)

            logger.info("Fold %d/%d: %s", fold_idx, n_splits, metrics)

        # Aggregate across folds.
        results: dict[str, Any] = {}
        for name, values in fold_metrics.items():
            arr = np.array(values)
            results[name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "per_fold": values,
            }

        return results

    def feature_importance(self) -> dict[str, float] | None:
        """
        Return feature importance scores, if supported by the model.

        Returns ``None`` for models that do not expose feature importances.
        """
        return None

    def get_torch_module(self) -> Any:
        """
        Return the underlying ``torch.nn.Module`` for differentiable models.

        Only meaningful when :attr:`is_differentiable` is ``True``.
        Override in differentiable subclasses.

        Raises
        ------
        NotImplementedError
            If the model is not differentiable.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is not differentiable and has no torch module."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Serialisation  (save / load)
    # ─────────────────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Persist the trained model (weights + hyperparams + metadata) to *path*.

        The concrete format is up to the subclass (``torch.save``, ``joblib.dump``, …).
        """
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str | Path) -> "ModelBase":
        """
        Restore a model that was previously saved with :meth:`save`.

        Returns
        -------
        ModelBase
            A fully initialised, ready-to-predict model instance.
        """
        pass

    def export_onnx(self, path: str | Path, sample_input: Any = None) -> None:
        """
        Export the model to ONNX format for deployment.

        Parameters
        ----------
        sample_input:
            A representative input tensor required by the ONNX tracer.
        """
        self._require_trained("export_onnx")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "ONNX export requires PyTorch.  Install it with `pip install torch`."
            ) from exc

        if not isinstance(self._model, torch.nn.Module):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support ONNX export "
                f"(underlying model is not a torch.nn.Module)."
            )

        if sample_input is None:
            raise ValueError(
                "A `sample_input` tensor is required for ONNX tracing."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self._model,
            sample_input,
            str(path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
        )
        logger.info("%s: exported ONNX model → %s", self.model_name, path)

    def export_sklearn(self, path: str | Path) -> None:
        """
        Export the model to a joblib file (scikit-learn convention).

        This writes two artefacts into *path* (a directory):

        - ``model.joblib``  – the fitted sklearn estimator
        - ``meta.json``     – hyperparams, metadata, and model identity

        Parameters
        ----------
        path:
            Target directory.  Created if it does not exist.
        """
        self._require_trained("export_sklearn")

        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError(
                "sklearn export requires joblib.  Install it with "
                "`pip install joblib` (included with scikit-learn)."
            ) from exc

        from sklearn.base import BaseEstimator

        if not isinstance(self._model, BaseEstimator):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support sklearn export "
                f"(underlying model is not a sklearn BaseEstimator)."
            )

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_file = path / "model.joblib"
        meta_file = path / "meta.json"

        joblib.dump(self._model, model_file)

        meta = {
            "class": self.__class__.__qualname__,
            "model_name": self.model_name,
            "model_category": self.model_category,
            "hyperparams": self._hyperparams,
            "metadata": self._metadata,
        }
        meta_file.write_text(json.dumps(meta, indent=2, default=str))

        logger.info(
            "%s: exported sklearn model → %s  (model.joblib + meta.json)",
            self.model_name,
            path,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Metadata & introspection
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """
        Return a human-readable summary of the model architecture and parameters.

        Used by the frontend Inspector panel and CLI tooling.
        """
        lines: list[str] = [
            f"{'=' * 56}",
            f"  {self.model_name}  ({self.model_category})",
            f"{'=' * 56}",
            f"  Class       : {self.__class__.__qualname__}",
            f"  Trained     : {self._is_trained}",
            f"  Parameters  : {self.parameter_count():,}",
            f"{'─' * 56}",
            f"  Hyperparameters:",
        ]
        for key, value in self._hyperparams.items():
            lines.append(f"    {key:<28s} {value!r}")
        if self._metadata:
            lines.append(f"{'─' * 56}")
            lines.append("  Metadata:")
            for key, value in self._metadata.items():
                lines.append(f"    {key:<28s} {value!r}")
        lines.append(f"{'=' * 56}")
        return "\n".join(lines)

    def parameter_count(self) -> int:
        """Return the total number of trainable parameters."""
        if self._model is None:
            return 0

        # PyTorch models
        try:
            import torch
            if isinstance(self._model, torch.nn.Module):
                return sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        except ImportError:
            pass

        # scikit-learn estimators – count from coef_ / feature_importances_
        count = 0
        for attr in ("coef_", "intercept_", "feature_importances_", "dual_coef_", "support_vectors_"):
            val = getattr(self._model, attr, None)
            if val is not None:
                count += int(np.asarray(val).size)
        return count

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise model configuration (hyperparams + metadata) to a plain dict.

        Used when saving pipeline state to JSON.
        """
        return {
            "class": self.__class__.__qualname__,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_category": self.model_category,
            "hyperparams": copy.deepcopy(self._hyperparams),
            "metadata": copy.deepcopy(self._metadata),
            "is_trained": self._is_trained,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelBase":
        """
        Restore a model configuration from a dict produced by :meth:`to_dict`.

        Uses the registry when called on :class:`ModelBase` directly, or
        instantiates ``cls`` when called on a concrete subclass.
        """
        hyperparams = data.get("hyperparams")

        if cls is ModelBase:
            # Resolve via registry so we get the right subclass.
            model_type = data.get("model_type", "undefined")
            instance = cls.create(model_type, hyperparams=hyperparams)
        else:
            instance = cls(hyperparams=hyperparams)

        instance._metadata = data.get("metadata", {})
        instance._is_trained = data.get("is_trained", False)
        return instance

    # ─────────────────────────────────────────────────────────────────────────
    # Frontend ↔ Backend serialisation
    # ─────────────────────────────────────────────────────────────────────────

    def to_node_data(self) -> dict[str, Any]:
        """
        Serialise to the shape expected by the frontend node-data interfaces
        (``RegressorNodeData`` / ``ClassifierNodeData``).

        Returns a dict like::

            {
                "label": "MLP",
                "category": "regressor",
                "model": "MLP",
                "hyperparams": { ... },
                "is_trained": true,
                "metrics": { ... }
            }
        """
        data: dict[str, Any] = {
            "label": self.model_name,
            "category": self.model_category,
            "model": self.model_type,
            "hyperparams": copy.deepcopy(self._hyperparams),
            "is_trained": self._is_trained,
        }
        # Attach last-known metrics so the Inspector can show them.
        if "last_train_metrics" in self._metadata:
            data["metrics"] = self._metadata["last_train_metrics"]
        if "last_train_time_s" in self._metadata:
            data["train_time_s"] = self._metadata["last_train_time_s"]
        # Multi-channel info from Data Splitter pipeline.
        if "n_channels" in self._metadata:
            data["n_channels"] = self._metadata["n_channels"]
        if "per_channel_metrics" in self._metadata:
            data["per_channel_metrics"] = self._metadata["per_channel_metrics"]
        return data

    @classmethod
    def from_node_data(cls, node_data: dict[str, Any]) -> "ModelBase":
        """
        Factory: create a model instance from a frontend node-data payload.

        Looks up the concrete subclass via the ``model`` key (which matches
        ``model_type`` on the Python side) and passes ``hyperparams`` through.

        Parameters
        ----------
        node_data:
            A dict matching ``RegressorNodeData`` or ``ClassifierNodeData``
            from the TypeScript types, e.g.
            ``{"label": "MLP", "category": "regressor", "model": "MLP", "hyperparams": {...}}``.
        """
        model_type = node_data["model"]
        hyperparams = node_data.get("hyperparams")
        instance = cls.create(model_type, hyperparams=hyperparams)
        instance._role = node_data.get("role", "final")
        return instance

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline node interface  (called by the node graph executor)
    # ─────────────────────────────────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Node-graph entry point.  Receives the outputs of upstream nodes and
        returns a dict passed to downstream nodes.

        The default implementation routes to :meth:`train` or :meth:`predict`
        depending on the pipeline execution mode.

        If the input contains **multiple channel streams** (keyed as
        ``"channels"``, a list of arrays), this method delegates to
        :meth:`execute_multi_channel` which spawns one independent sub-model
        per channel.
        """
        # ── Multi-channel path (from Data Splitter) ──────────────────────────
        if "channels" in inputs:
            return self.execute_multi_channel(inputs)

        # ── Single-input path ────────────────────────────────────────────────
        mode = inputs.get("mode", "train")

        if mode == "train":
            X = inputs["X"]
            y = inputs["y"]

            if not self._model:
                self.build()
                self.compile()

            t0 = time.perf_counter()

            if "X_val" in inputs and "y_val" in inputs:
                history = self.train_with_validation(
                    X, y, inputs["X_val"], inputs["y_val"]
                )
            else:
                self.train(X, y)
                history = None

            elapsed = time.perf_counter() - t0
            metrics = self.score(X, y)

            self._metadata["last_train_time_s"] = round(elapsed, 4)
            self._metadata["last_train_metrics"] = metrics

            logger.info(
                "%s: training complete in %.2fs — %s",
                self.model_name,
                elapsed,
                metrics,
            )

            return {
                "model": self,
                "metrics": metrics,
                "history": history,
            }

        elif mode == "predict":
            self._require_trained("execute(mode='predict')")
            predictions = self.predict(inputs["X"])
            return {"predictions": predictions}

        else:
            raise ValueError(
                f"Unknown execution mode '{mode}'.  Expected 'train' or 'predict'."
            )

    def execute_multi_channel(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Handle multi-channel input from a Data Splitter node.

        When a Data Splitter decomposes ``(N, C, H, W)`` into C separate
        ``(N, H, W)`` streams, they arrive here as::

            inputs = {
                "mode": "train",
                "channels": [X_ch0, X_ch1, …, X_chC],
                "y": y,                   # shared target
                "y_val": y_val,           # optional
                "channels_val": [...],    # optional per-channel val data
            }

        This method creates **one independent sub-model per channel**, trains
        (or predicts with) each, and returns per-channel results::

            {
                "models": [model_ch0, model_ch1, …],
                "per_channel_metrics": [{…}, {…}, …],
                "predictions": [pred_ch0, pred_ch1, …],   # predict mode
            }

        Override in subclasses to change how channels are combined (e.g.
        concatenate latent vectors, ensemble predictions, etc.).
        """
        mode = inputs.get("mode", "train")
        channels: list[Any] = inputs["channels"]
        n_channels = len(channels)

        # Ensure we have a list of sub-models (one per channel).
        if not hasattr(self, "_channel_models") or self._channel_models is None:
            self._channel_models: list[ModelBase] = []
            for _ in range(n_channels):
                sub = copy.deepcopy(self)
                sub._channel_models = None          # prevent recursion
                sub.build()
                sub.compile()
                self._channel_models.append(sub)

        if mode == "train":
            y = inputs["y"]
            channels_val = inputs.get("channels_val")
            y_val = inputs.get("y_val")

            all_metrics: list[dict[str, float]] = []
            all_histories: list[dict[str, list[float]] | None] = []

            t0 = time.perf_counter()

            for i, (sub_model, X_ch) in enumerate(
                zip(self._channel_models, channels)
            ):
                if channels_val is not None and y_val is not None:
                    hist = sub_model.train_with_validation(
                        X_ch, y, channels_val[i], y_val
                    )
                else:
                    sub_model.train(X_ch, y)
                    hist = None

                metrics = sub_model.score(X_ch, y)
                all_metrics.append(metrics)
                all_histories.append(hist)

                logger.info(
                    "%s channel %d/%d: %s",
                    self.model_name, i, n_channels, metrics,
                )

            elapsed = time.perf_counter() - t0
            self._is_trained = all(m._is_trained for m in self._channel_models)
            self._metadata["last_train_time_s"] = round(elapsed, 4)
            self._metadata["per_channel_metrics"] = all_metrics
            self._metadata["n_channels"] = n_channels

            return {
                "models": self._channel_models,
                "per_channel_metrics": all_metrics,
                "per_channel_history": all_histories,
            }

        elif mode == "predict":
            predictions = []
            for sub_model, X_ch in zip(self._channel_models, channels):
                sub_model._require_trained("execute_multi_channel(predict)")
                predictions.append(sub_model.predict(X_ch))
            return {"predictions": predictions}

        else:
            raise ValueError(
                f"Unknown execution mode '{mode}'.  Expected 'train' or 'predict'."
            )

    def reset(self) -> None:
        """
        Tear down the current model instance and clear training state.

        Allows the same node to be re-trained without creating a new object.
        """
        self._model = None
        self._is_trained = False
        self._metadata.clear()
        if hasattr(self, "_channel_models"):
            self._channel_models = None
        logger.info("%s: model state reset.", self.model_name)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _require_trained(self, action: str) -> None:
        """Raise if the model has not been trained yet."""
        if not self._is_trained:
            raise RuntimeError(
                f"Cannot {action}: {self.model_name} has not been trained yet. "
                f"Call build() → train() first."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Dunder helpers
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return (
            f"{self.__class__.__name__}("
            f"category={self.model_category!r}, "
            f"status={status!r}, "
            f"hyperparams={self._hyperparams!r})"
        )
