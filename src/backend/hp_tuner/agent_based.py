"""
Agent-Based HP Tuner
====================
LLM-driven hyperparameter optimisation.

An LLM (loaded from a local GGUF model via ``llama-cpp-python``) iteratively
proposes hyperparameter configurations, observes the resulting validation
score from a full pipeline run, and uses the growing evaluation history to
guide its next suggestion — combining the flexibility of LLMs with the
rigour of systematic hyperparameter search.

Flow
----
1.  Load the LLM from ``backend_config.yaml``.
2.  For each iteration:
    a.  Build a structured prompt with the search space + history.
    b.  Ask the LLM to return the next HP config as JSON.
    c.  Parse the response (with retries / random fallback).
    d.  Patch the predictor node's hyperparams in the pipeline.
    e.  Run the full pipeline via ``run_pipeline``.
    f.  Extract the target metric from node results.
    g.  Append ``{config, score}`` to the history.
3.  Return the full history + best config.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Metric direction ─────────────────────────────────────────────────────────

_MAXIMIZE_METRICS = {"r2", "accuracy", "f1", "precision", "recall"}
_MINIMIZE_METRICS = {"rmse", "mae", "mse", "loss"}


def _is_maximize(metric: str) -> bool:
    m = metric.lower().strip()
    if m in _MAXIMIZE_METRICS:
        return True
    if m in _MINIMIZE_METRICS:
        return False
    # Default: assume higher is better
    return True


# ── LLM loading ──────────────────────────────────────────────────────────────

def _load_llm(config_path: str) -> Any:
    """
    Load a ``llama_cpp.Llama`` instance using settings from the YAML config.

    Returns ``None`` if the model file does not exist or ``llama-cpp-python``
    is not installed (the tuner falls back to random search).
    """
    try:
        cfg = yaml.safe_load(Path(config_path).read_text())
    except Exception as exc:
        logger.warning("Could not read backend_config.yaml: %s — LLM disabled.", exc)
        return None

    llm_cfg = cfg.get("llm", {})
    model_path = llm_cfg.get("model_path", "")

    if not model_path or not Path(model_path).exists():
        logger.warning(
            "LLM model file not found: %s — falling back to random search.",
            model_path,
        )
        return None

    try:
        from llama_cpp import Llama  # type: ignore[import-untyped]

        llm = Llama(
            model_path=model_path,
            n_ctx=int(llm_cfg.get("n_ctx", 8192)),
            n_threads=int(llm_cfg.get("n_threads", 4)),
            n_gpu_layers=int(llm_cfg.get("n_gpu_layers", 0)),
            main_gpu=int(llm_cfg.get("main_gpu", 0)),
            verbose=False,
        )
        logger.info("LLM loaded from %s (ctx=%d)", model_path, llm_cfg.get("n_ctx", 8192))
        return llm
    except ImportError:
        logger.warning(
            "llama-cpp-python is not installed — falling back to random search. "
            "Install with: pip install llama-cpp-python"
        )
        return None
    except Exception as exc:
        logger.warning("LLM loading failed: %s — falling back to random search.", exc)
        return None


# ── Prompt construction ──────────────────────────────────────────────────────

def _build_prompt(
    selected_params: list[dict[str, Any]],
    history: list[dict[str, Any]],
    scoring_metric: str,
    maximize: bool,
    iteration: int,
    n_iterations: int,
    data_info: dict[str, Any] | None = None,
) -> str:
    """Build the instruction prompt for the LLM."""
    direction = "higher is better" if maximize else "lower is better"
    phase = (
        "EXPLORE diverse regions of the search space"
        if iteration < n_iterations / 3
        else "EXPLOIT promising regions near the best configurations found so far"
    )

    # Search space description
    space_lines: list[str] = []
    for p in selected_params:
        key = p["key"]
        ptype = p["type"]
        current = p["currentValue"]
        if ptype == "number":
            lo, hi = p.get("min", 0), p.get("max", 1)
            step = p.get("step")
            step_hint = f", step={step}" if step else ""
            # Determine if integer
            is_int = isinstance(current, int) or (
                isinstance(current, float)
                and current == int(current)
                and (step is None or step == int(step))
            )
            type_label = "int" if is_int else "float"
            space_lines.append(
                f"  - {key} ({type_label}): range [{lo}, {hi}]{step_hint}, current={current}"
            )
        elif ptype == "string":
            options = p.get("options", [])
            space_lines.append(
                f"  - {key} (categorical): options={options}, current=\"{current}\""
            )
        elif ptype == "boolean":
            space_lines.append(
                f"  - {key} (boolean): true or false, current={current}"
            )
    search_space = "\n".join(space_lines)

    # History
    if history:
        best_entry = (
            max(history, key=lambda h: h["score"])
            if maximize
            else min(history, key=lambda h: h["score"])
        )
        hist_lines = []
        for h in history:
            cfg_str = json.dumps(h["config"], separators=(", ", ": "))
            marker = " ★ BEST" if h is best_entry else ""
            hist_lines.append(
                f"  Iter {h['iteration']}: {cfg_str} → {scoring_metric}={h['score']:.6f}{marker}"
            )
        history_text = "\n".join(hist_lines)
        best_text = (
            f"Current best: {scoring_metric}={best_entry['score']:.6f} "
            f"with {json.dumps(best_entry['config'], separators=(', ', ': '))}"
        )
    else:
        history_text = "  (no evaluations yet)"
        best_text = "(no evaluations yet)"

    # Example JSON for the expected output
    example_config: dict[str, Any] = {}
    for p in selected_params:
        key = p["key"]
        if p["type"] == "number":
            mid = (p.get("min", 0) + p.get("max", 1)) / 2
            example_config[key] = round(mid, 4)
        elif p["type"] == "string":
            opts = p.get("options", [p["currentValue"]])
            example_config[key] = opts[0] if opts else p["currentValue"]
        elif p["type"] == "boolean":
            example_config[key] = True
    example_json = json.dumps(example_config, separators=(", ", ": "))

    # ── Dataset information section ─────────────────────────────────────
    dataset_section = ""
    if data_info:
        ds_lines: list[str] = []

        n_features = data_info.get("n_features", 0)
        n_labels = data_info.get("n_labels", 0)
        n_samples = data_info.get("n_samples")
        n_train = data_info.get("n_train_samples")
        n_holdout = data_info.get("n_holdout_samples")
        input_kind = data_info.get("input_kind", "unknown")
        file_name = data_info.get("file_name", "")
        holdout_ratio = data_info.get("holdout_ratio")
        col_dtypes = data_info.get("column_dtypes", {})
        feature_names = data_info.get("feature_names", [])
        label_names = data_info.get("label_names", [])

        if file_name:
            ds_lines.append(f"  - Data file: {file_name}")
        ds_lines.append(f"  - Input type: {input_kind}")
        if n_samples is not None:
            ds_lines.append(f"  - Total samples: {n_samples}")
        if n_train is not None and n_holdout is not None:
            ds_lines.append(
                f"  - Train / holdout split: {n_train} / {n_holdout}"
                f" (holdout_ratio={holdout_ratio})"
            )
        ds_lines.append(f"  - Number of features (inputs): {n_features}")
        ds_lines.append(f"  - Number of labels (outputs): {n_labels}")

        if feature_names:
            ds_lines.append(f"  - Feature columns: {', '.join(feature_names)}")
        if label_names:
            ds_lines.append(f"  - Label columns: {', '.join(label_names)}")

        if col_dtypes:
            dtype_summary: dict[str, int] = {}
            for _col, dt in col_dtypes.items():
                dtype_summary[dt] = dtype_summary.get(dt, 0) + 1
            dtype_parts = [f"{cnt}× {dt}" for dt, cnt in dtype_summary.items()]
            ds_lines.append(f"  - Column types: {', '.join(dtype_parts)}")

        dataset_section = "\n## Dataset Information\n" + "\n".join(ds_lines) + "\n\n"

    prompt = (
        "You are an expert machine learning hyperparameter tuning agent. "
        "Your goal is to find the best hyperparameters for a model by "
        "suggesting configurations to evaluate.\n\n"
        f"## Search Space\n{search_space}\n\n"
        f"{dataset_section}"
        f"## Evaluation History (metric: {scoring_metric}, {direction})\n"
        f"{history_text}\n\n"
        f"{best_text}\n\n"
        f"## Strategy\n"
        f"This is iteration {iteration}/{n_iterations}. You should {phase}.\n"
        "Consider the dataset characteristics (size, dimensionality, data types) "
        "when choosing hyperparameters — e.g. smaller datasets benefit from "
        "stronger regularisation, high-dimensional data may need lower complexity, "
        "and the train/holdout split affects over-fitting risk.\n\n"
        "## Instructions\n"
        "Suggest the NEXT hyperparameter configuration to evaluate.\n"
        "- For numeric parameters, stay within the specified ranges.\n"
        "- For categorical parameters, choose one of the listed options.\n"
        "- For boolean parameters, use true or false.\n\n"
        "Respond with ONLY a valid JSON object. "
        "No explanation, no markdown fences, no extra text.\n"
        f"Example format: {example_json}"
    )
    return prompt


# ── LLM response parsing ────────────────────────────────────────────────────

def _parse_llm_response(
    text: str,
    selected_params: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Extract a JSON HP config from the LLM's text output.

    Returns ``None`` if parsing fails.
    """
    text = text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return _validate_and_cast(obj, selected_params)
    except json.JSONDecodeError:
        pass

    # Try to find the first JSON object in the text
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return _validate_and_cast(obj, selected_params)
        except json.JSONDecodeError:
            pass

    # Try a more permissive regex for nested/multiline JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return _validate_and_cast(obj, selected_params)
        except json.JSONDecodeError:
            pass

    return None


def _validate_and_cast(
    config: dict[str, Any],
    selected_params: list[dict[str, Any]],
) -> dict[str, Any]:
    """Clamp numeric values to their ranges and cast types."""
    param_map = {p["key"]: p for p in selected_params}
    validated: dict[str, Any] = {}

    for key, spec in param_map.items():
        if key not in config:
            # Use current value as fallback
            validated[key] = spec["currentValue"]
            continue

        val = config[key]
        ptype = spec["type"]

        if ptype == "number":
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = float(spec["currentValue"])
            lo = float(spec.get("min", -math.inf))
            hi = float(spec.get("max", math.inf))
            val = max(lo, min(hi, val))
            step = spec.get("step")
            # Round to step if integer-like
            cv = spec["currentValue"]
            if isinstance(cv, int) or (
                step is not None and step == int(step)
            ):
                val = int(round(val))
            elif step:
                val = round(val / step) * step
                val = round(val, 8)
            validated[key] = val

        elif ptype == "string":
            options = spec.get("options", [])
            if options and str(val) not in options:
                # Pick closest match or fallback
                val_lower = str(val).lower()
                for opt in options:
                    if opt.lower() == val_lower:
                        val = opt
                        break
                else:
                    val = spec["currentValue"]
            validated[key] = str(val)

        elif ptype == "boolean":
            if isinstance(val, bool):
                validated[key] = val
            elif isinstance(val, str):
                validated[key] = val.lower() in ("true", "1", "yes")
            else:
                validated[key] = bool(val)
        else:
            validated[key] = val

    return validated


# ── Random config fallback ───────────────────────────────────────────────────

def _random_config(selected_params: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a random configuration within the search space."""
    config: dict[str, Any] = {}
    for p in selected_params:
        key = p["key"]
        ptype = p["type"]
        if ptype == "number":
            lo = float(p.get("min", 0))
            hi = float(p.get("max", 1))
            step = p.get("step")
            cv = p["currentValue"]
            if isinstance(cv, int) or (step is not None and step == int(step)):
                config[key] = random.randint(int(lo), int(hi))
            else:
                val = random.uniform(lo, hi)
                if step:
                    val = round(val / step) * step
                config[key] = round(val, 8)
        elif ptype == "string":
            options = p.get("options", [p["currentValue"]])
            config[key] = random.choice(options) if options else p["currentValue"]
        elif ptype == "boolean":
            config[key] = random.choice([True, False])
    return config


# ── Metric extraction ────────────────────────────────────────────────────────

def _extract_metric(
    pipeline_result: dict[str, Any],
    scoring_metric: str,
) -> float | None:
    """
    Scan all node results from a pipeline run and return the requested metric.

    Preference order: validator nodes > rbl_aggregator > regressor.
    """
    node_results = pipeline_result.get("node_results", {})

    best_source: str | None = None
    best_value: float | None = None

    # Priority: higher number = preferred
    priority_map: dict[str, int] = {
        "validator": 3,
        "rbl_aggregator": 2,
        "regressor": 1,
    }

    for _nid, result in node_results.items():
        if not isinstance(result, dict):
            continue

        # Check for metrics at various nesting levels
        metrics: dict[str, float] | None = None

        # Direct metrics dict
        if "metrics" in result and isinstance(result["metrics"], dict):
            metrics = result["metrics"]

        if metrics is None:
            continue

        if scoring_metric in metrics:
            val = float(metrics[scoring_metric])
            # Determine this node's priority
            source_type = "regressor"  # default
            if "per_label" in result or "multi_model" in result:
                source_type = "validator"
            elif "y_hat" in result:
                source_type = "rbl_aggregator"

            current_priority = priority_map.get(source_type, 0)
            best_priority = priority_map.get(best_source, -1) if best_source else -1

            if best_value is None or current_priority > best_priority:
                best_value = val
                best_source = source_type
            elif current_priority == best_priority:
                # Same priority — keep latest
                best_value = val

    return best_value


# ── Main tuner class ─────────────────────────────────────────────────────────

class AgentBasedTuner:
    """
    LLM-driven hyperparameter tuner.

    Parameters
    ----------
    config_path : str
        Path to ``backend_config.yaml`` containing LLM settings.
    """

    def __init__(self, config_path: str = "") -> None:
        self._config_path = config_path
        self._llm: Any = None
        self._llm_loaded = False

    def _ensure_llm(self) -> None:
        """Lazy-load the LLM on first use."""
        if not self._llm_loaded:
            self._llm = _load_llm(self._config_path)
            self._llm_loaded = True

    def _ask_llm(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Query the LLM and return the raw text response."""
        if self._llm is None:
            return ""

        try:
            result = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                repeat_penalty=1.1,
                stop=["}\n\n", "\n\n\n"],
            )
            text = result["choices"][0]["text"]
            # The stop token might have cut off the closing brace
            if "{" in text and "}" not in text:
                text = text.rstrip() + "}"
            return text
        except Exception as exc:
            logger.warning("LLM inference failed: %s", exc)
            return ""

    def run(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        predictor_node_id: str,
        selected_params: list[dict[str, Any]],
        n_iterations: int = 50,
        exploration_rate: float = 0.1,
        scoring_metric: str = "r2",
        seed: int | None = None,
        data_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute agent-based hyperparameter search.

        Parameters
        ----------
        nodes, edges : Pipeline graph from the frontend.
        predictor_node_id : ID of the node whose HPs are being tuned.
        selected_params : List of param specs with key, type, min/max/options.
        n_iterations : Number of HP evaluations.
        exploration_rate : Controls LLM temperature (0=greedy, 1=creative).
        scoring_metric : Metric name to optimise (e.g. ``'r2'``, ``'rmse'``).
        seed : Optional random seed.
        data_info : Optional dict with training data metadata (n_samples,
            n_features, n_labels, column_dtypes, etc.).

        Returns
        -------
        dict with ``history``, ``best_config``, ``best_score``.
        """
        from src.backend.pipeline_executor import run_pipeline

        self._ensure_llm()

        if self._llm is None:
            msg = (
                "HP Tuner: LLM is not available. "
                "Ensure llama-cpp-python is installed and "
                "backend_config.yaml points to a valid GGUF model file."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if seed is not None:
            random.seed(seed)

        maximize = _is_maximize(scoring_metric)
        history: list[dict[str, Any]] = []

        # Map temperature: exploration_rate 0→0.2, 0.5→0.7, 1→1.0
        base_temp = 0.2 + exploration_rate * 0.8

        logger.info(
            "HP Tuner: starting %d iterations (metric=%s, %s)",
            n_iterations,
            scoring_metric,
            "maximize" if maximize else "minimize",
        )

        if data_info:
            logger.info(
                "HP Tuner: dataset — %d features, %d labels, %s samples, input=%s, file=%s",
                data_info.get("n_features", 0),
                data_info.get("n_labels", 0),
                data_info.get("n_samples", "?"),
                data_info.get("input_kind", "?"),
                data_info.get("file_name", "?"),
            )

        for iteration in range(1, n_iterations + 1):
            # ── 1. Get HP suggestion ─────────────────────────────────────
            config: dict[str, Any] | None = None
            llm_retries = 2

            prompt = _build_prompt(
                selected_params,
                history,
                scoring_metric,
                maximize,
                iteration,
                n_iterations,
                data_info=data_info,
            )
            # Anneal temperature: more exploration early, exploitation late
            progress = iteration / n_iterations
            temp = base_temp * (1.0 - 0.3 * progress)

            for attempt in range(llm_retries + 1):
                raw = self._ask_llm(prompt, temperature=temp)
                config = _parse_llm_response(raw, selected_params)
                if config is not None:
                    break
                logger.warning(
                    "HP Tuner iter %d: LLM parse failed (attempt %d/%d): %s - prompt: %s",
                    iteration,
                    attempt + 1,
                    llm_retries + 1,
                    prompt,
                )

            if config is None:
                msg = (
                    f"HP Tuner iter {iteration}: LLM failed to produce a valid "
                    f"config after {llm_retries + 1} attempts. Stopping tuning."
                )
                logger.error(msg)
                raise RuntimeError(msg)

            # ── 2. Patch predictor HPs in the pipeline ───────────────────
            modified_nodes = copy.deepcopy(nodes)
            for nd in modified_nodes:
                nid = nd.get("id", "")
                if nid == predictor_node_id:
                    nd_data = nd.get("data", nd)
                    hp = nd_data.get("hyperparams", {})
                    hp.update(config)
                    nd_data["hyperparams"] = hp
                    break

            # ── 3. Run the pipeline ──────────────────────────────────────
            score: float | None = None
            try:
                result = run_pipeline(modified_nodes, edges, seed=seed)
                score = _extract_metric(result, scoring_metric)
            except Exception as exc:
                logger.error(
                    "HP Tuner iter %d: pipeline run failed: %s",
                    iteration,
                    exc,
                )

            if score is None:
                # Assign a very bad score so this config is not selected
                score = -1e10 if maximize else 1e10
                logger.warning(
                    "HP Tuner iter %d: could not extract metric '%s' "
                    "— assigned penalty score.",
                    iteration,
                    scoring_metric,
                )

            entry = {
                "iteration": iteration,
                "config": config,
                "score": round(score, 8),
            }
            history.append(entry)

            logger.info(
                "HP Tuner iter %d/%d: %s=%.6f  config=%s",
                iteration,
                n_iterations,
                scoring_metric,
                score,
                json.dumps(config, separators=(", ", ": ")),
            )

        # ── Find best ────────────────────────────────────────────────────
        if maximize:
            best = max(history, key=lambda h: h["score"])
        else:
            best = min(history, key=lambda h: h["score"])

        logger.info(
            "HP Tuner: done. Best %s=%.6f with %s",
            scoring_metric,
            best["score"],
            json.dumps(best["config"], separators=(", ", ": ")),
        )

        return {
            "history": history,
            "best_config": best["config"],
            "best_score": best["score"],
        }
