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
import datetime as dt
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from collections.abc import Callable

import yaml

logger = logging.getLogger(__name__)


def _dump_prompt(prompt_dump_dir: str | None, iteration: int, prompt: str) -> None:
    """Persist the constructed LLM prompt to disk (best effort)."""
    if not prompt_dump_dir:
        return
    try:
        out_dir = Path(prompt_dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_file = out_dir / f"prompt_iter_{iteration:04d}_{ts}.txt"
        out_file.write_text(prompt, encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not dump HP tuning prompt for iteration %d: %s", iteration, exc)


def _cuda_probe_ok() -> bool:
    """
    Spawn a tiny subprocess that triggers CUDA initialisation via the
    **driver** API (``libcuda.so.1``).

    Using the driver API instead of the runtime API (``libcudart.so``)
    avoids segfaults caused by CUDA toolkit / driver version mismatches
    that are common on WSL 2.

    Returns ``True`` if CUDA is usable, ``False`` if the probe fails.
    """
    script = (
        "import ctypes, sys\n"
        "try:\n"
        "    cuda = ctypes.CDLL('libcuda.so.1')\n"
        "    rc = cuda.cuInit(0)\n"
        "    if rc != 0:\n"
        "        sys.exit(1)\n"
        "    n = ctypes.c_int(0)\n"
        "    cuda.cuDeviceGetCount(ctypes.byref(n))\n"
        "    sys.exit(0 if n.value > 0 else 1)\n"
        "except Exception:\n"
        "    sys.exit(1)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False

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

        requested_gpu_layers = int(llm_cfg.get("n_gpu_layers", 0))

        # ── CUDA probe: detect broken GPU drivers before they segfault ──
        use_gpu_layers = requested_gpu_layers
        if requested_gpu_layers != 0:
            if _cuda_probe_ok():
                logger.info("CUDA probe succeeded — GPU offload enabled.")
            else:
                logger.warning(
                    "⚠ CUDA initialisation failed (GPU driver crash / not "
                    "available). Falling back to CPU-only mode "
                    "(n_gpu_layers forced to 0). LLM inference will be "
                    "slower. To fix: update your GPU driver or, on WSL 2, "
                    "ensure the Windows NVIDIA driver matches the WSL CUDA "
                    "toolkit version."
                )
                use_gpu_layers = 0
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

        llm = Llama(
            model_path=model_path,
            n_ctx=int(llm_cfg.get("n_ctx", 8192)),
            n_threads=int(llm_cfg.get("n_threads", 4)),
            n_gpu_layers=use_gpu_layers,
            main_gpu=int(llm_cfg.get("main_gpu", 0)),
            verbose=False,
        )

        # ── GPU verification ─────────────────────────────────────────
        gpu_active = False
        try:
            import llama_cpp
            # llama_cpp exposes supports_gpu_offload since v0.2.x
            if hasattr(llama_cpp, "llama_supports_gpu_offload"):
                gpu_active = llama_cpp.llama_supports_gpu_offload()
            elif hasattr(llama_cpp, "supports_gpu_offload"):
                gpu_active = llama_cpp.supports_gpu_offload()
            else:
                # Heuristic: if n_gpu_layers != 0, check if CUDA build
                gpu_active = use_gpu_layers != 0
        except Exception:  # noqa: BLE001
            pass

        if use_gpu_layers != 0 and not gpu_active:
            logger.warning(
                "GPU offload requested (n_gpu_layers=%d) but llama-cpp-python "
                "appears to be a CPU-only build. Reinstall with CUDA support:\n"
                "  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python "
                "--force-reinstall --no-cache-dir",
                use_gpu_layers,
            )
        elif use_gpu_layers != 0:
            logger.info(
                "GPU offload active (n_gpu_layers=%d, main_gpu=%d).",
                use_gpu_layers,
                int(llm_cfg.get("main_gpu", 0)),
            )
        else:
            logger.info("Running LLM on CPU only (n_gpu_layers=0).")

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
    """Build a compact, information-dense prompt for the LLM.

    Strategy:
    - Keep the immutable parts (search space, dataset) short.
    - Show a rolling window: top-K best + last-N iterations.
    - Include delta/trend feedback so the LLM can see what's working.
    - Produce an example JSON so the format is unambiguous.
    """
    direction = "maximize" if maximize else "minimize"
    progress = iteration / n_iterations
    if progress < 0.6:
        phase = "EXPLORE: try diverse, spread-out values"
    elif progress < 0.9:
        phase = "REFINE: narrow in on promising regions"
    else:
        phase = "EXPLOIT: fine-tune near the best configs found"

    # ── Search space (compact) ──────────────────────────────────────────
    space_lines: list[str] = []
    for p in selected_params:
        key = p["key"]
        ptype = p["type"]
        current = p["currentValue"]
        # Support discrete values (semicolon-separated list)
        discrete = p.get("discreteValues")
        if discrete and isinstance(discrete, list) and len(discrete) > 0:
            space_lines.append(f"  {key}: one of {discrete}")
        elif ptype == "number":
            lo, hi = p.get("min", 0), p.get("max", 1)
            step = p.get("step")
            is_int = isinstance(current, int) or (
                isinstance(current, float)
                and current == int(current)
                and (step is None or step == int(step))
            )
            t = "int" if is_int else "float"
            s = f", step {step}" if step else ""
            space_lines.append(f"  {key} ({t}): [{lo} .. {hi}]{s}")
        elif ptype == "string":
            opts = p.get("options", [])
            space_lines.append(f"  {key}: one of {opts}")
        elif ptype == "boolean":
            space_lines.append(f"  {key}: true | false")
    search_space = "\n".join(space_lines)

    # ── Dataset info (one-liner) ────────────────────────────────────────
    ds_line = ""
    if data_info:
        parts: list[str] = []
        ns = data_info.get("n_samples")
        nf = data_info.get("n_features", 0)
        nl = data_info.get("n_labels", 0)
        nt = data_info.get("n_train_samples")
        nh = data_info.get("n_holdout_samples")
        fn = data_info.get("file_name", "")
        if fn:
            parts.append(fn)
        if ns is not None:
            parts.append(f"{ns} samples")
        parts.append(f"{nf} features")
        parts.append(f"{nl} labels")
        if nt is not None and nh is not None:
            parts.append(f"train/holdout {nt}/{nh}")
        ds_line = f"Dataset: {', '.join(parts)}\n"

    # ── History: top-K best + last-N (with deltas) ──────────────────────
    TOP_K = 3
    LAST_N = 5

    history_text = ""
    best_text = ""
    if history:
        sorted_hist = sorted(
            history,
            key=lambda h: h["score"],
            reverse=maximize,
        )
        best_entry = sorted_hist[0]
        best_text = (
            f"Best so far: {scoring_metric}={best_entry['score']:.6f} "
            f"config={json.dumps(best_entry['config'], separators=(',', ':'))}\n"
        )

        # Top K
        top_k = sorted_hist[:TOP_K]
        top_lines = []
        for h in top_k:
            cfg = json.dumps(h["config"], separators=(",", ":"))
            top_lines.append(f"  #{h['iteration']} {scoring_metric}={h['score']:.6f} {cfg}")
        history_text += "Top configs:\n" + "\n".join(top_lines) + "\n"

        # Last N with deltas
        recent = history[-LAST_N:]
        rec_lines = []
        for i, h in enumerate(recent):
            cfg = json.dumps(h["config"], separators=(",", ":"))
            delta = ""
            if i > 0:
                prev_score = recent[i - 1]["score"]
                diff = h["score"] - prev_score
                arrow = "↑" if (diff > 0) == maximize else "↓" if diff != 0 else "→"
                delta = f" ({arrow}{abs(diff):.6f})"
            # Mark if this iteration had a penalty (pipeline failure)
            penalty = ""
            if (maximize and h["score"] <= -1e9) or (not maximize and h["score"] >= 1e9):
                penalty = " [FAILED]"
            rec_lines.append(
                f"  #{h['iteration']} {scoring_metric}={h['score']:.6f}{delta}{penalty} {cfg}"
            )
        history_text += "Recent:\n" + "\n".join(rec_lines) + "\n"

        # Trend summary
        if len(history) >= 3:
            last3 = [h["score"] for h in history[-3:]]
            if maximize:
                improving = last3[-1] > last3[0]
            else:
                improving = last3[-1] < last3[0]
            spread = max(last3) - min(last3)
            if spread < 1e-6:
                history_text += "Trend: stagnant — try a different region.\n"
            elif improving:
                history_text += "Trend: improving — keep refining this direction.\n"
            else:
                history_text += "Trend: worsening — change strategy.\n"
    else:
        history_text = "No evaluations yet.\n"

    # ── Example JSON ────────────────────────────────────────────────────
    example_config: dict[str, Any] = {}
    for p in selected_params:
        key = p["key"]
        discrete = p.get("discreteValues")
        if discrete and isinstance(discrete, list) and len(discrete) > 0:
            example_config[key] = discrete[0]
        elif p["type"] == "number":
            mid = (p.get("min", 0) + p.get("max", 1)) / 2
            cv = p["currentValue"]
            step = p.get("step")
            if isinstance(cv, int) or (step is not None and step == int(step)):
                example_config[key] = int(round(mid))
            else:
                example_config[key] = round(mid, 4)
        elif p["type"] == "string":
            opts = p.get("options", [p["currentValue"]])
            example_config[key] = opts[0] if opts else p["currentValue"]
        elif p["type"] == "boolean":
            example_config[key] = True
    example_json = json.dumps(example_config, separators=(", ", ": "))

    prompt = (
        f"You are an ML hyperparameter tuning agent. {direction} {scoring_metric}.\n"
        f"Make sure you dont use the same config twice. Always respond with ONLY a JSON object containing the next HP config to try.\n"
        f"{ds_line}"
        f"Be creative and try feasible ranges for the hyperparameters, not just small tweaks around the current values. The search space is:\n"
        f"\n{search_space}\n"
        f"\n{history_text}"
        f"{best_text}"
        f"\nIteration {iteration}/{n_iterations}. Strategy: {phase}\n"
        f"\nRespond with ONLY a JSON object, no explanation.\n"
        f"Format: {example_json}"
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

    # ── Fix common LLM mistakes: trailing commas, single quotes ──────────
    cleaned = text
    # Replace single quotes with double quotes
    cleaned = cleaned.replace("'", '"')
    # Remove trailing commas before closing braces
    cleaned = re.sub(r",\s*}", "}", cleaned)
    # Remove trailing commas before closing brackets
    cleaned = re.sub(r",\s*]", "]", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return _validate_and_cast(obj, selected_params)
        except json.JSONDecodeError:
            pass

    # ── Last resort: extract key-value pairs with regex ──────────────────
    param_keys = {p["key"] for p in selected_params}
    kv_config: dict[str, Any] = {}
    for key in param_keys:
        # Match patterns like: "key": value, key = value, key: value
        pattern = rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*(["\']?)(.+?)\1(?:\s*[,}}\n]|$)'
        kv_match = re.search(pattern, text, re.IGNORECASE)
        if kv_match:
            raw_val = kv_match.group(2).strip()
            # Try to parse as number/bool
            if raw_val.lower() in ("true", "false"):
                kv_config[key] = raw_val.lower() == "true"
            else:
                try:
                    kv_config[key] = json.loads(raw_val)
                except (json.JSONDecodeError, ValueError):
                    kv_config[key] = raw_val
    if kv_config:
        return _validate_and_cast(kv_config, selected_params)

    return None


def _validate_and_cast(
    config: dict[str, Any],
    selected_params: list[dict[str, Any]],
) -> dict[str, Any]:
    """Clamp numeric values to their ranges, cast types, and enforce discrete sets."""
    param_map = {p["key"]: p for p in selected_params}
    validated: dict[str, Any] = {}

    for key, spec in param_map.items():
        if key not in config:
            validated[key] = spec["currentValue"]
            continue

        val = config[key]
        ptype = spec["type"]

        # ── Discrete values override (semicolon-separated from frontend) ──
        discrete = spec.get("discreteValues")
        if discrete and isinstance(discrete, list) and len(discrete) > 0:
            # Snap to the closest allowed discrete value
            try:
                fval = float(val)
                closest = min(discrete, key=lambda d: abs(float(d) - fval))
                # Preserve int type if current value is int
                if isinstance(spec["currentValue"], int):
                    validated[key] = int(float(closest))
                else:
                    validated[key] = type(closest)(closest)
            except (TypeError, ValueError):
                # String match fallback
                sval = str(val).strip()
                matched = next((d for d in discrete if str(d).strip() == sval), None)
                validated[key] = matched if matched is not None else discrete[0]
            continue

        if ptype == "number":
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = float(spec["currentValue"])
            lo = float(spec.get("min", -math.inf))
            hi = float(spec.get("max", math.inf))
            val = max(lo, min(hi, val))
            step = spec.get("step")
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

# ── Metric extraction ────────────────────────────────────────────────────────

def _extract_metric(
    pipeline_result: dict[str, Any],
    scoring_metric: str,
) -> float | None:
    """
    Scan all node results from a pipeline run and return the requested metric.

    Preference order: validator nodes > rbl_aggregator > regressor.
    """
    full = _extract_metrics_full(pipeline_result, scoring_metric)
    return full.get("train")


def _extract_metrics_full(
    pipeline_result: dict[str, Any],
    scoring_metric: str,
) -> dict[str, Any]:
    """
    Extract both train and holdout metrics + model parameter count.

    Returns ``{"train": float|None, "holdout": float|None, "n_params": int|None}``.
    """
    node_results = pipeline_result.get("node_results", {})

    best_source: str | None = None
    best_train: float | None = None
    best_holdout: float | None = None
    best_n_params: int | None = None

    # Priority: higher number = preferred
    priority_map: dict[str, int] = {
        "validator": 3,
        "rbl_aggregator": 2,
        "regressor": 1,
    }

    for _nid, result in node_results.items():
        if not isinstance(result, dict):
            continue

        # ── Collect n_params from regressor / predictor nodes ─────────
        n_params = result.get("n_params")
        if n_params is not None:
            best_n_params = int(n_params)

        # ── Check for metrics at various nesting levels ──────────────
        metrics: dict[str, float] | None = None

        # Direct metrics dict (train)
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

            if best_train is None or current_priority > best_priority:
                best_train = val
                best_source = source_type
                # Extract holdout from same node
                holdout_block = result.get("holdout")
                if isinstance(holdout_block, dict):
                    h_metrics = holdout_block.get("metrics", {})
                    best_holdout = float(h_metrics[scoring_metric]) if scoring_metric in h_metrics else None
                else:
                    best_holdout = None
            elif current_priority == best_priority:
                best_train = val
                holdout_block = result.get("holdout")
                if isinstance(holdout_block, dict):
                    h_metrics = holdout_block.get("metrics", {})
                    best_holdout = float(h_metrics[scoring_metric]) if scoring_metric in h_metrics else None
                else:
                    best_holdout = None

    return {"train": best_train, "holdout": best_holdout, "n_params": best_n_params}


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
        """Query the LLM via the **chat completion** API.

        Instruct / chat-tuned models (Mistral-Instruct, Llama-Chat, etc.)
        expect a structured message list. ``llama-cpp-python``'s
        ``create_chat_completion`` applies the correct chat template
        automatically, whereas the raw completion API often produces
        empty or nonsensical output for these models.
        """
        if self._llm is None:
            return ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a hyperparameter tuning assistant. "
                    "Always respond with ONLY a single JSON object. "
                    "No explanation, no markdown fences, no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            result = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                repeat_penalty=1.1,
            )
            text = result["choices"][0]["message"]["content"] or ""
            logger.debug("LLM raw response: %.500s", text)
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
        metric_source: str = "train",
        seed: int | None = None,
        data_info: dict[str, Any] | None = None,
        prompt_dump_dir: str | None = None,
        stop_requested: Callable[[], bool] | None = None,
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
        metric_source : Which split to optimise on (``'train'`` or ``'holdout'``).
        seed : Optional random seed.
        data_info : Optional dict with training data metadata (n_samples,
            n_features, n_labels, column_dtypes, etc.).
        prompt_dump_dir : Optional directory where each prompt is written.

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

        metric_source_normalized = "holdout" if str(metric_source).lower() == "holdout" else "train"
        if metric_source_normalized == "holdout" and not bool((data_info or {}).get("has_train_test_split")):
            logger.warning(
                "HP Tuner: holdout optimization requested but no upstream TrainTestSplit was detected; "
                "falling back to train metrics."
            )
            metric_source_normalized = "train"

        maximize = _is_maximize(scoring_metric)
        history: list[dict[str, Any]] = []
        stopped = False

        # Map temperature: exploration_rate 0→0.2, 0.5→0.7, 1→1.0
        base_temp = 0.2 + exploration_rate * 0.8

        logger.info(
            "HP Tuner: starting %d iterations (metric=%s, source=%s, %s)",
            n_iterations,
            scoring_metric,
            metric_source_normalized,
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
            if stop_requested is not None and stop_requested():
                stopped = True
                logger.info(
                    "HP Tuner: stop requested before iteration %d; ending search.",
                    iteration,
                )
                break

            # ── 1. Get HP suggestion ─────────────────────────────────────
            config: dict[str, Any] | None = None
            llm_retries = 3

            prompt = _build_prompt(
                selected_params,
                history,
                scoring_metric,
                maximize,
                iteration,
                n_iterations,
                data_info=data_info,
            )
            _dump_prompt(prompt_dump_dir, iteration, prompt)
            # Anneal temperature: more exploration early, exploitation late
            progress = iteration / n_iterations
            temp = base_temp * (1.0 - 0.3 * progress)

            for attempt in range(llm_retries + 1):
                raw = self._ask_llm(prompt, temperature=temp)
                config = _parse_llm_response(raw, selected_params)
                if config is not None:
                    break
                logger.warning(
                    "HP Tuner iter %d: LLM parse failed (attempt %d/%d). "
                    "Raw response: %.200s",
                    iteration,
                    attempt + 1,
                    llm_retries + 1,
                    raw,
                )

            if config is None:
                logger.error(
                    "HP Tuner iter %d: LLM failed to produce a valid "
                    "config after %d attempts. Skipping this iteration.",
                    iteration,
                    llm_retries + 1,
                )
                penalty = -1e10 if maximize else 1e10
                history.append(
                    {
                        "iteration": iteration,
                        "config": {p["key"]: p["currentValue"] for p in selected_params},
                        "score": round(penalty, 8),
                        "train_score": round(penalty, 8),
                        "holdout_score": None,
                        "n_params": None,
                    }
                )
                continue

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
            train_score: float | None = None
            holdout_score: float | None = None
            n_params: int | None = None
            try:
                result = run_pipeline(modified_nodes, edges, seed=seed)
                metrics_full = _extract_metrics_full(result, scoring_metric)
                train_score = metrics_full.get("train")
                holdout_score = metrics_full.get("holdout")
                n_params = metrics_full.get("n_params")
                score = holdout_score if metric_source_normalized == "holdout" else train_score
            except Exception as exc:
                logger.error(
                    "HP Tuner iter %d: pipeline run failed: %s",
                    iteration,
                    exc,
                )

            if score is None:
                # Assign a very bad score so this config is not selected
                score = -1e10 if maximize else 1e10
                train_score = score
                logger.warning(
                    "HP Tuner iter %d: could not extract metric '%s' from %s split "
                    "— assigned penalty score.",
                    iteration,
                    scoring_metric,
                    metric_source_normalized,
                )

            entry: dict[str, Any] = {
                "iteration": iteration,
                "config": config,
                "score": round(score, 8),
                "train_score": round(train_score, 8) if train_score is not None else None,
                "holdout_score": round(holdout_score, 8) if holdout_score is not None else None,
                "n_params": n_params,
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
        if not history:
            if stopped:
                logger.info("HP Tuner: stopped before first completed iteration.")
                return {
                    "history": [],
                    "best_config": {p["key"]: p["currentValue"] for p in selected_params},
                    "best_score": 0.0,
                    "stopped": True,
                }

            logger.error("HP Tuner: no iterations completed. Returning empty result.")
            return {
                "history": [],
                "best_config": {p["key"]: p["currentValue"] for p in selected_params},
                "best_score": 0.0,
                "error": "All iterations failed — no valid configs were produced.",
                "stopped": False,
            }

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
            "stopped": stopped,
        }
