"""
SurroMod – Automated Test Suite
================================
Loads saved workflow pickles from ``test/test_workflows/``, executes them
through the pipeline executor, and asserts that every trained model meets
the minimum R² threshold defined in ``test/configuration.yaml``.

Usage
-----
    pytest test/testsuite.py -v          # from project root
    python -m pytest test/testsuite.py   # alternative
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

# ── Ensure project root is on sys.path ────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent          # → SurroMod/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend.pipeline_executor import run_pipeline  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────

TEST_DIR        = Path(__file__).resolve().parent
WORKFLOWS_DIR   = TEST_DIR / "test_workflows"
CONFIG_PATH     = TEST_DIR / "configuration.yaml"
UPLOADS_DIR     = ROOT / "uploads"

# ── Load configuration ───────────────────────────────────────────────────

def _load_config() -> dict[str, Any]:
    """Read test/configuration.yaml and return the parsed dict."""
    if not CONFIG_PATH.exists():
        pytest.fail(f"Configuration file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def _workflow_ids() -> list[str]:
    """Return workflow filenames listed in configuration.yaml."""
    cfg = _load_config()
    return list(cfg.get("workflows", {}).keys())


# ── Helpers ──────────────────────────────────────────────────────────────

def _load_workflow(filename: str) -> dict[str, Any]:
    """Unpickle a workflow and restore its embedded data files."""
    path = WORKFLOWS_DIR / filename
    if not path.exists():
        pytest.fail(f"Workflow file not found: {path}")

    with open(path, "rb") as fh:
        bundle: dict[str, Any] = pickle.load(fh)

    # Restore embedded CSV / data files into uploads/
    UPLOADS_DIR.mkdir(exist_ok=True)
    for file_id, content in bundle.get("data_files", {}).items():
        dest = UPLOADS_DIR / file_id
        if not dest.exists():
            dest.write_bytes(content)

    return bundle


def _find_validator_results(
    node_results: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """
    Return a list of ``(node_id, result_dict)`` for every validator node.
    """
    validators: list[tuple[str, dict[str, Any]]] = []
    for nd in nodes:
        data = nd.get("data", nd)
        if data.get("category") == "validator":
            nid = nd["id"]
            if nid in node_results:
                validators.append((nid, node_results[nid]))
    return validators


def _extract_model_r2_values(
    validator_result: dict[str, Any],
) -> list[tuple[str, float]]:
    """
    Extract ``(model_name, r2)`` pairs from a validator result dict.

    Handles both single-model and multi-model formats:

    * **Single model**: result contains ``metrics.r2`` directly.
    * **Multi-model**: result contains ``model_results``, each with
      ``model_name`` and ``metrics.r2``.
    """
    pairs: list[tuple[str, float]] = []

    if validator_result.get("multi_model"):
        for entry in validator_result.get("model_results", []):
            name = entry.get("model_name", "unknown")
            r2 = entry.get("metrics", {}).get("r2", float("-inf"))
            pairs.append((name, r2))
    elif "metrics" in validator_result:
        r2 = validator_result["metrics"].get("r2", float("-inf"))
        pairs.append(("model", r2))

    return pairs


# ── Parametrised test ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    return _load_config()


@pytest.mark.parametrize("workflow_file", _workflow_ids())
def test_workflow_r2(workflow_file: str, config: dict[str, Any]) -> None:
    """
    End-to-end test for a single workflow:

    1. Load the workflow pickle.
    2. Run the pipeline.
    3. Find the validator node(s).
    4. For every model, assert that R² ≥ the configured minimum.
    """
    wf_config = config["workflows"][workflow_file]
    expected_models: list[dict[str, Any]] = wf_config.get("models", [])

    # ── 1. Load workflow ─────────────────────────────────────────────────
    bundle = _load_workflow(workflow_file)
    nodes = bundle["nodes"]
    edges = bundle["edges"]

    # ── 2. Execute the pipeline ──────────────────────────────────────────
    result = run_pipeline(nodes, edges)
    assert result is not None, "run_pipeline returned None"

    node_results: dict[str, Any] = result.get("node_results", {})
    assert node_results, "Pipeline produced no node results"

    # ── 3. Collect validator outputs ─────────────────────────────────────
    validators = _find_validator_results(node_results, nodes)
    assert validators, (
        f"Workflow '{workflow_file}' has no validator node with results"
    )

    # Flatten all model R² values across all validator nodes
    actual_models: list[tuple[str, float]] = []
    for _vid, vresult in validators:
        actual_models.extend(_extract_model_r2_values(vresult))

    # ── 4. Assert R² barriers ────────────────────────────────────────────
    assert len(actual_models) == len(expected_models), (
        f"Workflow '{workflow_file}': expected {len(expected_models)} "
        f"model(s) but got {len(actual_models)}.\n"
        f"  Expected: {[m['name'] for m in expected_models]}\n"
        f"  Actual:   {[name for name, _ in actual_models]}"
    )

    for idx, ((actual_name, actual_r2), expected) in enumerate(
        zip(actual_models, expected_models)
    ):
        exp_name = expected["name"]
        min_r2 = expected["min_r2"]

        assert actual_name == exp_name, (
            f"Model #{idx} name mismatch: expected '{exp_name}', "
            f"got '{actual_name}'"
        )
        assert actual_r2 >= min_r2, (
            f"Workflow '{workflow_file}', model '{actual_name}' (#{idx}): "
            f"R² = {actual_r2:.6f} < minimum {min_r2}"
        )

        # Print for visibility in verbose mode
        print(
            f"  ✓ {workflow_file} / {actual_name} #{idx}: "
            f"R² = {actual_r2:.6f}  (min = {min_r2})"
        )
