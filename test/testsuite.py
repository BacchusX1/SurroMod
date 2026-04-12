"""
SurroMod – Automated Test Suite
================================
Loads saved workflow pickles from ``test/test_workflows/``, executes them
through the pipeline executor, and asserts that every trained model meets
the minimum R² threshold defined in ``test/configuration.yaml``.

Also verifies that the local LLM configured in ``backend_config.yaml`` can
be loaded and produces valid responses (used by the agent-based HP tuner).
LLM tests run in subprocesses to isolate potential CUDA driver segfaults.

Usage
-----
    pytest test/testsuite.py -v          # from project root
    python -m pytest test/testsuite.py   # alternative
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import textwrap
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
BACKEND_CONFIG  = ROOT / "src" / "backend" / "backend_config.yaml"
PYTHON          = sys.executable

# ═══════════════════════════════════════════════════════════════════════════
# LLM test helpers (subprocess-isolated to survive CUDA segfaults)
# ═══════════════════════════════════════════════════════════════════════════

def _read_llm_config() -> dict[str, Any]:
    """Parse backend_config.yaml and return the llm section."""
    if not BACKEND_CONFIG.exists():
        pytest.skip(f"backend_config.yaml not found at {BACKEND_CONFIG}")
    with open(BACKEND_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("llm", {})


def _run_llm_script(
    script: str,
    *,
    timeout: int = 120,
    force_cpu: bool = False,
) -> subprocess.CompletedProcess[str]:
    """
    Run *script* in a child process with the project root on ``sys.path``.

    If the child segfaults (rc -11 / 139) and ``force_cpu`` is False,
    it retries once with ``CUDA_VISIBLE_DEVICES=""`` (CPU-only) so
    that a broken CUDA driver doesn't fail the test outright.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""

    result = subprocess.run(
        [PYTHON, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(ROOT),
    )

    # Segfault → retry CPU-only (unless we already are)
    if result.returncode in (-11, 139) and not force_cpu:
        print(
            "  ⚠  GPU/CUDA segfault detected — retrying in CPU-only mode "
            "(CUDA_VISIBLE_DEVICES=\"\")"
        )
        return _run_llm_script(script, timeout=timeout, force_cpu=True)

    return result


def _assert_script_ok(result: subprocess.CompletedProcess[str]) -> str:
    """Assert the subprocess succeeded and return its stdout."""
    if result.returncode != 0:
        detail = (result.stderr or "") + (result.stdout or "")
        pytest.fail(
            f"LLM subprocess exited with code {result.returncode}.\n"
            f"Output:\n{detail[-2000:]}"
        )
    return result.stdout

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

def _normalise_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Apply the same normalisation that the frontend performs before
    sending edges to the backend:

    1. **Deduplicate by edge ID** – pickled workflows may contain the
       same edge twice (once without ``targetHandle``, once with it).
       React Flow silently deduplicates by ID when loading into the
       store; we must do the same.  The *most complete* version (i.e.
       the one with more keys) wins so that ``targetHandle`` is retained.
    2. **Strip to API shape** – the frontend only sends ``source``,
       ``target``, ``sourceHandle`` and ``targetHandle`` to the backend
       (see ``store.ts → runPipeline``).  Passing extra fields like
       ``id`` can trigger legacy fallback-parsing in the executor that
       the frontend never hits.
    """
    # Deduplicate: keep the version with the most keys per edge ID
    by_id: dict[str, dict[str, Any]] = {}
    for e in edges:
        eid = e.get("id", id(e))
        prev = by_id.get(eid)
        if prev is None or len(e) > len(prev):
            by_id[eid] = e

    # Project to the four fields the frontend sends to /api/pipeline/run
    normalised: list[dict[str, Any]] = []
    for e in by_id.values():
        cleaned: dict[str, Any] = {
            "source": e["source"],
            "target": e["target"],
        }
        if e.get("sourceHandle") is not None:
            cleaned["sourceHandle"] = e["sourceHandle"]
        if e.get("targetHandle") is not None:
            cleaned["targetHandle"] = e["targetHandle"]
        normalised.append(cleaned)

    return normalised


def _load_workflow(filename: str) -> dict[str, Any]:
    """Unpickle a workflow and restore its embedded data files."""
    path = WORKFLOWS_DIR / filename
    # Try appending .pkl if the file is not found
    if not path.exists() and not filename.endswith(".pkl"):
        path = WORKFLOWS_DIR / f"{filename}.pkl"
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

    # Normalise edges to match frontend behaviour (dedup + strip)
    bundle["edges"] = _normalise_edges(bundle.get("edges", []))

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
    #  Match by model name (order-independent) so the test is robust
    #  against topological-sort differences between frontend and test.
    expected_by_name: dict[str, float] = {
        m["name"]: m["min_r2"] for m in expected_models
    }
    actual_names = {name for name, _ in actual_models}
    expected_names = set(expected_by_name.keys())

    assert actual_names == expected_names, (
        f"Workflow '{workflow_file}': model name mismatch.\n"
        f"  Expected: {sorted(expected_names)}\n"
        f"  Actual:   {sorted(actual_names)}"
    )

    for idx, (actual_name, actual_r2) in enumerate(actual_models):
        min_r2 = expected_by_name[actual_name]

        assert actual_r2 >= min_r2, (
            f"Workflow '{workflow_file}', model '{actual_name}': "
            f"R² = {actual_r2:.6f} < minimum {min_r2}"
        )

        # Print for visibility in verbose mode
        print(
            f"  ✓ {workflow_file} / {actual_name} #{idx}: "
            f"R² = {actual_r2:.6f}  (min = {min_r2})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3D Flow Prediction workflow tests
# ═══════════════════════════════════════════════════════════════════════════

def _flow_prediction_ids() -> list[str]:
    """Return workflow filenames listed under flow_prediction_workflows."""
    cfg = _load_config()
    return list(cfg.get("flow_prediction_workflows", {}).keys())


@pytest.mark.parametrize("workflow_file", _flow_prediction_ids())
def test_workflow_flow_prediction(workflow_file: str, config: dict[str, Any]) -> None:
    """
    End-to-end test for a 3D flow prediction workflow:

    1. Load the workflow pickle.
    2. Run the pipeline.
    3. Verify the regressor node was trained successfully.
    4. Assert R² targets for train and test splits.
    """
    wf_config = config["flow_prediction_workflows"][workflow_file]
    min_r2_train = wf_config.get("min_r2_train")
    min_r2_test = wf_config.get("min_r2_test")

    # ── 1. Load workflow ─────────────────────────────────────────────────
    bundle = _load_workflow(workflow_file)
    nodes = bundle["nodes"]
    edges = bundle["edges"]

    # ── 2. Execute the pipeline ──────────────────────────────────────────
    result = run_pipeline(nodes, edges, seed=42)
    assert result is not None, "run_pipeline returned None"

    node_results: dict[str, Any] = result.get("node_results", {})
    assert node_results, "Pipeline produced no node results"

    # ── 3. Find the regressor and verify training ────────────────────────
    regressor_found = False
    for nd in nodes:
        data = nd.get("data", nd)
        if data.get("category") == "regressor":
            nid = nd["id"]
            assert nid in node_results, (
                f"Regressor node '{nid}' missing from results"
            )
            nr = node_results[nid]
            assert nr.get("is_trained"), (
                f"Regressor node '{nid}' was not trained"
            )
            regressor_found = True
            label = data.get("label", data.get("model", nid))

            # ── 4. Assert R² targets ─────────────────────────────────────
            metrics = nr.get("metrics", {}) or {}
            r2_train = metrics.get("r2_train")
            r2_test = metrics.get("r2_test")

            if min_r2_train is not None and r2_train is not None:
                assert r2_train >= min_r2_train, (
                    f"Workflow '{workflow_file}', {label}: "
                    f"R²(train)={r2_train:.4f} < min {min_r2_train}"
                )
            if min_r2_test is not None and r2_test is not None:
                assert r2_test >= min_r2_test, (
                    f"Workflow '{workflow_file}', {label}: "
                    f"R²(test)={r2_test:.4f} < min {min_r2_test}"
                )

            print(
                f"  ✓ {workflow_file} / {label}: trained  "
                f"R²(train)={r2_train}  R²(test)={r2_test}"
            )

    assert regressor_found, (
        f"Workflow '{workflow_file}' contains no regressor node"
    )


# ═══════════════════════════════════════════════════════════════════════════
# LLM smoke tests (agent-based HP tuner)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def llm_config() -> dict[str, Any]:
    """Load and validate the LLM config section."""
    cfg = _read_llm_config()
    model_path = cfg.get("model_path", "")
    if not model_path or not Path(model_path).exists():
        pytest.skip(
            f"LLM model file not found: {model_path!r}. "
            "Update backend_config.yaml → llm.model_path."
        )
    return cfg


class TestLLMLoading:
    """Tests that the LLM can be loaded from backend_config.yaml."""

    def test_config_valid(self, llm_config: dict[str, Any]) -> None:
        """backend_config.yaml must contain a valid llm section."""
        assert "model_path" in llm_config
        assert Path(llm_config["model_path"]).exists(), (
            f"Model file does not exist: {llm_config['model_path']}"
        )
        print(f"  ✓ Model file: {llm_config['model_path']}")

    def test_llama_cpp_installed(self) -> None:
        """llama-cpp-python must be importable."""
        result = _run_llm_script(
            "import llama_cpp; print(llama_cpp.__version__)",
            timeout=15,
        )
        out = _assert_script_ok(result)
        print(f"  ✓ llama-cpp-python version: {out.strip()}")

    def test_load_llm(self, llm_config: dict[str, Any]) -> None:
        """_load_llm should return a working Llama instance."""
        script = textwrap.dedent(f"""\
            from src.backend.hp_tuner.agent_based import _load_llm
            llm = _load_llm({str(BACKEND_CONFIG)!r})
            assert llm is not None, "LLM returned None"
            assert hasattr(llm, "create_chat_completion"), (
                "LLM missing create_chat_completion method"
            )
            print("OK")
        """)
        result = _run_llm_script(script)
        out = _assert_script_ok(result)
        assert "OK" in out
        print("  ✓ _load_llm returned a valid Llama instance")


class TestLLMInference:
    """Tests that the LLM produces meaningful output."""

    def test_simple_chat_completion(self, llm_config: dict[str, Any]) -> None:
        """Send a trivial prompt and verify a non-empty response."""
        script = textwrap.dedent(f"""\
            import json
            from src.backend.hp_tuner.agent_based import _load_llm
            llm = _load_llm({str(BACKEND_CONFIG)!r})
            assert llm is not None, "LLM returned None"
            result = llm.create_chat_completion(
                messages=[
                    {{"role": "system", "content": "You are a helpful assistant."}},
                    {{"role": "user", "content": "Reply with the single word: hello"}},
                ],
                max_tokens=32,
                temperature=0.0,
            )
            text = result["choices"][0]["message"]["content"]
            assert text and len(text.strip()) > 0, "Empty response"
            print(json.dumps({{"response": text.strip()}}))
        """)
        result = _run_llm_script(script)
        out = _assert_script_ok(result)
        data = json.loads(out.strip().splitlines()[-1])
        assert len(data["response"]) > 0, "LLM returned empty response"
        print(f"  ✓ LLM response: {data['response']!r}")

    def test_json_generation(self, llm_config: dict[str, Any]) -> None:
        """Ask the LLM for JSON and verify it is parseable."""
        script = textwrap.dedent(f"""\
            import json, re
            from src.backend.hp_tuner.agent_based import _load_llm
            llm = _load_llm({str(BACKEND_CONFIG)!r})
            assert llm is not None, "LLM returned None"
            result = llm.create_chat_completion(
                messages=[
                    {{
                        "role": "system",
                        "content": (
                            "You are a hyperparameter tuning assistant. "
                            "Always respond with ONLY a single JSON object. "
                            "No explanation, no markdown fences, no extra text."
                        ),
                    }},
                    {{
                        "role": "user",
                        "content": (
                            'Suggest hyperparameters as JSON with keys '
                            '"learning_rate" (float 0.0001-0.1) and '
                            '"n_layers" (int 1-5). '
                            "Respond with ONLY a JSON object."
                        ),
                    }},
                ],
                max_tokens=128,
                temperature=0.0,
            )
            text = (result["choices"][0]["message"]["content"] or "").strip()
            assert len(text) > 0, "Empty response"
            # Strip markdown fences
            text = re.sub(r"^```(?:json)?\\s*", "", text)
            text = re.sub(r"\\s*```$", "", text)
            text = text.strip()
            if "{{" in text and "}}" not in text:
                text = text.rstrip() + "}}"
            parsed = json.loads(text)
            assert isinstance(parsed, dict), f"Expected dict, got {{type(parsed)}}"
            print(json.dumps({{"raw": text, "parsed": parsed}}))
        """)
        result = _run_llm_script(script)
        out = _assert_script_ok(result)
        data = json.loads(out.strip().splitlines()[-1])
        assert isinstance(data["parsed"], dict)
        print(f"  ✓ LLM raw JSON: {data['raw']!r}")
        print(f"  ✓ Parsed: {data['parsed']}")


class TestAgentBasedTunerLLM:
    """Tests the AgentBasedTuner._ask_llm integration end-to-end."""

    def test_tuner_ask_llm(self, llm_config: dict[str, Any]) -> None:
        """
        Instantiate AgentBasedTuner, call _ask_llm with a realistic
        HP-tuning prompt, and verify the response is valid JSON.
        """
        script = textwrap.dedent(f"""\
            import json, re
            from src.backend.hp_tuner.agent_based import AgentBasedTuner

            tuner = AgentBasedTuner(config_path={str(BACKEND_CONFIG)!r})
            tuner._ensure_llm()
            assert tuner._llm is not None, "Tuner LLM is None"

            response = tuner._ask_llm(
                prompt=(
                    "You are an ML hyperparameter tuning agent. maximize r2.\\n"
                    "Search space:\\n"
                    "  learning_rate: float [0.0001, 0.1] (current: 0.01)\\n"
                    "  n_layers: int [1, 5] (current: 3)\\n"
                    "\\nNo evaluations yet.\\n"
                    "\\nIteration 1/10. Strategy: EXPLORE: try diverse, spread-out values\\n"
                    "\\nRespond with ONLY a JSON object, no explanation.\\n"
                    'Format: {{"learning_rate": 0.01, "n_layers": 3}}'
                ),
                temperature=0.0,
                max_tokens=128,
            )

            assert isinstance(response, str) and len(response.strip()) > 0, (
                f"_ask_llm returned empty: {{response!r}}"
            )

            # Try parsing as JSON
            cleaned = response.strip()
            cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned)
            cleaned = re.sub(r"\\s*```$", "", cleaned)
            cleaned = cleaned.strip()
            if "{{" in cleaned and "}}" not in cleaned:
                cleaned = cleaned.rstrip() + "}}"

            parsed = json.loads(cleaned)
            assert isinstance(parsed, dict), f"Expected dict, got {{type(parsed)}}"
            print(json.dumps({{"response": response.strip(), "parsed": parsed}}))
        """)
        result = _run_llm_script(script, timeout=180)
        out = _assert_script_ok(result)
        data = json.loads(out.strip().splitlines()[-1])
        assert isinstance(data["parsed"], dict)
        print(f"  ✓ Tuner _ask_llm response: {data['response']!r}")
        print(f"  ✓ Parsed HP config: {data['parsed']}")