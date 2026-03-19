#!/usr/bin/env python3
"""
test_llm.py — Quick diagnostic to verify the LLM loads and runs on the GPU.

Usage:
    python scripts/test_llm.py            # auto-detect GPU vs CPU
    python scripts/test_llm.py --cpu      # force CPU-only mode
    python scripts/test_llm.py --gpu      # require GPU (fail if unavailable)

Reads model settings from src/backend/backend_config.yaml.

All LLM work runs in a subprocess to survive CUDA segfaults (common on WSL2
with mismatched NVIDIA drivers).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# ── Resolve project root ────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BACKEND_CONFIG = ROOT / "src" / "backend" / "backend_config.yaml"
PYTHON = sys.executable

# ── Helpers ──────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def _heading(msg: str) -> None:
    print(f"\n{BOLD}── {msg} ──{RESET}")


def _run_subprocess(
    script: str,
    *,
    timeout: int = 120,
    force_cpu: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run *script* in a child process, retrying CPU-only on segfault."""
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
        _warn("GPU/CUDA segfault detected — retrying in CPU-only mode")
        return _run_subprocess(script, timeout=timeout, force_cpu=True)

    return result


# ── CUDA probe (subprocess-safe) ────────────────────────────────────────────

def _cuda_probe() -> dict:
    """Probe CUDA via the driver API (not runtime) in a subprocess.

    Uses ``cuInit`` / ``cuDeviceGetCount`` from ``libcuda.so.1`` which is
    provided by the GPU driver and doesn't suffer from the toolkit-version
    mismatch that can crash ``libcudart.so``.
    """
    script = textwrap.dedent("""\
        import ctypes, json, sys
        try:
            cuda = ctypes.CDLL('libcuda.so.1')
            rc = cuda.cuInit(0)
            if rc != 0:
                print(json.dumps({"ok": False, "devices": 0, "name": "cuInit failed"}))
                sys.exit(0)
            n = ctypes.c_int(0)
            rc = cuda.cuDeviceGetCount(ctypes.byref(n))
            if rc != 0 or n.value <= 0:
                print(json.dumps({"ok": False, "devices": 0, "name": "no devices"}))
                sys.exit(0)
            # Get device name
            buf = ctypes.create_string_buffer(256)
            dev = ctypes.c_int(0)
            cuda.cuDeviceGet(ctypes.byref(dev), 0)
            cuda.cuDeviceGetName(buf, 256, dev)
            name = buf.value.decode("utf-8", errors="replace")
            print(json.dumps({"ok": True, "devices": n.value, "name": name}))
        except Exception as e:
            print(json.dumps({"ok": False, "devices": 0, "name": str(e)}))
    """)
    try:
        r = subprocess.run(
            [PYTHON, "-c", script],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode in (-11, 139):
            return {"ok": False, "devices": 0, "name": "segfault"}
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout.strip())
    except Exception:
        pass
    return {"ok": False, "devices": 0, "name": "probe failed"}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Test LLM loading and GPU inference")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu", action="store_true", help="Force CPU-only mode")
    mode.add_argument("--gpu", action="store_true", help="Require GPU (fail if unavailable)")
    args = parser.parse_args()

    # ── 1. Read config ───────────────────────────────────────────────────
    _heading("Configuration")

    if not BACKEND_CONFIG.exists():
        _fail(f"backend_config.yaml not found at {BACKEND_CONFIG}")
        return 1

    import yaml

    with open(BACKEND_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    llm_cfg = cfg.get("llm", {})
    model_path = llm_cfg.get("model_path", "")

    print(f"  Config:       {BACKEND_CONFIG}")
    print(f"  Model:        {model_path}")
    print(f"  n_ctx:        {llm_cfg.get('n_ctx', '?')}")
    print(f"  n_threads:    {llm_cfg.get('n_threads', '?')}")
    print(f"  n_gpu_layers: {llm_cfg.get('n_gpu_layers', '?')}")

    if not model_path or not Path(model_path).exists():
        _fail(f"Model file not found: {model_path}")
        return 1
    _ok("Model file exists")

    # ── 2. CUDA probe (subprocess) ───────────────────────────────────────
    _heading("CUDA Probe")

    cuda = _cuda_probe()
    if cuda["ok"]:
        _ok(f"CUDA available — {cuda['devices']} device(s): {cuda['name']}")
    else:
        _warn(f"CUDA unavailable ({cuda['name']})")

    # Determine GPU layer setting
    requested_gpu_layers = int(llm_cfg.get("n_gpu_layers", 0))
    if args.cpu:
        use_gpu_layers = 0
        force_cpu = True
        _warn("Forced CPU-only mode (--cpu)")
    elif args.gpu:
        if not cuda["ok"]:
            _fail("GPU required (--gpu) but no CUDA devices available")
            return 1
        use_gpu_layers = requested_gpu_layers if requested_gpu_layers != 0 else -1
        force_cpu = False
        _ok(f"GPU mode required — using n_gpu_layers={use_gpu_layers}")
    else:
        force_cpu = False
        if cuda["ok"] and requested_gpu_layers != 0:
            use_gpu_layers = requested_gpu_layers
            _ok(f"Auto-detected GPU — using n_gpu_layers={use_gpu_layers}")
        else:
            use_gpu_layers = 0
            force_cpu = True
            if requested_gpu_layers != 0 and not cuda["ok"]:
                _warn("GPU layers requested but CUDA unavailable — falling back to CPU")
            else:
                _ok("Using CPU mode (n_gpu_layers=0 in config)")

    # ── 3. Load + Infer (subprocess) ─────────────────────────────────────
    _heading("LLM Loading & Inference")

    # Use a smaller context window for CPU-only to speed up loading
    ctx_size = int(llm_cfg.get('n_ctx', 8192))
    if use_gpu_layers == 0:
        ctx_size = min(ctx_size, 2048)

    # The subprocess does: import, check GPU build, load model, run two
    # prompts, and prints a JSON report on stdout.
    worker_script = textwrap.dedent(f"""\
        import json, sys, time

        report = {{}}

        # Import check
        try:
            from llama_cpp import Llama
            import llama_cpp
        except ImportError:
            report["error"] = "llama-cpp-python not installed"
            print(json.dumps(report))
            sys.exit(0)

        # GPU build check
        gpu_build = False
        try:
            if hasattr(llama_cpp, "llama_supports_gpu_offload"):
                gpu_build = llama_cpp.llama_supports_gpu_offload()
            elif hasattr(llama_cpp, "supports_gpu_offload"):
                gpu_build = llama_cpp.supports_gpu_offload()
        except Exception:
            pass
        report["gpu_build"] = gpu_build

        # Load model
        t0 = time.perf_counter()
        try:
            llm = Llama(
                model_path={model_path!r},
                n_ctx={ctx_size},
                n_threads={int(llm_cfg.get('n_threads', 4))},
                n_gpu_layers={use_gpu_layers},
                main_gpu={int(llm_cfg.get('main_gpu', 0))},
                verbose=False,
            )
        except Exception as exc:
            report["error"] = f"Load failed: {{exc}}"
            print(json.dumps(report))
            sys.exit(0)
        report["load_time"] = round(time.perf_counter() - t0, 2)

        # Short inference
        t0 = time.perf_counter()
        try:
            out1 = llm("Q: What is 2 + 2?\\nA:", max_tokens=32, temperature=0.0, echo=False)
        except Exception as exc:
            report["error"] = f"Inference failed: {{exc}}"
            print(json.dumps(report))
            sys.exit(0)
        dt1 = time.perf_counter() - t0
        report["short_text"] = out1["choices"][0]["text"].strip()
        report["short_tokens"] = out1["usage"]["completion_tokens"]
        report["short_time"] = round(dt1, 2)

        # Longer benchmark (128 tokens)
        t0 = time.perf_counter()
        try:
            out2 = llm(
                "[INST] Explain in one paragraph what a surrogate model is "
                "in engineering. [/INST]",
                max_tokens=128, temperature=0.7, echo=False,
            )
        except Exception as exc:
            report["error"] = f"Benchmark failed: {{exc}}"
            print(json.dumps(report))
            sys.exit(0)
        dt2 = time.perf_counter() - t0
        report["bench_text"] = out2["choices"][0]["text"].strip()
        report["bench_tokens"] = out2["usage"]["completion_tokens"]
        report["bench_time"] = round(dt2, 2)

        print(json.dumps(report))
    """)

    # CPU-only 7B models need more time for loading + inference
    sub_timeout = 120 if use_gpu_layers != 0 else 300
    print(f"  Running LLM in subprocess (timeout {sub_timeout}s)…")
    result = _run_subprocess(worker_script, timeout=sub_timeout, force_cpu=force_cpu)

    if result.returncode != 0:
        detail = (result.stderr or "") + (result.stdout or "")
        _fail(f"Subprocess exited with code {result.returncode}")
        if detail.strip():
            print(f"  Detail: {detail.strip()[-500:]}")
        return 1

    try:
        report = json.loads(result.stdout.strip().split("\n")[-1])
    except (json.JSONDecodeError, IndexError):
        _fail("Could not parse subprocess output")
        if result.stdout.strip():
            print(f"  stdout: {result.stdout.strip()[-500:]}")
        return 1

    if "error" in report:
        _fail(report["error"])
        if "llama-cpp-python not installed" in report["error"]:
            print('    Install with: CMAKE_ARGS="-DGGML_CUDA=on" pip install '
                  "llama-cpp-python --no-cache-dir")
        return 1

    # Check if we ended up on CPU due to segfault retry
    landed_on_cpu = (result.returncode == 0 and force_cpu) or (
        use_gpu_layers != 0 and not report.get("gpu_build", False)
    )
    running_on = "CPU" if (use_gpu_layers == 0 or landed_on_cpu) else "GPU"

    # ── Report ───────────────────────────────────────────────────────────
    gpu_build = report.get("gpu_build", False)
    if gpu_build:
        _ok("llama-cpp-python has GPU offload support")
    else:
        _warn("llama-cpp-python is a CPU-only build")
        if args.gpu:
            _fail("GPU required but llama-cpp-python lacks GPU support.\n"
                  '    Reinstall: CMAKE_ARGS="-DGGML_CUDA=on" pip install '
                  "llama-cpp-python --force-reinstall --no-cache-dir")
            return 1

    load_time = report.get("load_time", 0)
    _ok(f"Model loaded in {load_time}s")

    short_tok = report.get("short_tokens", 0)
    short_time = report.get("short_time", 0)
    short_tps = short_tok / short_time if short_time > 0 else 0
    _ok(f"Short test: {short_tok} tokens in {short_time}s ({short_tps:.1f} tok/s)")
    print(f"  Output: {report.get('short_text', '')!r}")

    bench_tok = report.get("bench_tokens", 0)
    bench_time = report.get("bench_time", 0)
    bench_tps = bench_tok / bench_time if bench_time > 0 else 0
    _ok(f"Benchmark: {bench_tok} tokens in {bench_time}s ({bench_tps:.1f} tok/s)")
    bench_text = report.get("bench_text", "")
    print(f"  Output: {bench_text[:200]}{'…' if len(bench_text) > 200 else ''}")

    # ── Summary ──────────────────────────────────────────────────────────
    _heading("Summary")
    _ok(f"LLM is working on {BOLD}{running_on}{RESET}")
    if running_on == "GPU" and cuda["ok"]:
        _ok(f"GPU: {cuda['name']}")
    elif running_on == "CPU" and args.gpu:
        _fail("Expected GPU but ended up on CPU")
        return 1
    print(f"  Load time:    {load_time}s")
    print(f"  Throughput:   {bench_tps:.1f} tokens/s ({bench_tok}-token generation)")

    if running_on == "CPU" and cuda["ok"] and use_gpu_layers != 0:
        _warn("CUDA was detected but LLM fell back to CPU (likely driver segfault).")
        _warn("Update your NVIDIA driver or WSL CUDA toolkit to fix GPU offload.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
