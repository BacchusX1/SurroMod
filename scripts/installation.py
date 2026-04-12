#!/usr/bin/env python3
"""
SurroMod Installation Script
=============================
Sets up a fresh environment on a new machine:

  1. Checks Python >= 3.10 and Node.js >= 18
  2. Creates a virtual environment at <repo_root>/venv/
  3. Installs all Python backend dependencies (PyTorch with CUDA if available)
  4. Optionally installs llama-cpp-python for agent-based HP tuning
  5. Installs npm frontend dependencies
  6. Creates a default backend_config.yaml if one does not exist

Usage:
    python scripts/installation.py               # full install (auto-detects GPU)
    python scripts/installation.py --cpu         # force CPU-only PyTorch
    python scripts/installation.py --skip-llm    # skip llama-cpp-python
    python scripts/installation.py --skip-npm    # skip frontend npm install
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR   = Path(__file__).resolve().parent.parent
VENV_DIR   = ROOT_DIR / "venv"
FRONTEND   = ROOT_DIR / "src" / "frontend"
CONFIG_DST = ROOT_DIR / "src" / "backend" / "backend_config.yaml"

# ─── ANSI colours ─────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty() and platform.system() != "Windows"

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def ok(msg: str)    -> None: print(_c(f"  ✔  {msg}", "32"))
def info(msg: str)  -> None: print(_c(f"  ·  {msg}", "36"))
def warn(msg: str)  -> None: print(_c(f"  ⚠  {msg}", "33"))
def error(msg: str) -> None: print(_c(f"  ✖  {msg}", "31"))
def step(msg: str)  -> None: print(_c(f"\n── {msg}", "1;34"))

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: list[str], *, cwd: Path = ROOT_DIR, check: bool = True) -> subprocess.CompletedProcess:
    info(" ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=check)


def _run_venv(args: list[str], *, cwd: Path = ROOT_DIR) -> subprocess.CompletedProcess:
    """Run a command using the pip/python inside the venv."""
    pip = _venv_bin("pip")
    return _run([str(pip)] + args, cwd=cwd)


def _venv_bin(name: str) -> Path:
    """Return the path to a binary inside the venv."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / (name + ".exe")
    return VENV_DIR / "bin" / name


def _check_python() -> None:
    step("Checking Python version")
    v = sys.version_info
    info(f"Python {v.major}.{v.minor}.{v.micro} — {sys.executable}")
    if (v.major, v.minor) < (3, 10):
        error(f"Python 3.10+ is required (found {v.major}.{v.minor}).")
        sys.exit(1)
    ok("Python version OK")


def _check_node() -> bool:
    step("Checking Node.js / npm")
    node = shutil.which("node")
    npm  = shutil.which("npm")
    if not node or not npm:
        warn("Node.js / npm not found — frontend installation will be skipped.")
        warn("Install Node.js >= 18 from https://nodejs.org and re-run to set up the frontend.")
        return False

    ver_raw = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
    try:
        major = int(ver_raw.lstrip("v").split(".")[0])
    except ValueError:
        major = 0

    if major < 18:
        warn(f"Node.js {ver_raw} found but >= 18 is recommended.")
    else:
        ok(f"Node.js {ver_raw}")
    ok(f"npm  {npm}")
    return True


def _detect_cuda() -> str | None:
    """
    Try to detect the CUDA version available on this machine.
    Returns a PyTorch extra-index URL string, or None for CPU-only.
    """
    # 1. Ask nvidia-smi
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        r = subprocess.run([nvsmi], capture_output=True, text=True, check=False)
        if r.returncode == 0:
            # Parse "CUDA Version: 12.4" from the header
            for line in r.stdout.splitlines():
                if "CUDA Version" in line:
                    try:
                        raw = line.strip().split("CUDA Version:")[-1].strip().split()[0]
                        major, minor = int(raw.split(".")[0]), int(raw.split(".")[1])
                        # Map to nearest PyTorch CUDA wheel
                        if (major, minor) >= (12, 4):
                            return "cu124"
                        if (major, minor) >= (12, 1):
                            return "cu121"
                        if (major, minor) >= (11, 8):
                            return "cu118"
                    except (IndexError, ValueError):
                        pass

    # 2. Check if nvcc is on PATH
    if shutil.which("nvcc"):
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False)
        for line in r.stdout.splitlines():
            if "release" in line.lower():
                try:
                    raw = line.split("release")[-1].strip().split(",")[0].strip()
                    major, minor = int(raw.split(".")[0]), int(raw.split(".")[1])
                    if (major, minor) >= (12, 4):
                        return "cu124"
                    if (major, minor) >= (12, 1):
                        return "cu121"
                    if (major, minor) >= (11, 8):
                        return "cu118"
                except (IndexError, ValueError):
                    pass

    return None


def _create_venv() -> None:
    step("Creating virtual environment")
    if VENV_DIR.exists():
        ok(f"venv already exists at {VENV_DIR} — skipping creation")
        return
    _run([sys.executable, "-m", "venv", str(VENV_DIR)])
    ok(f"venv created at {VENV_DIR}")


def _upgrade_pip() -> None:
    step("Upgrading pip / setuptools / wheel")
    _run_venv(["install", "--upgrade", "pip", "setuptools", "wheel"])
    ok("pip upgraded")


def _install_torch(cuda_tag: str | None, *, force_cpu: bool) -> None:
    step("Installing PyTorch")

    if force_cpu or cuda_tag is None:
        if force_cpu:
            info("--cpu flag set — installing CPU-only PyTorch")
        else:
            warn("No CUDA detected — installing CPU-only PyTorch")
            warn("For GPU support install CUDA drivers and re-run without --cpu")
        _run_venv([
            "install", "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ])
        ok("PyTorch (CPU) installed")
    else:
        info(f"CUDA {cuda_tag} detected — installing GPU PyTorch")
        _run_venv([
            "install", "torch", "torchvision", "torchaudio",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}",
        ])
        ok(f"PyTorch ({cuda_tag}) installed")


def _install_python_deps() -> None:
    step("Installing Python backend dependencies")
    packages = [
        # Web server
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
        # Data
        "numpy",
        "pandas",
        "scipy",
        "h5py",
        "pyyaml",
        # ML
        "scikit-learn",
        "matplotlib",
        # Misc
        "tqdm",
        "einops",
    ]
    _run_venv(["install"] + packages)
    ok("Core Python dependencies installed")


def _install_llm(cuda_tag: str | None, *, force_cpu: bool) -> None:
    step("Installing llama-cpp-python (agent-based HP tuning)")
    info("This may take several minutes — it compiles from source.")

    env = dict(os.environ)

    if not force_cpu and cuda_tag is not None:
        info("Building with CUDA support (CMAKE_ARGS=-DGGML_CUDA=on)")
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        env["FORCE_CMAKE"] = "1"
    else:
        info("Building CPU-only llama-cpp-python")

    pip = _venv_bin("pip")
    r = subprocess.run(
        [str(pip), "install", "llama-cpp-python"],
        cwd=ROOT_DIR,
        env=env,
        check=False,
    )
    if r.returncode == 0:
        ok("llama-cpp-python installed")
    else:
        warn("llama-cpp-python build failed — agent-based HP tuning will fall back to random search.")
        warn("You can try again manually:")
        if cuda_tag:
            warn('  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python')
        else:
            warn("  pip install llama-cpp-python")


def _install_npm_deps() -> None:
    step("Installing frontend npm dependencies")
    npm = shutil.which("npm")
    if not npm:
        warn("npm not found — skipping frontend install.")
        return
    if (FRONTEND / "node_modules").exists():
        ok("node_modules already present — skipping (delete it to force reinstall)")
        return
    _run([npm, "install"], cwd=FRONTEND)
    ok("npm dependencies installed")


def _create_backend_config() -> None:
    step("Backend config (backend_config.yaml)")
    if CONFIG_DST.exists():
        ok(f"Config already exists at {CONFIG_DST} — skipping")
        return

    template = textwrap.dedent("""\
        llm:
          # Path to your GGUF model file (used by the agent-based HP tuner).
          # Download a model from e.g. https://huggingface.co/bartowski and set
          # the path here.  Leave blank to disable LLM-based HP suggestions
          # (the tuner will fall back to random search).
          model_path: ""

          # Context window size in tokens
          n_ctx: 16384

          # Number of CPU threads for LLM inference
          n_threads: 8

          # GPU layers to offload:
          #   0   = CPU-only
          #  -1   = all layers (recommended for dedicated GPU with enough VRAM)
          #   N   = N layers on GPU, rest on CPU
          # Rough guide: 25-35 for 8 GB VRAM, 35-45 for 16 GB VRAM
          n_gpu_layers: 0

          # Which GPU to use (0-indexed, for multi-GPU systems)
          main_gpu: 0
    """)

    CONFIG_DST.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_DST.write_text(template, encoding="utf-8")
    ok(f"Default config written to {CONFIG_DST}")
    info("Edit the 'model_path' field to point to a GGUF model for LLM-based HP tuning.")


def _print_summary(node_ok: bool) -> None:
    print()
    print(_c("═" * 60, "1;32"))
    print(_c("  ✔  SurroMod installation complete!", "1;32"))
    print(_c("═" * 60, "1;32"))
    print()
    print("  Next steps:")
    print()
    if platform.system() == "Windows":
        print(f"    1.  Activate the venv:   venv\\Scripts\\activate")
    else:
        print(f"    1.  Activate the venv:   source venv/bin/activate")
    print( "    2.  Start SurroMod:      python launcher.py")
    print()
    if not node_ok:
        warn("Remember to install Node.js >= 18 and run  python scripts/installation.py --skip-llm")
        print()
    print("  Optional:")
    print("    · Edit  src/backend/backend_config.yaml  to set a GGUF model path")
    print("      for agent-based HP tuning.")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SurroMod installer — sets up venv + all dependencies",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only PyTorch (skip CUDA detection)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip llama-cpp-python installation (agent HP tuning will use random fallback)",
    )
    parser.add_argument(
        "--skip-npm",
        action="store_true",
        help="Skip frontend npm install",
    )
    args = parser.parse_args()

    print()
    print(_c("═" * 60, "1;34"))
    print(_c("  SurroMod — Installation", "1;34"))
    print(_c("═" * 60, "1;34"))
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  Root     : {ROOT_DIR}")
    print()

    _check_python()
    node_ok = _check_node()

    cuda_tag = None if args.cpu else _detect_cuda()
    if not args.cpu:
        step("CUDA detection")
        if cuda_tag:
            ok(f"CUDA detected → will install PyTorch with {cuda_tag}")
        else:
            warn("No CUDA detected → CPU-only PyTorch")

    _create_venv()
    _upgrade_pip()
    _install_torch(cuda_tag, force_cpu=args.cpu)
    _install_python_deps()

    if not args.skip_llm:
        _install_llm(cuda_tag, force_cpu=args.cpu)
    else:
        info("--skip-llm set — skipping llama-cpp-python")

    if not args.skip_npm and node_ok:
        _install_npm_deps()
    elif args.skip_npm:
        info("--skip-npm set — skipping npm install")

    _create_backend_config()
    _print_summary(node_ok)


if __name__ == "__main__":
    main()
