#!/usr/bin/env python3
"""
SurroMod Launcher
=================
Starts the frontend dev-server (Vite) from a single Python entry point.

Usage:
    python launcher.py          # install deps (if needed) + start dev server
    python launcher.py --install  # force re-install of npm dependencies
    python launcher.py --build    # create a production build
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "src" / "frontend"
NODE_MODULES = FRONTEND_DIR / "node_modules"
PACKAGE_JSON = FRONTEND_DIR / "package.json"

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _find_npm() -> str:
    """Return the npm executable path, or exit with a helpful message."""
    npm = shutil.which("npm")
    if npm is None:
        print(
            "\n[ERROR] 'npm' not found on PATH.\n"
            "  Please install Node.js (>= 18) from https://nodejs.org\n"
        )
        sys.exit(1)
    return npm


def _run(cmd: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, streaming output to the terminal."""
    print(f"\n  → {' '.join(cmd)}  (in {cwd})\n")
    return subprocess.run(cmd, cwd=cwd, check=check)


def _run_background(cmd: list[str], cwd: Path) -> subprocess.Popen:
    """Start a long-running process in the background."""
    print(f"\n  → {' '.join(cmd)}  (in {cwd})\n")
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


# ─── Actions ─────────────────────────────────────────────────────────────────


def install_deps(npm: str, *, force: bool = False) -> None:
    """Install npm dependencies if node_modules is missing or --install flag set."""
    if not PACKAGE_JSON.exists():
        print(f"[ERROR] package.json not found at {PACKAGE_JSON}")
        sys.exit(1)

    if NODE_MODULES.exists() and not force:
        print("  ✔ node_modules already present – skipping install (use --install to force)")
        return

    print("  ⏳ Installing npm dependencies …")
    _run([npm, "install"], cwd=FRONTEND_DIR)
    print("  ✔ Dependencies installed.")


def start_dev_server(npm: str) -> None:
    """Start the Vite dev-server and open the browser."""
    print("\n🚀 Starting SurroMod frontend dev-server …\n")

    proc = _run_background([npm, "run", "dev", "--", "--host"], cwd=FRONTEND_DIR)

    # Read output lines until we see the local URL, then open the browser
    url: str | None = None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            if "Local:" in line and url is None:
                # Vite prints something like "  ➜  Local:   http://localhost:5173/"
                parts = line.strip().split()
                for part in parts:
                    if part.startswith("http"):
                        url = part
                        break
                if url:
                    print(f"\n  🌐 Opening {url} in your browser …\n")
                    webbrowser.open(url)
    except KeyboardInterrupt:
        print("\n\n  ⏹ Shutting down dev-server …")
        proc.terminate()
        proc.wait(timeout=5)
        print("  ✔ Stopped.")
        return

    # If the process ended on its own
    proc.wait()
    if proc.returncode and proc.returncode != 0:
        print(f"\n[ERROR] Dev-server exited with code {proc.returncode}")
        sys.exit(proc.returncode)


def build_production(npm: str) -> None:
    """Create a production build."""
    print("\n📦 Building production bundle …\n")
    _run([npm, "run", "build"], cwd=FRONTEND_DIR)
    print("\n  ✔ Build complete.  Output in src/frontend/dist/")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SurroMod Launcher – start the surrogate-model builder UI",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Force re-install of npm dependencies",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Create a production build instead of starting the dev-server",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ⚙️  SurroMod Launcher")
    print("=" * 60)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Root    : {ROOT_DIR}")

    npm = _find_npm()

    # Check Node.js version
    node_version = subprocess.run(
        ["node", "--version"], capture_output=True, text=True
    ).stdout.strip()
    print(f"  Node.js : {node_version}")
    print(f"  npm     : {npm}")
    print("=" * 60)

    install_deps(npm, force=args.install)

    if args.build:
        build_production(npm)
    else:
        start_dev_server(npm)


if __name__ == "__main__":
    main()
