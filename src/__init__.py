"""
SurroMod package root.

Provides the canonical ``PROJECT_ROOT`` path and a shared
``resolve_upload_path`` helper so that every module resolves
upload IDs the same way.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent  # → SurroMod/

UPLOADS_DIR: Path = PROJECT_ROOT / "uploads"


def resolve_upload_path(source: str) -> Path:
    """
    Resolve a source string to an actual file path.

    Checks ``uploads/`` first (for UUID-style upload IDs),
    then falls back to treating *source* as a raw filesystem path.
    """
    candidate = UPLOADS_DIR / source
    if candidate.exists():
        return candidate
    return Path(source)
