"""
SurroMod Backend Server
=======================
FastAPI-based REST server that exposes pipeline execution and helper
endpoints consumed by the Vite frontend.

Endpoints:
    POST /api/pipeline/run      – execute a pipeline (nodes + edges)
    POST /api/data/structure     – introspect a data file (CSV or HDF5)
    POST /api/csv/columns       – (compat) column names from a CSV file
    GET  /api/logs/stream       – SSE stream of backend log messages
    GET  /api/health             – health check
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import queue
import shutil
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

# ── Ensure project root is importable ─────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent  # → SurroMod/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend.pipeline_executor import run_pipeline  # noqa: E402

# ── Upload directory ──────────────────────────────────────────────────────

UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

WORKFLOWS_DIR = ROOT / "workflows"
WORKFLOWS_DIR.mkdir(exist_ok=True)

# ── Log broadcasting infrastructure ──────────────────────────────────────

class _LogBroadcaster(logging.Handler):
    """
    Logging handler that pushes formatted log records into an
    async-safe queue so SSE clients can consume them in real time.
    """

    def __init__(self) -> None:
        super().__init__()
        self._subscribers: list[asyncio.Queue[str]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def subscribe(self) -> asyncio.Queue[str]:
        """Create and return a new subscriber queue."""
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=500)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        """Remove a subscriber queue."""
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        dead: list[asyncio.Queue[str]] = []
        for q in self._subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                # Drop oldest message to make room
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
            except Exception:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)

log_broadcaster = _LogBroadcaster()
log_broadcaster.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
)

# ── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# Attach the broadcaster to the root logger so ALL log messages are captured
logging.getLogger().addHandler(log_broadcaster)

logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(title="SurroMod Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────

class PipelineRequest(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    seed: int | None = None


class CSVColumnsRequest(BaseModel):
    path: str


class StructureRequest(BaseModel):
    path: str


# ── Helpers ───────────────────────────────────────────────────────────────

def resolve_upload_path(source: str) -> Path:
    """
    Resolve a source string to an actual file path.
    Accepts either an upload ID (uuid-style filename in uploads/) or a raw path.
    """
    # Check if it's an uploaded file ID (lives in UPLOAD_DIR)
    candidate = UPLOAD_DIR / source
    if candidate.exists():
        return candidate
    # Fall back to treating as a raw filesystem path
    return Path(source)


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/pipeline/run")
def pipeline_run(req: PipelineRequest) -> dict[str, Any]:
    """Execute a pipeline graph and return per-node results."""
    try:
        result = run_pipeline(req.nodes, req.edges, seed=req.seed)
        return {"ok": True, **result}
    except Exception as exc:
        logger.error("Pipeline error:\n%s", traceback.format_exc())
        return {"ok": False, "error": str(exc)}


@app.post("/api/data/structure")
def data_structure(req: StructureRequest) -> dict[str, Any]:
    """
    Return the structure of a data file (CSV or HDF5).

    CSV  → ``{"format": "csv", "columns": [...]}``
    HDF5 → ``{"format": "h5", "groups": {...}}``
    """
    try:
        resolved = resolve_upload_path(req.path)
        from src.backend.data_digester import DataDigester

        fmt = DataDigester.detect_format(str(resolved))

        if fmt == "h5":
            from src.backend.data_digester.utils.h5_loader import H5Loader
            struct = H5Loader.read_structure(str(resolved))
        else:
            from src.backend.data_digester.scalar_data_digester import ScalarDataDigester
            struct = ScalarDataDigester.read_structure(str(resolved))

        return {"ok": True, "structure": struct}
    except Exception as exc:
        logger.error("Data structure error: %s", exc)
        return {"ok": False, "error": str(exc)}


@app.post("/api/csv/columns")
def csv_columns(req: CSVColumnsRequest) -> dict[str, Any]:
    """Backward-compatible endpoint: return column names from a CSV file."""
    try:
        resolved = resolve_upload_path(req.path)
        from src.backend.data_digester.scalar_data_digester import ScalarDataDigester

        struct = ScalarDataDigester.read_structure(str(resolved))
        columns = struct.get("columns", [])
        return {"ok": True, "columns": columns}
    except Exception as exc:
        logger.error("CSV columns error: %s", exc)
        return {"ok": False, "error": str(exc)}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Accept a file upload (e.g. CSV), store it in the uploads/ directory,
    and return the file ID and detected columns.
    """
    try:
        # Generate a unique filename preserving the original extension
        original = file.filename or "uploaded_file"
        suffix = Path(original).suffix or ".csv"
        file_id = f"{uuid.uuid4().hex}{suffix}"
        dest = UPLOAD_DIR / file_id

        # Stream file to disk
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info("Uploaded file saved: %s → %s", original, dest)

        # Introspect file structure (best-effort)
        structure: dict[str, Any] = {}
        columns: list[str] = []
        try:
            from src.backend.data_digester import DataDigester
            fmt = DataDigester.detect_format(str(dest))
            if fmt == "h5":
                from src.backend.data_digester.utils.h5_loader import H5Loader
                structure = H5Loader.read_structure(str(dest))
            else:
                import pandas as pd
                df = pd.read_csv(dest, nrows=0)
                columns = list(df.columns)
                structure = {"format": "csv", "columns": columns}
        except Exception:
            logger.warning("Could not read structure from uploaded file %s", file_id)

        return {
            "ok": True,
            "fileId": file_id,
            "originalName": original,
            "columns": columns,
            "structure": structure,
        }
    except Exception as exc:
        logger.error("Upload error: %s", exc)
        return {"ok": False, "error": str(exc)}


@app.get("/api/logs/stream")
async def logs_stream():
    """
    Server-Sent Events endpoint that streams backend log messages
    to the frontend output panel in real time.
    """
    sub_queue = log_broadcaster.subscribe()

    async def event_generator():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(sub_queue.get(), timeout=30.0)
                    # SSE format: data lines followed by blank line
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    # Send a keep-alive comment to prevent connection timeout
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            log_broadcaster.unsubscribe(sub_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Workflow Save / Load ─────────────────────────────────────────────────

class WorkflowSaveRequest(BaseModel):
    name: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


@app.post("/api/workflow/save")
def workflow_save(req: WorkflowSaveRequest) -> dict[str, Any]:
    """
    Save a workflow (nodes + edges + referenced data files) as a pickle.
    The pickle bundles:
      - nodes, edges (JSON-serialisable dicts)
      - data_files: {fileId: bytes} for every upload referenced by Input nodes
    """
    try:
        # Collect referenced upload files
        data_files: dict[str, bytes] = {}
        for nd in req.nodes:
            nd_data = nd.get("data", nd)
            if nd_data.get("category") == "input":
                source = nd_data.get("source", "")
                if source:
                    fpath = UPLOAD_DIR / source
                    if fpath.exists():
                        data_files[source] = fpath.read_bytes()

        bundle = {
            "name": req.name,
            "nodes": req.nodes,
            "edges": req.edges,
            "data_files": data_files,
        }

        safe_name = "".join(
            c if c.isalnum() or c in "-_ " else "_" for c in req.name
        ).strip() or "workflow"
        file_id = f"{safe_name}_{uuid.uuid4().hex[:8]}.pkl"
        dest = WORKFLOWS_DIR / file_id

        with open(dest, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Workflow saved: %s → %s", req.name, dest)
        return {"ok": True, "fileId": file_id, "path": str(dest)}
    except Exception as exc:
        logger.error("Workflow save error: %s", exc)
        return {"ok": False, "error": str(exc)}


@app.get("/api/workflow/download/{file_id}")
def workflow_download(file_id: str) -> Response:
    """Download a saved workflow pickle file."""
    fpath = WORKFLOWS_DIR / file_id
    if not fpath.exists():
        return Response(content="Not found", status_code=404)
    data = fpath.read_bytes()
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{file_id}"'},
    )


@app.post("/api/workflow/load")
async def workflow_load(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Load a workflow from an uploaded pickle file.
    Restores referenced data files into uploads/ and returns nodes + edges.
    """
    try:
        raw = await file.read()
        bundle = pickle.loads(raw)

        nodes = bundle.get("nodes", [])
        edges = bundle.get("edges", [])
        name = bundle.get("name", "Loaded Workflow")
        data_files: dict[str, bytes] = bundle.get("data_files", {})

        # Restore data files to uploads/
        for file_id, content in data_files.items():
            dest = UPLOAD_DIR / file_id
            if not dest.exists():
                dest.write_bytes(content)
                logger.info("Restored data file: %s", file_id)

        logger.info(
            "Workflow loaded: '%s' (%d nodes, %d edges, %d data files)",
            name, len(nodes), len(edges), len(data_files),
        )
        return {"ok": True, "name": name, "nodes": nodes, "edges": edges}
    except Exception as exc:
        logger.error("Workflow load error:\n%s", traceback.format_exc())
        return {"ok": False, "error": str(exc)}


@app.get("/api/workflow/list")
def workflow_list() -> dict[str, Any]:
    """List all saved workflow files."""
    try:
        files = sorted(WORKFLOWS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        items = [{"fileId": f.name, "name": f.stem} for f in files]
        return {"ok": True, "workflows": items}
    except Exception as exc:
        logger.error("Workflow list error: %s", exc)
        return {"ok": False, "error": str(exc)}


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
