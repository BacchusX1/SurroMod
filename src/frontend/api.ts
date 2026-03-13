/**
 * api.ts – backend communication helpers.
 */

import type { SurroNode, SurroEdge } from './types';

const API_BASE = 'http://localhost:8000';

/** (stub) Submit the current pipeline to the backend for execution */
export async function submitPipeline(
  _nodes: SurroNode[],
  _edges: SurroEdge[],
): Promise<{ status: string }> {
  console.log('[api] submitPipeline – no backend connected yet', API_BASE);
  return { status: 'offline' };
}

/** (stub) Load a saved pipeline from the backend */
export async function loadPipeline(
  _id: string,
): Promise<{ nodes: SurroNode[]; edges: SurroEdge[] } | null> {
  console.log('[api] loadPipeline – no backend connected yet');
  return null;
}

// ─── File upload ────────────────────────────────────────────────────────────

export interface UploadResult {
  ok: boolean;
  fileId?: string;
  originalName?: string;
  columns?: string[];
  structure?: DataStructure;
  error?: string;
}

/** Structure returned by the /api/data/structure endpoint */
export interface DatasetInfo {
  shape: number[];
  dtype: string;
}

export interface GroupInfo {
  datasets: Record<string, DatasetInfo>;
}

export interface DataStructure {
  format: 'csv' | 'h5';
  /** CSV-only: column names */
  columns?: string[];
  /** H5-only: group → datasets map */
  groups?: Record<string, GroupInfo>;
}

export interface StructureResult {
  ok: boolean;
  structure?: DataStructure;
  error?: string;
}

/**
 * Fetch the structure (columns / datasets) of an uploaded data file.
 * Works for both CSV and HDF5 files.
 */
export async function fetchStructure(source: string): Promise<StructureResult> {
  const res = await fetch('/api/data/structure', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path: source }),
  });
  return res.json();
}

/**
 * Upload a data file (e.g. CSV) to the backend.
 * Returns the upload ID and detected column names.
 */
export async function uploadFile(file: File): Promise<UploadResult> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/api/upload', {
    method: 'POST',
    body: formData,
  });

  return res.json();
}

// ─── Workflow save / load ───────────────────────────────────────────────────

export interface WorkflowSaveResult {
  ok: boolean;
  fileId?: string;
  path?: string;
  error?: string;
}

/**
 * Save the current workflow (nodes, edges, name) to the backend as a pickle.
 * Data files referenced by Input nodes are bundled automatically.
 */
export async function saveWorkflow(
  name: string,
  nodes: { id: string; type?: string; data: Record<string, unknown> }[],
  edges: { source: string; target: string }[],
): Promise<WorkflowSaveResult> {
  const res = await fetch('/api/workflow/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, nodes, edges }),
  });
  return res.json();
}

/**
 * Download a saved workflow pickle by file ID.
 * Returns a Blob the caller can save with a file-save dialog.
 */
export async function downloadWorkflow(fileId: string): Promise<Blob> {
  const res = await fetch(`/api/workflow/download/${encodeURIComponent(fileId)}`);
  return res.blob();
}

export interface WorkflowLoadResult {
  ok: boolean;
  name?: string;
  nodes?: { id: string; type?: string; data: Record<string, unknown> }[];
  edges?: { source: string; target: string }[];
  error?: string;
}

/**
 * Upload a workflow pickle file and restore it.
 */
export async function loadWorkflowFile(file: File): Promise<WorkflowLoadResult> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/api/workflow/load', {
    method: 'POST',
    body: formData,
  });
  return res.json();
}

export interface WorkflowListResult {
  ok: boolean;
  workflows?: { fileId: string; name: string }[];
  error?: string;
}

/**
 * List all saved workflows on the server.
 */
export async function listWorkflows(): Promise<WorkflowListResult> {
  const res = await fetch('/api/workflow/list');
  return res.json();
}

// ─── Agent-Based HP Tuning ──────────────────────────────────────────────────

export interface HPTuningDataInfo {
  n_features: number;
  n_labels: number;
  feature_names: string[];
  label_names: string[];
  input_kind: string;
  file_name: string;
  /** Upload file ID so backend can read the actual file for n_samples / dtypes */
  source: string;
  holdout_ratio?: number;
}

export interface HPTuningRunRequest {
  nodes: any[];
  edges: any[];
  tuner_node_id: string;
  predictor_node_id: string;
  selected_params: {
    key: string;
    type: string;
    currentValue: string | number | boolean;
    min?: number;
    max?: number;
    step?: number;
    options?: string[];
    discreteValues?: (number | string)[];
  }[];
  n_iterations: number;
  exploration_rate: number;
  scoring_metric: string;
  seed?: number | null;
  data_info?: HPTuningDataInfo;
}

export interface HPTuningRunResult {
  ok: boolean;
  history?: { iteration: number; config: Record<string, any>; score: number; train_score?: number | null; holdout_score?: number | null; n_params?: number | null }[];
  best_config?: Record<string, any>;
  best_score?: number;
  error?: string;
}

/**
 * Run agent-based HP tuning via the local LLM.
 * This is a long-running request — progress is streamed via the SSE log endpoint.
 */
export async function runAgentHPTuning(req: HPTuningRunRequest): Promise<HPTuningRunResult> {
  let res: Response;
  try {
    res = await fetch('/api/hp-tuner/agent/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
  } catch (err) {
    // Network error / connection reset (e.g. backend restart during tuning)
    return { ok: false, error: `Network error: ${err instanceof Error ? err.message : String(err)}` };
  }

  // Try to parse JSON — the response body may be empty/truncated if the
  // backend connection was dropped ("socket hang up").
  let body: any;
  try {
    body = await res.json();
  } catch {
    const status = res.status;
    return {
      ok: false,
      error: status === 502 || status === 504
        ? 'Backend connection lost — the server may have restarted. Please try again.'
        : `Server returned invalid response (HTTP ${status}). Check the Output panel for backend logs.`,
    };
  }

  return body as HPTuningRunResult;
}
