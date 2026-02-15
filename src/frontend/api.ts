/**
 * api.ts – placeholder for future backend communication.
 *
 * Right now the frontend works completely offline.
 * When the backend is ready, functions here will send the pipeline graph
 * (nodes + edges) to a REST / WebSocket endpoint for execution.
 */

import type { SurroNode, SurroEdge } from './types';

const API_BASE = 'http://localhost:8000';

/** (stub) Submit the current pipeline to the backend for execution */
export async function submitPipeline(
  _nodes: SurroNode[],
  _edges: SurroEdge[],
): Promise<{ status: string }> {
  // TODO: wire up when backend is ready
  // return fetch(`${API_BASE}/pipeline`, { method: 'POST', body: JSON.stringify({ nodes, edges }) })
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
