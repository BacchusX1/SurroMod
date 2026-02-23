import { create } from 'zustand';
import {
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  type Connection,
  type NodeChange,
  type EdgeChange,
} from '@xyflow/react';
import type { SurroNode, SurroEdge, SurroNodeData, Tab, Theme, NodeResult } from './types';
import { nextNodeId, nextTabId } from './utils';
import { saveWorkflow, loadWorkflowFile, downloadWorkflow } from './api';

// ─── Clipboard types ────────────────────────────────────────────────────────

interface Clipboard {
  nodes: SurroNode[];
  edges: SurroEdge[];
}

// ─── Undo / Redo History ────────────────────────────────────────────────────

const HISTORY_LIMIT = 100;

interface HistoryEntry {
  nodes: SurroNode[];
  edges: SurroEdge[];
}

// Managed outside Zustand state to avoid triggering re-renders on every push.
const _undoStack: HistoryEntry[] = [];
const _redoStack: HistoryEntry[] = [];

/** Whether a node drag is currently in progress (to avoid flooding the stack). */
let _isDragging = false;

/** Deep-clone the current canvas state into a history entry. */
function _snapshot(state: { nodes: SurroNode[]; edges: SurroEdge[] }): HistoryEntry {
  return {
    nodes: state.nodes.map((n) => ({ ...n, data: { ...n.data } })),
    edges: state.edges.map((e) => ({ ...e })),
  };
}

/** Push the current canvas state onto the undo stack and clear redo. */
function _pushHistory(state: { nodes: SurroNode[]; edges: SurroEdge[] }): void {
  _undoStack.push(_snapshot(state));
  if (_undoStack.length > HISTORY_LIMIT) _undoStack.shift();
  _redoStack.length = 0; // any new action invalidates the redo branch
}

// ─── Store shape ────────────────────────────────────────────────────────────

interface StoreState {
  // Tabs
  tabs: Tab[];
  activeTabId: string;

  // Active tab state (mirrored for convenience)
  nodes: SurroNode[];
  edges: SurroEdge[];
  selectedNodeId: string | null;

  // Settings
  theme: Theme;
  settingsOpen: boolean;
  inspectorWidth: number;
  globalSeed: number | null;

  // Pipeline execution
  pipelineRunning: boolean;
  pipelineError: string | null;
  nodeResults: Record<string, NodeResult>;

  // Output panel
  outputPanelOpen: boolean;
  outputPanelHeight: number;
  logMessages: string[];

  // Clipboard
  clipboard: Clipboard | null;

  // Tab actions
  addTab: () => void;
  removeTab: (id: string) => void;
  setActiveTab: (id: string) => void;
  renameTab: (id: string, name: string) => void;

  // Flow actions
  onNodesChange: OnNodesChange<SurroNode>;
  onEdgesChange: OnEdgesChange<SurroEdge>;
  onConnect: OnConnect;
  addNode: (data: SurroNodeData, position: { x: number; y: number }) => void;
  setSelectedNode: (id: string | null) => void;
  updateNodeData: (id: string, partial: Partial<SurroNodeData>) => void;
  deleteNode: (id: string) => void;
  deleteEdge: (id: string) => void;

  // Copy / Paste
  copySelected: () => void;
  pasteClipboard: () => void;

  // Undo / Redo
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;

  // Workflow save / load
  saveCurrentWorkflow: (name: string) => Promise<void>;
  loadWorkflow: (file: File) => Promise<void>;
  downloadCurrentWorkflow: (name: string) => Promise<void>;

  // Settings actions
  setTheme: (theme: Theme) => void;
  toggleSettings: () => void;
  setInspectorWidth: (width: number) => void;
  setGlobalSeed: (seed: number | null) => void;

  // Pipeline actions
  runPipeline: () => Promise<void>;
  clearResults: () => void;

  // Output panel actions
  toggleOutputPanel: () => void;
  setOutputPanelHeight: (height: number) => void;
  addLogMessage: (msg: string) => void;
  clearLogs: () => void;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

const initialTabId = nextTabId();

/** Persist current nodes/edges back into the tabs array */
function saveTabState(state: StoreState): Tab[] {
  return state.tabs.map((tab) =>
    tab.id === state.activeTabId
      ? { ...tab, nodes: state.nodes, edges: state.edges }
      : tab,
  );
}

// ─── Zustand store ──────────────────────────────────────────────────────────

const useStore = create<StoreState>((set, get) => ({
  tabs: [{ id: initialTabId, name: 'Pipeline 1', nodes: [], edges: [] }],
  activeTabId: initialTabId,
  nodes: [],
  edges: [],
  selectedNodeId: null,

  theme: 'dark',
  settingsOpen: false,
  inspectorWidth: 280,
  globalSeed: null,

  pipelineRunning: false,
  pipelineError: null,
  nodeResults: {},

  outputPanelOpen: false,
  outputPanelHeight: 200,
  logMessages: [],

  clipboard: null,

  // ─── Tab actions ──────────────────────────────────────────────────────────

  addTab: () => {
    const state = get();
    const updatedTabs = saveTabState(state);
    const id = nextTabId();
    const newTab: Tab = {
      id,
      name: `Pipeline ${updatedTabs.length + 1}`,
      nodes: [],
      edges: [],
    };
    set({
      tabs: [...updatedTabs, newTab],
      activeTabId: id,
      nodes: [],
      edges: [],
      selectedNodeId: null,
    });
  },

  removeTab: (id: string) => {
    const state = get();
    if (state.tabs.length <= 1) return;
    const updatedTabs = saveTabState(state).filter((t) => t.id !== id);
    if (state.activeTabId === id) {
      const next = updatedTabs[0];
      set({
        tabs: updatedTabs,
        activeTabId: next.id,
        nodes: next.nodes,
        edges: next.edges,
        selectedNodeId: null,
      });
    } else {
      set({ tabs: updatedTabs });
    }
  },

  setActiveTab: (id: string) => {
    const state = get();
    if (id === state.activeTabId) return;
    const updatedTabs = saveTabState(state);
    const target = updatedTabs.find((t) => t.id === id);
    if (!target) return;
    set({
      tabs: updatedTabs,
      activeTabId: id,
      nodes: target.nodes,
      edges: target.edges,
      selectedNodeId: null,
    });
  },

  renameTab: (id: string, name: string) => {
    set({
      tabs: get().tabs.map((t) => (t.id === id ? { ...t, name } : t)),
    });
  },

  // ─── Flow actions ─────────────────────────────────────────────────────────

  onNodesChange: (changes: NodeChange<SurroNode>[]) => {
    const state = get();

    // Detect drag start → push history once at the beginning of a drag
    const hasPositionDrag = changes.some(
      (c) => c.type === 'position' && 'dragging' in c && c.dragging,
    );
    if (hasPositionDrag && !_isDragging) {
      _isDragging = true;
      _pushHistory(state);
    }
    // Detect drag end
    const dragEnded = changes.some(
      (c) => c.type === 'position' && 'dragging' in c && !c.dragging,
    );
    if (dragEnded) _isDragging = false;

    // Detect removals → push history before applying
    const hasRemove = changes.some((c) => c.type === 'remove');
    if (hasRemove) _pushHistory(state);

    set({ nodes: applyNodeChanges(changes, state.nodes) });
  },

  onEdgesChange: (changes: EdgeChange<SurroEdge>[]) => {
    const state = get();
    const hasRemove = changes.some((c) => c.type === 'remove');
    if (hasRemove) _pushHistory(state);
    set({ edges: applyEdgeChanges(changes, state.edges) });
  },

  onConnect: (connection: Connection) => {
    const state = get();
    _pushHistory(state);
    set({ edges: addEdge({ ...connection, animated: true }, state.edges) });
  },

  addNode: (data: SurroNodeData, position: { x: number; y: number }) => {
    const state = get();
    _pushHistory(state);
    const id = nextNodeId();
    const newNode: SurroNode = {
      id,
      type: data.category,
      position,
      data,
    };
    set({ nodes: [...state.nodes, newNode] });
  },

  setSelectedNode: (id: string | null) => {
    set({ selectedNodeId: id });
  },

  updateNodeData: (id: string, partial: Partial<SurroNodeData>) => {
    const state = get();
    _pushHistory(state);
    set({
      nodes: state.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, ...partial } as SurroNodeData } : n,
      ),
    });
  },

  deleteNode: (id: string) => {
    const state = get();
    _pushHistory(state);
    set({
      nodes: state.nodes.filter((n) => n.id !== id),
      edges: state.edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: state.selectedNodeId === id ? null : state.selectedNodeId,
    });
  },

  deleteEdge: (id: string) => {
    const state = get();
    _pushHistory(state);
    set({ edges: state.edges.filter((e) => e.id !== id) });
  },

  // ─── Copy / Paste ─────────────────────────────────────────────────────────

  copySelected: () => {
    const { nodes, edges } = get();
    const selected = nodes.filter((n) => n.selected);
    if (selected.length === 0) return;

    const selectedIds = new Set(selected.map((n) => n.id));
    // Copy edges that connect only selected nodes
    const selectedEdges = edges.filter(
      (e) => selectedIds.has(e.source) && selectedIds.has(e.target),
    );

    set({
      clipboard: {
        nodes: selected.map((n) => ({ ...n, data: { ...n.data } })),
        edges: selectedEdges.map((e) => ({ ...e })),
      },
    });
  },

  pasteClipboard: () => {
    const state = get();
    const { clipboard, nodes, edges } = state;
    if (!clipboard || clipboard.nodes.length === 0) return;
    _pushHistory(state);

    // Map old node ID → new node ID
    const idMap = new Map<string, string>();
    clipboard.nodes.forEach((n) => {
      idMap.set(n.id, nextNodeId());
    });

    // Offset pasted nodes so they don't fully overlap
    const offsetX = 50;
    const offsetY = 50;

    const newNodes: SurroNode[] = clipboard.nodes.map((n) => ({
      ...n,
      id: idMap.get(n.id)!,
      position: { x: n.position.x + offsetX, y: n.position.y + offsetY },
      selected: true,
      data: { ...n.data, label: `${n.data.label} (copy)` } as SurroNodeData,
    }));

    const newEdges: SurroEdge[] = clipboard.edges.map((e) => ({
      ...e,
      id: `edge_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      source: idMap.get(e.source) ?? e.source,
      target: idMap.get(e.target) ?? e.target,
    }));

    // Deselect existing nodes
    const deselected = nodes.map((n) => ({ ...n, selected: false }));

    set({
      nodes: [...deselected, ...newNodes],
      edges: [...edges, ...newEdges],
      // Update clipboard positions for subsequent pastes
      clipboard: {
        nodes: clipboard.nodes.map((n) => ({
          ...n,
          position: { x: n.position.x + offsetX, y: n.position.y + offsetY },
        })),
        edges: clipboard.edges,
      },
    });
  },

  // ─── Undo / Redo ──────────────────────────────────────────────────────────

  undo: () => {
    if (_undoStack.length === 0) return;
    const state = get();
    // Push current state onto redo
    _redoStack.push(_snapshot(state));
    const prev = _undoStack.pop()!;
    set({ nodes: prev.nodes, edges: prev.edges });
  },

  redo: () => {
    if (_redoStack.length === 0) return;
    const state = get();
    // Push current state onto undo
    _undoStack.push(_snapshot(state));
    const next = _redoStack.pop()!;
    set({ nodes: next.nodes, edges: next.edges });
  },

  canUndo: () => _undoStack.length > 0,
  canRedo: () => _redoStack.length > 0,

  // ─── Workflow save / load ─────────────────────────────────────────────────

  saveCurrentWorkflow: async (name: string) => {
    const state = get();
    const payload = {
      nodes: state.nodes.map((n) => ({ id: n.id, type: n.type, position: n.position, data: n.data })),
      edges: state.edges.map((e) => ({ id: e.id, source: e.source, target: e.target, sourceHandle: e.sourceHandle, targetHandle: e.targetHandle })),
    };
    try {
      const result = await saveWorkflow(name, payload.nodes as any, payload.edges);
      if (result.ok && result.fileId) {
        // Trigger download
        const blob = await downloadWorkflow(result.fileId);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = result.fileId;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } else {
        alert(`Save error: ${result.error}`);
      }
    } catch {
      alert('Cannot reach backend. Is the server running?');
    }
  },

  loadWorkflow: async (file: File) => {
    try {
      const result = await loadWorkflowFile(file);
      if (result.ok && result.nodes && result.edges) {
        const restoredNodes: SurroNode[] = result.nodes.map((n: any) => ({
          id: n.id,
          type: n.type ?? n.data?.category,
          position: n.position ?? { x: Math.random() * 600, y: Math.random() * 400 },
          data: n.data,
        }));
        const restoredEdges: SurroEdge[] = result.edges.map((e: any, i: number) => ({
          id: e.id ?? `edge_restored_${i}`,
          source: e.source,
          target: e.target,
          sourceHandle: e.sourceHandle,
          targetHandle: e.targetHandle,
          animated: true,
        }));

        // Create a new tab for the loaded workflow
        const state = get();
        const updatedTabs = saveTabState(state);
        const tabId = nextTabId();
        const newTab: Tab = {
          id: tabId,
          name: result.name ?? 'Loaded Workflow',
          nodes: restoredNodes,
          edges: restoredEdges,
        };
        set({
          tabs: [...updatedTabs, newTab],
          activeTabId: tabId,
          nodes: restoredNodes,
          edges: restoredEdges,
          selectedNodeId: null,
          nodeResults: {},
          pipelineError: null,
        });
      } else {
        alert(`Load error: ${result.error}`);
      }
    } catch {
      alert('Cannot reach backend. Is the server running?');
    }
  },

  downloadCurrentWorkflow: async (name: string) => {
    // Alias – the save action already triggers download
    await get().saveCurrentWorkflow(name);
  },

  // ─── Settings actions ─────────────────────────────────────────────────────

  setTheme: (theme: Theme) => {
    set({ theme });
    document.documentElement.setAttribute('data-theme', theme);
  },

  toggleSettings: () => {
    set({ settingsOpen: !get().settingsOpen });
  },

  setInspectorWidth: (width: number) => {
    set({ inspectorWidth: Math.max(200, Math.min(window.innerWidth * 0.5, width)) });
  },

  setGlobalSeed: (seed: number | null) => {
    set({ globalSeed: seed });
  },

  // ── Pipeline actions ──────────────────────────────────────────────────

  runPipeline: async () => {
    const state = get();
    set({ pipelineRunning: true, pipelineError: null, nodeResults: {} });

    try {
      const payload = {
        nodes: state.nodes.map((n) => ({ id: n.id, type: n.type, data: n.data })),
        edges: state.edges.map((e) => ({ source: e.source, target: e.target, sourceHandle: e.sourceHandle, targetHandle: e.targetHandle })),
        seed: state.globalSeed,
      };

      const res = await fetch('/api/pipeline/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const json = await res.json();

      if (!json.ok) {
        set({ pipelineRunning: false, pipelineError: json.error ?? 'Unknown error' });
        return;
      }

      set({
        pipelineRunning: false,
        nodeResults: json.node_results ?? {},
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      set({ pipelineRunning: false, pipelineError: msg });
    }
  },

  clearResults: () => {
    set({ nodeResults: {}, pipelineError: null });
  },

  // ── Output panel actions ───────────────────────────────────────────────

  toggleOutputPanel: () => {
    set({ outputPanelOpen: !get().outputPanelOpen });
  },

  setOutputPanelHeight: (height: number) => {
    set({ outputPanelHeight: Math.max(80, Math.min(window.innerHeight * 0.7, height)) });
  },

  addLogMessage: (msg: string) => {
    set((state) => {
      const msgs = [...state.logMessages, msg];
      // Keep at most 2000 lines to avoid memory issues
      if (msgs.length > 2000) msgs.splice(0, msgs.length - 2000);
      return { logMessages: msgs };
    });
  },

  clearLogs: () => {
    set({ logMessages: [] });
  },
}));

export default useStore;
