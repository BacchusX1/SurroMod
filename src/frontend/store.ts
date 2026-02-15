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
import type { SurroNode, SurroEdge, SurroNodeData } from './types';
import { nextNodeId } from './utils';

// ─── Store shape ────────────────────────────────────────────────────────────

interface StoreState {
  nodes: SurroNode[];
  edges: SurroEdge[];
  selectedNodeId: string | null;

  onNodesChange: OnNodesChange<SurroNode>;
  onEdgesChange: OnEdgesChange<SurroEdge>;
  onConnect: OnConnect;

  addNode: (data: SurroNodeData, position: { x: number; y: number }) => void;
  setSelectedNode: (id: string | null) => void;
  updateNodeData: (id: string, partial: Partial<SurroNodeData>) => void;
  deleteNode: (id: string) => void;
  deleteEdge: (id: string) => void;
}

// ─── Zustand store ──────────────────────────────────────────────────────────

const useStore = create<StoreState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,

  onNodesChange: (changes: NodeChange<SurroNode>[]) => {
    set({ nodes: applyNodeChanges(changes, get().nodes) });
  },

  onEdgesChange: (changes: EdgeChange<SurroEdge>[]) => {
    set({ edges: applyEdgeChanges(changes, get().edges) });
  },

  onConnect: (connection: Connection) => {
    set({ edges: addEdge({ ...connection, animated: true }, get().edges) });
  },

  addNode: (data: SurroNodeData, position: { x: number; y: number }) => {
    const id = nextNodeId();
    const newNode: SurroNode = {
      id,
      type: data.category, // maps to our registered node types
      position,
      data,
    };
    set({ nodes: [...get().nodes, newNode] });
  },

  setSelectedNode: (id: string | null) => {
    set({ selectedNodeId: id });
  },

  updateNodeData: (id: string, partial: Partial<SurroNodeData>) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, ...partial } } : n,
      ),
    });
  },

  deleteNode: (id: string) => {
    set({
      nodes: get().nodes.filter((n) => n.id !== id),
      edges: get().edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: get().selectedNodeId === id ? null : get().selectedNodeId,
    });
  },

  deleteEdge: (id: string) => {
    set({ edges: get().edges.filter((e) => e.id !== id) });
  },
}));

export default useStore;
