import { type Node, type Edge } from '@xyflow/react';

// ─── Node Categories (mirror backend structure) ─────────────────────────────

/** Data type that a node can produce / consume */
export type DataKind = 'scalar' | '2d_field' | '3d_field' | 'time_series' | 'step';

/** The four high-level pipeline stages */
export type NodeCategory = 'input' | 'regressor' | 'classifier' | 'validator';

// ─── Per-node data payloads ─────────────────────────────────────────────────

export interface InputNodeData extends Record<string, unknown> {
  label: string;
  category: 'input';
  dataKind: DataKind;
  /** placeholder: file path or source id selected by the user */
  source: string;
}

export interface RegressorNodeData extends Record<string, unknown> {
  label: string;
  category: 'regressor';
  dataKind: DataKind;
  method: string;
}

export interface ClassifierNodeData extends Record<string, unknown> {
  label: string;
  category: 'classifier';
  dataKind: DataKind;
  method: string;
}

export interface ValidatorNodeData extends Record<string, unknown> {
  label: string;
  category: 'validator';
  validatorKind: 'classifier_validator' | 'regressor_validator' | 'relation_seeker';
}

export type SurroNodeData =
  | InputNodeData
  | RegressorNodeData
  | ClassifierNodeData
  | ValidatorNodeData;

// ─── Typed aliases for React Flow ───────────────────────────────────────────

export type SurroNode = Node<SurroNodeData>;
export type SurroEdge = Edge;

// ─── Sidebar palette item ──────────────────────────────────────────────────

export interface PaletteItem {
  category: NodeCategory;
  label: string;
  defaultData: SurroNodeData;
}
