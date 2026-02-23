import { type Node, type Edge } from '@xyflow/react';

// ─── Node Categories ────────────────────────────────────────────────────────

/**
 * Each InputKind maps 1-to-1 to a backend data digester:
 *   scalar       → ScalarDataDigester   (flat CSV or HDF5 scalar datasets)
 *   time_series  → TimeSeriesDigester   (sequential / temporal data)
 *   2d_field     → TwoDFieldDigester    (images, 2-D spatial fields)
 *   3d_field     → ThreeDFieldDigester  (volumetric / 3-D fields)
 *   2d_geometry  → Geometry2DDigester   (2-D shape representations)
 *   3d_geometry  → Geometry3DDigester   (3-D shape representations)
 */
export type InputKind = 'scalar' | 'time_series' | '2d_field' | '3d_field' | '2d_geometry' | '3d_geometry';

export type NodeCategory =
  | 'input'
  | 'regressor'
  | 'classifier'
  | 'validator'
  | 'feature_engineering'
  | 'inference'
  | 'hp_tuner';

// ─── Model types ────────────────────────────────────────────────────────────

export type RegressorModel =
  | 'MLP'
  | 'LSTM'
  | 'CNN'
  | 'KRR'
  | 'Polynomial'
  | 'NeuralOperator'
  | 'PINN';

export type ClassifierModel =
  | 'RandomForest'
  | 'SVM'
  | 'DecisionTree'
  | 'KNN'
  | 'GradientBoosting'
  | 'LogisticRegression';

export type FeatureEngineeringMethod = 'PCA' | 'GeometrySampler' | 'Scaler' | 'DataSplitter' | 'Autoencoder' | 'TrainTestSplit';

export type ScalerType = 'MinMax' | 'Standard' | 'LogTransform';

export type HPTunerMethod = 'GridSearch' | 'AgentBased' | 'OptimiserBased';

/** How the Data Splitter decomposes multi-dimensional data. */
export type SplitMode =
  | 'channel'        // (N,C,H,W) → C × (N,H,W)
  | 'channel_x'      // (N,C,H,W) → C·W × (N,H)   — channel × x-slices
  | 'channel_y'      // (N,C,H,W) → C·H × (N,W)   — channel × y-slices
  | 'x'              // (N,C,H,W) → W × (N,C,H)    — x-slices only
  | 'y';             // (N,C,H,W) → H × (N,C,W)    — y-slices only

// ─── Hyperparameters ────────────────────────────────────────────────────────

export type HyperParams = Record<string, string | number | boolean>;

// ─── Per-node data payloads ─────────────────────────────────────────────────

export interface InputNodeData extends Record<string, unknown> {
  label: string;
  category: 'input';
  inputKind: InputKind;
  /** Upload file ID (uuid-based filename stored on the server) or legacy path */
  source: string;
  /** Original filename shown to the user (e.g. "concrete_data.csv") */
  fileName: string;
  /** Available column names (populated after CSV path is set) */
  columns: string[];
  /** Column names selected as features (X) */
  features: string[];
  /** Column names selected as labels (y) */
  labels: string[];
  /**
   * Full file structure returned by /api/data/structure.
   * For CSV: { format: 'csv', columns: [...] }
   * For H5:  { format: 'h5', groups: { ... } }
   */
  structure?: import('./api').DataStructure;
}

export type RegressorRole = 'transform' | 'final';

export interface RegressorNodeData extends Record<string, unknown> {
  label: string;
  category: 'regressor';
  model: RegressorModel;
  hyperparams: HyperParams;
  /** Whether this regressor transforms data (pass-through) or is the final predictor. */
  role: RegressorRole;
}

export interface ClassifierNodeData extends Record<string, unknown> {
  label: string;
  category: 'classifier';
  model: ClassifierModel;
  hyperparams: HyperParams;
}

export interface ValidatorNodeData extends Record<string, unknown> {
  label: string;
  category: 'validator';
  validatorKind: 'classifier_validator' | 'regressor_validator' | 'relation_seeker';
  /** Max number of true-vs-predicted plots per row (configurable layout) */
  plotsPerRow: number;
}

export interface FeatureEngineeringNodeData extends Record<string, unknown> {
  label: string;
  category: 'feature_engineering';
  method: FeatureEngineeringMethod;
  hyperparams: HyperParams;
}

export interface InferenceNodeData extends Record<string, unknown> {
  label: string;
  category: 'inference';
  modelSource: string;
  batchSize: number;
}

export interface HPTunerNodeData extends Record<string, unknown> {
  label: string;
  category: 'hp_tuner';
  method: HPTunerMethod;
  hyperparams: HyperParams;
}

export interface RBLNodeData extends Record<string, unknown> {
  label: string;
  category: 'rbl';
  lambda_kernel: number;
  lambda_residual: number;
}

export interface RBLAggregatorNodeData extends Record<string, unknown> {
  label: string;
  category: 'rbl_aggregator';
}

export type SurroNodeData =
  | InputNodeData
  | RegressorNodeData
  | ClassifierNodeData
  | ValidatorNodeData
  | FeatureEngineeringNodeData
  | InferenceNodeData
  | HPTunerNodeData
  | RBLNodeData
  | RBLAggregatorNodeData;

// ─── Typed aliases for React Flow ───────────────────────────────────────────

export type SurroNode = Node<SurroNodeData>;
export type SurroEdge = Edge;

// ─── Sidebar palette item ──────────────────────────────────────────────────

export interface PaletteItem {
  category: NodeCategory;
  label: string;
  defaultData: SurroNodeData;
}

// ─── Tabs ───────────────────────────────────────────────────────────────────

export interface Tab {
  id: string;
  name: string;
  nodes: SurroNode[];
  edges: SurroEdge[];
}

// ─── Settings ───────────────────────────────────────────────────────────────

export type Theme = 'dark' | 'light';

// ─── Pipeline run results ───────────────────────────────────────────────────

export interface ValidatorLabelResult {
  label: string;
  metrics: Record<string, number>;
  /** base64-encoded PNG of the true-vs-predicted plot */
  plot: string;
}

/** Result shape for a single-model run (backward compat) */
export interface ValidatorResult {
  metrics: Record<string, number>;
  per_label: ValidatorLabelResult[];
  /** Holdout evaluation results (present when a TrainTestSplit was used) */
  holdout?: {
    metrics: Record<string, number>;
    per_label: ValidatorLabelResult[];
  };
}

/** Per-model result entry in a multi-model comparison */
export interface MultiModelEntry {
  model_name: string;
  metrics: Record<string, number>;
  per_label: ValidatorLabelResult[];
}

/** Result shape for a multi-model comparison run */
export interface MultiModelValidatorResult {
  multi_model: true;
  plots_per_row: number;
  model_results: MultiModelEntry[];
  /** base64-encoded grouped bar chart comparing metrics */
  comparison_bar_plot: string;
  /** Holdout evaluation results (present when a TrainTestSplit was used) */
  holdout?: {
    model_results: MultiModelEntry[];
    comparison_bar_plot: string;
  };
}

/** Per-node result from a pipeline run */
export type NodeResult =
  | { status: string }
  | { metrics?: Record<string, number>; is_trained?: boolean }
  | ValidatorResult
  | MultiModelValidatorResult;
