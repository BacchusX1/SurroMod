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
  | 'hp_tuner'
  | 'postprocessing'
  | 'code_exporter'
  | 'gram_exporter';

// ─── Model types ────────────────────────────────────────────────────────────

export type RegressorModel =
  | 'MLP'
  | 'LSTM'
  | 'CNN'
  | 'KRR'
  | 'Polynomial'
  | 'NeuralOperator'
  | 'PINN'
  | 'GraphFlowForecaster';

export type ClassifierModel =
  | 'RandomForest'
  | 'SVM'
  | 'DecisionTree'
  | 'KNN'
  | 'GradientBoosting'
  | 'LogisticRegression';

export type FeatureEngineeringMethod = 'PCA' | 'GeometrySampler' | 'Scaler' | 'Autoencoder' | 'SpatialGraphBuilder' | 'SurfaceDistanceFeature' | 'TemporalStackFlatten' | 'PointFeatureFusion' | 'DatasetSplit' | 'FeatureNormalizer' | 'SpectralDecomposer' | 'HierarchicalGraphBuilder' | 'TemporalXLSTMEncoder';

export type ScalerType = 'MinMax' | 'Standard' | 'LogTransform';

export type HPTunerMethod = 'GridSearch' | 'AgentBased' | 'OptimiserBased';

/** The data kind / mode for the unified DatasetSplit node. */
export type DatasetSplitDataKind = 'scalar' | '3d_field';

/** Format modes for 3D Field input nodes. */
export type ThreeDFieldFormat = 'Temporal Point Cloud Field';

/** Format modes for 3D Geometry input nodes. */
export type ThreeDGeometryFormat = 'Point Cloud Surface Mask';

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
  /** Hyperparameters for specialised input formats (e.g. format_mode, field toggles). */
  hyperparams?: HyperParams;
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
  validatorKind: 'classifier_validator' | 'regressor_validator' | 'relation_seeker' | 'flow_forecast_validator';
  /** Max number of true-vs-predicted plots per row (configurable layout) */
  plotsPerRow: number;
  /** Hyperparameters for specialised validators (e.g. flow forecast). */
  hyperparams?: HyperParams;
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
  /** Specialised inference kind for flow pipelines. */
  inferenceKind?: 'standard' | 'flow_model_inference';
  hyperparams?: HyperParams;
}

/** Description of a single hyperparameter eligible for agent-based tuning. */
export interface HPTuneParam {
  key: string;
  type: 'number' | 'string' | 'boolean';
  currentValue: string | number | boolean;
  selected: boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  /** Subset of options that the user has enabled for tuning (string params). */
  selectedOptions?: string[];
  /** When true, use discrete values instead of min/max/step range. */
  useDiscreteValues?: boolean;
  /** Semicolon-separated discrete values (parsed into an array for backend). */
  discreteValues?: (number | string)[];
}

/** A single iteration result from agent-based HP tuning. */
export interface HPTuningIterationResult {
  iteration: number;
  config: Record<string, string | number | boolean>;
  score: number;
  /** Train metric value (same as score for backward compat). */
  train_score?: number | null;
  /** Holdout metric value (null when no TrainTestSplit is used). */
  holdout_score?: number | null;
  /** Total trainable model parameters for this config. */
  n_params?: number | null;
}

export interface HPTunerNodeData extends Record<string, unknown> {
  label: string;
  category: 'hp_tuner';
  method: HPTunerMethod;
  hyperparams: HyperParams;
  /** True when predictor has an upstream TrainTestSplit node. */
  hasUpstreamTrainTestSplit?: boolean;
  /** Node ID of the connected predictor (auto-detected from edges). */
  connectedPredictorId?: string;
  /** Hyperparameters loadable from the connected predictor. */
  tunableParams?: HPTuneParam[];
  /** Current tuning status for UI feedback. */
  tuningStatus?: 'idle' | 'loading-hps' | 'running' | 'done' | 'error' | 'stopped';
  /** Full iteration history. */
  tuningResults?: HPTuningIterationResult[];
  /** Best HP config found. */
  bestConfig?: Record<string, string | number | boolean>;
  /** Best metric score found. */
  bestScore?: number;
  /** Error message if tuning failed. */
  tuningError?: string;
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

export type PostprocessingKind =
  | 'field_slice_plot'
  | 'flow_metrics_summary'
  | 'prediction_comparison_report';

export interface PostprocessingNodeData extends Record<string, unknown> {
  label: string;
  category: 'postprocessing';
  postprocessingKind: PostprocessingKind;
  hyperparams: HyperParams;
}

export interface CodeExporterNodeData extends Record<string, unknown> {
  label: string;
  category: 'code_exporter';
  /** Status of the last export attempt */
  exportStatus?: 'idle' | 'exporting' | 'done' | 'error';
  /** Path of the last exported file */
  exportPath?: string;
  /** Error message if export failed */
  exportError?: string;
  /** Optional output filename override */
  outputFilename?: string;
}

export interface GRAMExporterNodeData extends Record<string, unknown> {
  label: string;
  category: 'gram_exporter';
  hyperparams: {
    gram_repo_dir: string;
    model_name: string;
    team_name: string;
    create_pr: boolean;
    auto_push: boolean;
    github_token: string;
  };
  exportStatus?: 'idle' | 'exporting' | 'done' | 'error';
  exportDir?: string;
  prUrl?: string;
  exportError?: string;
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
  | RBLAggregatorNodeData
  | PostprocessingNodeData
  | CodeExporterNodeData
  | GRAMExporterNodeData;

// ─── Typed aliases for React Flow ───────────────────────────────────────────

export type SurroNode = Node<SurroNodeData>;
export type SurroEdge = Edge;

// ─── Sidebar palette item ──────────────────────────────────────────────────

export interface PaletteItem {
  category: NodeCategory;
  label: string;
  defaultData: SurroNodeData;
  /** When false, item is greyed out in the sidebar and cannot be dragged onto the canvas */
  implemented?: boolean;
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
  | MultiModelValidatorResult
  | { tuning_results: HPTuningIterationResult[]; best_config: Record<string, string | number | boolean>; best_score: number };
