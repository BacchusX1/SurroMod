import type {
  PaletteItem,
  RegressorModel,
  ClassifierModel,
  FeatureEngineeringMethod,
  HPTunerMethod,
  HyperParams,
  RegressorRole,
  PostprocessingKind,
  SurroNodeData,
} from './types';

// ─── Default hyperparameters per regressor model ────────────────────────────

export const regressorDefaults: Record<RegressorModel, HyperParams> = {
  MLP: {
    hidden_layers: 3,
    neurons_per_layer: 64,
    activation: 'ReLU',
    learning_rate: 0.001,
    epochs: 100,
    batch_size: 32,
    output_dim: 0,
    optimizer: 'Adam',
    loss_function: 'MSE',
    weight_init: 'default',
    dropout: 0.0,
    batch_norm: false,
    lr_scheduler: 'none',
    lr_scheduler_step_size: 10,
    lr_scheduler_gamma: 0.1,
    lr_scheduler_patience: 5,
    gradient_clipping: 0.0,
    early_stopping: false,
    early_stopping_patience: 10,
    weight_decay: 0.0,
    layer_sizes: '',
  },
  LSTM: {
    hidden_size: 64,
    num_layers: 2,
    dropout: 0.1,
    learning_rate: 0.001,
    epochs: 100,
    sequence_length: 10,
  },
  CNN: {
    num_conv_layers: 3,
    filters: 32,
    kernel_size: 3,
    learning_rate: 0.001,
    epochs: 100,
    batch_size: 32,
  },
  KRR: {
    kernel: 'rbf',
    alpha: 1.0,
    gamma: 0.1,
    degree: 3,
    output_dim: 0,
  },
  Polynomial: {
    degree: 3,
    interaction_only: false,
    include_bias: true,
  },
  NeuralOperator: {
    modes: 12,
    width: 32,
    depth: 4,
    learning_rate: 0.001,
    epochs: 100,
  },
  PINN: {
    hidden_layers: 4,
    neurons_per_layer: 64,
    physics_loss_weight: 0.1,
    learning_rate: 0.001,
    epochs: 200,
  },
  GraphFlowForecaster: {
    latent_dim: 64,
    hidden_dim: 128,
    num_message_passing_layers: 3,
    aggregation_mode: 'mean',
    skip_connection_mode: 'none',
    dropout: 0.0,
    use_edge_attr: true,
    baseline_mode: 'none',
    learning_rate: 0.001,
    batch_size: 4,
    num_epochs: 100,
    optimizer: 'Adam',
    weight_decay: 0.0001,
    scheduler: 'plateau',
    early_stopping_patience: 15,
    // Hierarchical U-Net + xLSTM HPs
    xlstm_head_dim: 16,
    xlstm_num_layers: 2,
    xlstm_output_dim: 64,
    include_pressure_in_encoder: true,
    num_fine_mp_layers: 2,
    num_coarse_mp_layers: 3,
    proximity_loss_weight: 3.0,
    proximity_sigma: 0.02,
  },
};

// ─── Default hyperparameters per classifier model ───────────────────────────

export const classifierDefaults: Record<ClassifierModel, HyperParams> = {
  RandomForest: {
    n_estimators: 100,
    max_depth: 10,
    min_samples_split: 2,
    min_samples_leaf: 1,
  },
  SVM: {
    kernel: 'rbf',
    C: 1.0,
    gamma: 'scale',
    degree: 3,
  },
  DecisionTree: {
    max_depth: 10,
    min_samples_split: 2,
    criterion: 'gini',
  },
  KNN: {
    n_neighbors: 5,
    weights: 'uniform',
    metric: 'euclidean',
  },
  GradientBoosting: {
    n_estimators: 100,
    learning_rate: 0.1,
    max_depth: 3,
    subsample: 1.0,
  },
  LogisticRegression: {
    C: 1.0,
    penalty: 'l2',
    solver: 'lbfgs',
    max_iter: 100,
  },
};

// ─── Default hyperparameters per feature engineering method ──────────────────

export const featureEngineeringDefaults: Record<FeatureEngineeringMethod, HyperParams> = {
  PCA: {
    n_components: 2,
    whiten: false,
  },
  GeometrySampler: {
    n_points: 100,
    sampling_method: 'uniform',
  },
  Scaler: {
    method: 'MinMax',
  },
  Autoencoder: {
    latent_dim: 16,
    hidden_layers: 2,
    neurons_per_layer: 64,
    activation: 'ReLU',
    learning_rate: 0.001,
    epochs: 100,
    batch_size: 32,
  },
  SpatialGraphBuilder: {
    graph_mode: 'knn',
    k: 16,
    radius: 0.1,
    max_neighbors: 32,
    include_relative_displacement: true,
    include_distance: true,
    self_loops: false,
  },
  SurfaceDistanceFeature: {
    return_vector: false,
    normalize_distance: false,
  },
  TemporalStackFlatten: {
    flatten_order: 'time_major',
    source_field: 'velocity_in',
    output_field: 'velocity_history_features',
  },
  PointFeatureFusion: {
    include_pos: true,
    include_velocity_history: true,
    include_geometry_mask: true,
    include_dist_to_surface: true,
    include_nearest_surface_vec: false,
    include_pressure: false,
    include_low_freq: false,
  },
  DatasetSplit: {
    data_kind: 'scalar',
    split_mode: 'random',
    train_ratio: 0.7,
    val_ratio: 0.15,
    test_ratio: 0.15,
    random_seed: 42,
    shuffle: true,
  },
  FeatureNormalizer: {
    normalizer_mode: 'standard',
    per_component: false,
    epsilon: 1e-8,
  },
  SpectralDecomposer: {
    spectral_method: 'fft',
    cutoff_freq: 0.2,
    wavelet: 'db4',
    wavelet_levels: 2,
  },
  HierarchicalGraphBuilder: {
    fine_k: 8,
    coarse_ratio: 0.08,
    coarse_k: 8,
    k_unpool: 3,
    normalize_edge_attr: true,
  },
  TemporalXLSTMEncoder: {
    head_dim: 16,
    num_layers: 2,
    output_dim: 64,
    include_pressure: true,
  },
};

// ─── Default hyperparameters per HP tuner method ────────────────────────────

export const hpTunerDefaults: Record<HPTunerMethod, HyperParams> = {
  GridSearch: {
    n_folds: 5,
    scoring_metric: 'r2',
    parallel_jobs: -1,
  },
  AgentBased: {
    n_iterations: 50,
    exploration_rate: 0.1,
    scoring_metric: 'r2',
    metric_source: 'train',
  },
  OptimiserBased: {
    n_trials: 100,
    algorithm: 'tpe',
    scoring_metric: 'r2',
    timeout: 600,
  },
};

export const postprocessingDefaults: Record<PostprocessingKind, HyperParams> = {
  field_slice_plot: {
    slice_plane: 'xy',
    slice_quantile: 0.5,
    field_component: 'magnitude',
    timestep: -1,
    grid_resolution: 100,
  },
  flow_metrics_summary: {
    include_per_timestep: true,
    include_per_component: true,
  },
  prediction_comparison_report: {
    include_training_history: true,
    include_slice_plots: true,
  },
};

// ─── Advanced hyperparameter keys ───────────────────────────────────────────
// Keys listed here will appear in the "Advanced" tab rather than "Basic".

export const advancedKeys: Set<string> = new Set([
  'optimizer',
  'loss_function',
  'weight_init',
  'dropout',
  'batch_norm',
  'lr_scheduler',
  'lr_scheduler_step_size',
  'lr_scheduler_gamma',
  'lr_scheduler_patience',
  'gradient_clipping',
  'early_stopping',
  'early_stopping_patience',
  'weight_decay',
  'layer_sizes',
  'coef0',
  'output_dim',
]);

// ─── Palette items ──────────────────────────────────────────────────────────

const inputKinds: { kind: import('./types').InputKind; label: string }[] = [
  { kind: 'scalar', label: 'Scalar Data' },
  { kind: 'time_series', label: 'Time Series' },
  { kind: '2d_field', label: '2D Field' },
  { kind: '3d_field', label: '3D Field' },
  { kind: '2d_geometry', label: '2D Geometry' },
  { kind: '3d_geometry', label: '3D Geometry' },
];

const inputItems: PaletteItem[] = inputKinds.map(({ kind, label }) => {
  const base: any = {
    label,
    category: 'input' as const,
    inputKind: kind,
    source: '',
    fileName: '',
    columns: [],
    features: [],
    labels: [],
  };

  // Add default hyperparams for specialised 3D formats
  if (kind === '3d_field') {
    base.hyperparams = {
      format_mode: 'Temporal Point Cloud Field',
      field_select_velocity_in: true,
      field_select_velocity_out: true,
      field_select_pos: true,
      field_select_t: true,
      batch_dir: '',
      max_files: 0,
      max_points: 0,
      geometry_filter: '',
    };
  } else if (kind === '3d_geometry') {
    base.hyperparams = {
      format_mode: 'Point Cloud Surface Mask',
    };
  }

  return {
    category: 'input' as const,
    label,
    implemented: kind !== 'time_series' && kind !== '3d_geometry',
    defaultData: base,
  };
});

const regressorModels: RegressorModel[] = [
  'MLP',
  'LSTM',
  'CNN',
  'KRR',
  'Polynomial',
  'NeuralOperator',
  'PINN',
  'GraphFlowForecaster',
];

const _regressorStubs = new Set<RegressorModel>(['LSTM', 'CNN', 'Polynomial', 'NeuralOperator', 'PINN']);
const _hpTunerStubs = new Set<HPTunerMethod>(['GridSearch', 'OptimiserBased']);

const regressorItems: PaletteItem[] = regressorModels.map((model) => ({
  category: 'regressor' as const,
  label: model === 'NeuralOperator' ? 'Neural Operator' : model === 'PINN' ? 'Physics Informed' : model === 'GraphFlowForecaster' ? 'Graph Flow Forecaster' : model,
  implemented: !_regressorStubs.has(model),
  defaultData: {
    label: model === 'NeuralOperator' ? 'Neural Operator' : model === 'PINN' ? 'Physics Informed' : model === 'GraphFlowForecaster' ? 'Graph Flow Forecaster' : model,
    category: 'regressor' as const,
    model,
    hyperparams: { ...regressorDefaults[model] },
    role: 'final' as RegressorRole,
  },
}));

const classifierModels: ClassifierModel[] = [
  'RandomForest',
  'SVM',
  'DecisionTree',
  'KNN',
  'GradientBoosting',
  'LogisticRegression',
];

const classifierItems: PaletteItem[] = classifierModels.map((model) => {
  const displayName: Record<ClassifierModel, string> = {
    RandomForest: 'Random Forest',
    SVM: 'SVM',
    DecisionTree: 'Decision Tree',
    KNN: 'KNN',
    GradientBoosting: 'Gradient Boosting',
    LogisticRegression: 'Logistic Regression',
  };
  return {
    category: 'classifier' as const,
    label: displayName[model],
    implemented: false,
    defaultData: {
      label: displayName[model],
      category: 'classifier' as const,
      model,
      hyperparams: { ...classifierDefaults[model] },
    },
  };
});

const featureEngineeringMethods: FeatureEngineeringMethod[] = [
  'PCA',
  'GeometrySampler',
  'Scaler',
  'Autoencoder',
  'SpatialGraphBuilder',
  'SurfaceDistanceFeature',
  'TemporalStackFlatten',
  'PointFeatureFusion',
  'DatasetSplit',
  'FeatureNormalizer',
  'SpectralDecomposer',
  'HierarchicalGraphBuilder',
  'TemporalXLSTMEncoder',
];

const featureEngineeringItems: PaletteItem[] = featureEngineeringMethods.map((method) => {
  const displayName: Record<FeatureEngineeringMethod, string> = {
    PCA: 'PCA',
    GeometrySampler: 'Geometry Sampler',
    Scaler: 'Scaler',
    Autoencoder: 'Autoencoder',
    SpatialGraphBuilder: 'Spatial Graph Builder',
    SurfaceDistanceFeature: 'Surface Distance Feature',
    TemporalStackFlatten: 'Temporal Stack Flatten',
    PointFeatureFusion: 'Point Feature Fusion',
    DatasetSplit: 'Dataset Split',
    FeatureNormalizer: 'Feature Normalizer',
    SpectralDecomposer: 'Spectral Decomposer',
    HierarchicalGraphBuilder: 'Hierarchical Graph Builder',
    TemporalXLSTMEncoder: 'Temporal xLSTM Encoder',
  };
  return {
    category: 'feature_engineering' as const,
    label: displayName[method],
    defaultData: {
      label: displayName[method],
      category: 'feature_engineering' as const,
      method,
      hyperparams: { ...featureEngineeringDefaults[method] },
    },
  };
});

// Validator items appear under Postprocessing in the sidebar
const validatorItems: PaletteItem[] = [
  {
    category: 'postprocessing',
    label: 'Classifier Validator',
    implemented: false,
    defaultData: {
      label: 'Classifier Validator',
      category: 'validator',
      validatorKind: 'classifier_validator',
      plotsPerRow: 4,
    } as unknown as import('./types').SurroNodeData,
  },
  {
    category: 'postprocessing',
    label: 'Regressor Validator',
    defaultData: {
      label: 'Regressor Validator',
      category: 'validator',
      validatorKind: 'regressor_validator',
      plotsPerRow: 4,
    } as unknown as import('./types').SurroNodeData,
  },
  {
    category: 'postprocessing',
    label: 'Relation Seeker',
    implemented: false,
    defaultData: {
      label: 'Relation Seeker',
      category: 'validator',
      validatorKind: 'relation_seeker',
      plotsPerRow: 4,
    } as unknown as import('./types').SurroNodeData,
  },
  {
    category: 'postprocessing',
    label: 'Flow Forecast Validator',
    defaultData: {
      label: 'Flow Forecast Validator',
      category: 'validator',
      validatorKind: 'flow_forecast_validator',
      plotsPerRow: 4,
      hyperparams: {},
    } as unknown as import('./types').SurroNodeData,
  },
];

const inferenceItems: PaletteItem[] = [
  {
    category: 'inference',
    label: 'Inference',
    implemented: false,
    defaultData: {
      label: 'Inference',
      category: 'inference',
      modelSource: '',
      batchSize: 1,
    },
  },
  {
    category: 'inference',
    label: 'Flow Model Inference',
    defaultData: {
      label: 'Flow Model Inference',
      category: 'inference',
      modelSource: '',
      batchSize: 1,
      inferenceKind: 'flow_model_inference',
      hyperparams: {},
    },
  },
];

const hpTunerMethods: HPTunerMethod[] = [
  'GridSearch',
  'AgentBased',
  'OptimiserBased',
];

const hpTunerItems: PaletteItem[] = hpTunerMethods.map((method) => {
  const displayName: Record<HPTunerMethod, string> = {
    GridSearch: 'Grid Search',
    AgentBased: 'Agent Based',
    OptimiserBased: 'Optimiser Based',
  };
  return {
    category: 'hp_tuner' as const,
    label: displayName[method],
    implemented: !_hpTunerStubs.has(method),
    defaultData: {
      label: displayName[method],
      category: 'hp_tuner' as const,
      method,
      hyperparams: { ...hpTunerDefaults[method] },
    },
  };
});

const rblItems: PaletteItem[] = [
  {
    category: 'feature_engineering' as const,
    label: 'RBL',
    defaultData: {
      label: 'RBL',
      category: 'rbl' as const,
      lambda_kernel: 1.0,
      lambda_residual: 0.01,
    } as unknown as SurroNodeData,
  },
  {
    category: 'feature_engineering' as const,
    label: 'RBL Aggregator',
    defaultData: {
      label: 'RBL Aggregator',
      category: 'rbl_aggregator' as const,
    } as unknown as SurroNodeData,
  },
];

const postprocessingItems: PaletteItem[] = [
  {
    category: 'postprocessing' as const,
    label: 'Field Slice Plot',
    defaultData: {
      label: 'Field Slice Plot',
      category: 'postprocessing' as const,
      postprocessingKind: 'field_slice_plot' as const,
      hyperparams: { ...postprocessingDefaults.field_slice_plot },
    },
  },
  {
    category: 'postprocessing' as const,
    label: 'Flow Metrics Summary',
    defaultData: {
      label: 'Flow Metrics Summary',
      category: 'postprocessing' as const,
      postprocessingKind: 'flow_metrics_summary' as const,
      hyperparams: { ...postprocessingDefaults.flow_metrics_summary },
    },
  },
  {
    category: 'postprocessing' as const,
    label: 'Prediction Comparison Report',
    defaultData: {
      label: 'Prediction Comparison Report',
      category: 'postprocessing' as const,
      postprocessingKind: 'prediction_comparison_report' as const,
      hyperparams: { ...postprocessingDefaults.prediction_comparison_report },
    },
  },
];

const codeExporterItems: PaletteItem[] = [
  {
    category: 'code_exporter' as const,
    label: 'Code Exporter',
    defaultData: {
      label: 'Code Exporter',
      category: 'code_exporter' as const,
      exportStatus: 'idle',
      outputFilename: 'train.py',
    },
  },
];

const gramExporterItems: PaletteItem[] = [
  {
    category: 'code_exporter' as const,
    label: 'Model Exporter',
    defaultData: {
      label: 'Model Exporter',
      category: 'gram_exporter' as const,
      hyperparams: {
        gram_repo_dir: './gram_repo',
        model_name: 'surromod_gff',
        team_name: 'SurroMod Team',
        create_pr: false,
        auto_push: false,
        github_token: '',
      },
      exportStatus: 'idle',
    },
  },
];

export const paletteItems: PaletteItem[] = [
  ...inputItems,
  ...featureEngineeringItems,
  ...regressorItems,
  ...classifierItems,
  ...rblItems,
  ...hpTunerItems,
  ...validatorItems,
  ...inferenceItems,
  ...postprocessingItems,
  ...codeExporterItems,
  ...gramExporterItems,
];

// ─── ID generators ──────────────────────────────────────────────────────────

let _counter = 0;

export function nextNodeId(): string {
  return `node_${Date.now()}_${_counter++}`;
}

export function nextTabId(): string {
  return `tab_${Date.now()}_${_counter++}`;
}

// ─── Category → colour mapping ─────────────────────────────────────────────

export const categoryColor: Record<string, string> = {
  input: '#4ade80',
  regressor: '#60a5fa',
  classifier: '#f472b6',
  validator: '#facc15',
  feature_engineering: '#c084fc',
  inference: '#fb923c',
  hp_tuner: '#2dd4bf',
  rbl: '#f97316',
  rbl_aggregator: '#ef4444',
  postprocessing: '#38bdf8',
  code_exporter: '#34d399',
  gram_exporter: '#34d399',
};
