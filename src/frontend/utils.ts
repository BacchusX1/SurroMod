import type {
  PaletteItem,
  RegressorModel,
  ClassifierModel,
  FeatureEngineeringMethod,
  HPTunerMethod,
  HyperParams,
  RegressorRole,
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
  DataSplitter: {
    split_mode: 'channel',
    n_outputs: 3,
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
  TrainTestSplit: {
    holdout_ratio: 0.2,
    shuffle: true,
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

const inputItems: PaletteItem[] = inputKinds.map(({ kind, label }) => ({
  category: 'input' as const,
  label,
  defaultData: {
    label,
    category: 'input' as const,
    inputKind: kind,
    source: '',
    fileName: '',
    columns: [],
    features: [],
    labels: [],
  },
}));

const regressorModels: RegressorModel[] = [
  'MLP',
  'LSTM',
  'CNN',
  'KRR',
  'Polynomial',
  'NeuralOperator',
  'PINN',
];

const regressorItems: PaletteItem[] = regressorModels.map((model) => ({
  category: 'regressor' as const,
  label: model === 'NeuralOperator' ? 'Neural Operator' : model === 'PINN' ? 'Physics Informed' : model,
  defaultData: {
    label: model === 'NeuralOperator' ? 'Neural Operator' : model === 'PINN' ? 'Physics Informed' : model,
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
  'DataSplitter',
  'Autoencoder',
  'TrainTestSplit',
];

const featureEngineeringItems: PaletteItem[] = featureEngineeringMethods.map((method) => {
  const displayName: Record<FeatureEngineeringMethod, string> = {
    PCA: 'PCA',
    GeometrySampler: 'Geometry Sampler',
    Scaler: 'Scaler',
    DataSplitter: 'Data Splitter',
    Autoencoder: 'Autoencoder',
    TrainTestSplit: 'Train-Test Split',
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

const validatorItems: PaletteItem[] = [
  {
    category: 'validator',
    label: 'Classifier Validator',
    defaultData: {
      label: 'Classifier Validator',
      category: 'validator',
      validatorKind: 'classifier_validator',
      plotsPerRow: 4,
    },
  },
  {
    category: 'validator',
    label: 'Regressor Validator',
    defaultData: {
      label: 'Regressor Validator',
      category: 'validator',
      validatorKind: 'regressor_validator',
      plotsPerRow: 4,
    },
  },
  {
    category: 'validator',
    label: 'Relation Seeker',
    defaultData: {
      label: 'Relation Seeker',
      category: 'validator',
      validatorKind: 'relation_seeker',
      plotsPerRow: 4,
    },
  },
];

const inferenceItems: PaletteItem[] = [
  {
    category: 'inference',
    label: 'Inference',
    defaultData: {
      label: 'Inference',
      category: 'inference',
      modelSource: '',
      batchSize: 1,
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

export const paletteItems: PaletteItem[] = [
  ...inputItems,
  ...featureEngineeringItems,
  ...regressorItems,
  ...classifierItems,
  ...rblItems,
  ...hpTunerItems,
  ...validatorItems,
  ...inferenceItems,
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
};
