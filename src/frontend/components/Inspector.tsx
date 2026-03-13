import { useState, useCallback, useRef } from 'react';
import useStore from '../store';
import type {
  SurroNodeData,
  InputNodeData,
  RegressorNodeData,
  ClassifierNodeData,
  ValidatorNodeData,
  FeatureEngineeringNodeData,
  InferenceNodeData,
  HPTunerNodeData,
  RBLNodeData,
  RBLAggregatorNodeData,
  HyperParams,
  HPTuneParam,
  HPTuningIterationResult,
  ValidatorResult,
  MultiModelValidatorResult,
  MultiModelEntry,
} from '../types';
import { categoryColor, advancedKeys } from '../utils';
import { uploadFile, fetchStructure, runAgentHPTuning } from '../api';
import type { DataStructure, HPTuningDataInfo } from '../api';

// ─── Hyperparameter Editor ──────────────────────────────────────────────────

/** Select options for known enum-like hyperparameters */
const selectOptions: Record<string, { label: string; value: string }[]> = {
  activation: [
    { label: 'ReLU', value: 'ReLU' },
    { label: 'Tanh', value: 'Tanh' },
    { label: 'Sigmoid', value: 'Sigmoid' },
    { label: 'GELU', value: 'GELU' },
    { label: 'LeakyReLU', value: 'LeakyReLU' },
  ],
  kernel: [
    { label: 'RBF', value: 'rbf' },
    { label: 'Linear', value: 'linear' },
    { label: 'Polynomial', value: 'poly' },
    { label: 'Sigmoid', value: 'sigmoid' },
  ],
  criterion: [
    { label: 'Gini', value: 'gini' },
    { label: 'Entropy', value: 'entropy' },
    { label: 'Log Loss', value: 'log_loss' },
  ],
  weights: [
    { label: 'Uniform', value: 'uniform' },
    { label: 'Distance', value: 'distance' },
  ],
  metric: [
    { label: 'Euclidean', value: 'euclidean' },
    { label: 'Manhattan', value: 'manhattan' },
    { label: 'Cosine', value: 'cosine' },
  ],
  penalty: [
    { label: 'L1', value: 'l1' },
    { label: 'L2', value: 'l2' },
    { label: 'ElasticNet', value: 'elasticnet' },
    { label: 'None', value: 'none' },
  ],
  solver: [
    { label: 'LBFGS', value: 'lbfgs' },
    { label: 'Liblinear', value: 'liblinear' },
    { label: 'SAGA', value: 'saga' },
    { label: 'Newton-CG', value: 'newton-cg' },
  ],
  sampling_method: [
    { label: 'Uniform', value: 'uniform' },
    { label: 'Cosine', value: 'cosine' },
    { label: 'Curvature Based', value: 'curvature_based' },
  ],
  method: [
    { label: 'MinMax', value: 'MinMax' },
    { label: 'Standard Scaler', value: 'Standard' },
    { label: 'Log Transform', value: 'LogTransform' },
  ],
  optimizer: [
    { label: 'Adam', value: 'Adam' },
    { label: 'AdamW', value: 'AdamW' },
    { label: 'SGD', value: 'SGD' },
    { label: 'RMSprop', value: 'RMSprop' },
    { label: 'LBFGS', value: 'LBFGS' },
  ],
  loss_function: [
    { label: 'MSE', value: 'MSE' },
    { label: 'MAE', value: 'MAE' },
    { label: 'Huber', value: 'Huber' },
    { label: 'Smooth L1', value: 'SmoothL1' },
  ],
  weight_init: [
    { label: 'Default', value: 'default' },
    { label: 'Xavier Uniform', value: 'xavier_uniform' },
    { label: 'Xavier Normal', value: 'xavier_normal' },
    { label: 'Kaiming Uniform', value: 'kaiming_uniform' },
    { label: 'Kaiming Normal', value: 'kaiming_normal' },
    { label: 'Orthogonal', value: 'orthogonal' },
    { label: 'Zeros', value: 'zeros' },
  ],
  lr_scheduler: [
    { label: 'None', value: 'none' },
    { label: 'Step', value: 'step' },
    { label: 'Cosine', value: 'cosine' },
    { label: 'Plateau', value: 'plateau' },
    { label: 'Exponential', value: 'exponential' },
  ],
  scoring_metric: [
    { label: 'R²', value: 'r2' },
    { label: 'RMSE', value: 'rmse' },
    { label: 'MAE', value: 'mae' },
    { label: 'Accuracy', value: 'accuracy' },
    { label: 'F1', value: 'f1' },
  ],
  algorithm: [
    { label: 'TPE', value: 'tpe' },
    { label: 'CMA-ES', value: 'cma_es' },
    { label: 'Random', value: 'random' },
    { label: 'Bayesian', value: 'bayesian' },
  ],
};

function HyperParamsEditor({
  hyperparams,
  onChange,
  filterKeys,
}: {
  hyperparams: HyperParams;
  onChange: (key: string, value: string | number | boolean) => void;
  filterKeys?: (key: string) => boolean;
}) {
  const entries = filterKeys
    ? Object.entries(hyperparams).filter(([key]) => filterKeys(key))
    : Object.entries(hyperparams);

  if (entries.length === 0) {
    return <p className="inspector__empty-tab">No parameters in this tab.</p>;
  }

  return (
    <>
      {entries.map(([key, value]) => {
        const label = key.replace(/_/g, ' ');

        // Boolean → checkbox
        if (typeof value === 'boolean') {
          return (
            <label key={key} className="inspector__field inspector__field--checkbox">
              <span>{label}</span>
              <input
                type="checkbox"
                checked={value}
                onChange={(e) => onChange(key, e.target.checked)}
              />
            </label>
          );
        }

        // Known select fields
        if (selectOptions[key] && typeof value === 'string') {
          return (
            <label key={key} className="inspector__field">
              <span>{label}</span>
              <select value={value} onChange={(e) => onChange(key, e.target.value)}>
                {selectOptions[key].map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </label>
          );
        }

        // Number → number input
        if (typeof value === 'number') {
          return (
            <label key={key} className="inspector__field">
              <span>{label}</span>
              <input
                type="number"
                value={value}
                step={value < 1 && value > 0 ? 0.001 : 1}
                onChange={(e) => onChange(key, parseFloat(e.target.value) || 0)}
              />
            </label>
          );
        }

        // String fallback → text input
        return (
          <label key={key} className="inspector__field">
            <span>{label}</span>
            <input
              type="text"
              value={value as string}
              onChange={(e) => onChange(key, e.target.value)}
            />
          </label>
        );
      })}
    </>
  );
}
// ── Column Picker (features vs labels for Input nodes) ──────────────────────

function ColumnPicker({
  columns,
  features,
  labels,
  onToggle,
}: {
  columns: string[];
  features: string[];
  labels: string[];
  onToggle: (col: string, role: 'feature' | 'label' | 'none') => void;
}) {
  if (columns.length === 0) return null;

  return (
    <div className="inspector__column-picker">
      {columns.map((col) => {
        const role = features.includes(col)
          ? 'feature'
          : labels.includes(col)
          ? 'label'
          : 'none';

        return (
          <div key={col} className="inspector__col-row">
            <span className="inspector__col-name" title={col}>{col}</span>
            <div className="inspector__col-buttons">
              <button
                className={`inspector__col-btn ${role === 'feature' ? 'inspector__col-btn--feature' : ''}`}
                onClick={() => onToggle(col, role === 'feature' ? 'none' : 'feature')}
                title="Feature"
              >F</button>
              <button
                className={`inspector__col-btn ${role === 'label' ? 'inspector__col-btn--label' : ''}`}
                onClick={() => onToggle(col, role === 'label' ? 'none' : 'label')}
                title="Label"
              >L</button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Validator Results Display ────────────────────────────────────────────────

/** Single-model results (backward compat) */
function ValidatorResultsView({ result }: { result: ValidatorResult }) {
  const hasHoldout = !!result.holdout;
  const [view, setView] = useState<'train' | 'holdout'>('train');
  const active = view === 'holdout' && hasHoldout ? result.holdout! : result;

  return (
    <div className="validator-results">
      {hasHoldout && (
        <div className="inspector__tabs" style={{ marginBottom: 8 }}>
          <button
            className={`inspector__tab ${view === 'train' ? 'inspector__tab--active' : ''}`}
            onClick={() => setView('train')}
          >
            Train
          </button>
          <button
            className={`inspector__tab ${view === 'holdout' ? 'inspector__tab--active' : ''}`}
            onClick={() => setView('holdout')}
          >
            Holdout
          </button>
        </div>
      )}

      <div className="inspector__section-title">Overall Metrics</div>
      <div className="validator-results__metrics">
        {Object.entries(active.metrics).map(([key, val]) => (
          <div key={key} className="validator-results__metric">
            <span className="validator-results__metric-key">{key.toUpperCase()}</span>
            <span className="validator-results__metric-val">{(val as number).toFixed(4)}</span>
          </div>
        ))}
      </div>

      {active.per_label.map((pl) => (
        <div key={pl.label} className="validator-results__label-block">
          <div className="inspector__section-title">{pl.label}</div>
          <div className="validator-results__metrics">
            {Object.entries(pl.metrics).map(([key, val]) => (
              <div key={key} className="validator-results__metric">
                <span className="validator-results__metric-key">{key.toUpperCase()}</span>
                <span className="validator-results__metric-val">{(val as number).toFixed(4)}</span>
              </div>
            ))}
          </div>
          {pl.plot && (
            <img
              className="validator-results__plot"
              src={`data:image/png;base64,${pl.plot}`}
              alt={`True vs Predicted – ${pl.label}`}
            />
          )}
        </div>
      ))}
    </div>
  );
}

/** Multi-model comparison results */
function MultiModelResultsView({
  result,
  plotsPerRow,
}: {
  result: MultiModelValidatorResult;
  plotsPerRow: number;
}) {
  const hasHoldout = !!result.holdout;
  const [view, setView] = useState<'train' | 'holdout'>('train');

  const activeResults = view === 'holdout' && hasHoldout
    ? result.holdout!.model_results
    : result.model_results;
  const activeBarPlot = view === 'holdout' && hasHoldout
    ? result.holdout!.comparison_bar_plot
    : result.comparison_bar_plot;

  // Collect all unique label names across models
  const labelNames = activeResults.length > 0
    ? activeResults[0].per_label.map((pl) => pl.label)
    : [];

  return (
    <div className="validator-results">
      {hasHoldout && (
        <div className="inspector__tabs" style={{ marginBottom: 8 }}>
          <button
            className={`inspector__tab ${view === 'train' ? 'inspector__tab--active' : ''}`}
            onClick={() => setView('train')}
          >
            Train
          </button>
          <button
            className={`inspector__tab ${view === 'holdout' ? 'inspector__tab--active' : ''}`}
            onClick={() => setView('holdout')}
          >
            Holdout
          </button>
        </div>
      )}

      {/* ── Comparison bar chart ─────────────────────────────────────── */}
      <div className="inspector__section-title">Metrics Comparison</div>
      {activeBarPlot && (
        <img
          className="validator-results__plot"
          src={`data:image/png;base64,${activeBarPlot}`}
          alt="Metrics Comparison"
        />
      )}

      {/* ── Per-model metrics table ─────────────────────────────────── */}
      <div className="inspector__section-title">Per-Model Metrics</div>
      <div className="validator-results__model-table">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              {Object.keys(activeResults[0]?.metrics ?? {}).map((k) => (
                <th key={k}>{k.toUpperCase()}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {activeResults.map((mr: MultiModelEntry) => (
              <tr key={mr.model_name}>
                <td className="validator-results__model-name">{mr.model_name}</td>
                {Object.values(mr.metrics).map((v, i) => (
                  <td key={i}>{(v as number).toFixed(4)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ── Per-label scatter plot grids ─────────────────────────────── */}
      {labelNames.map((labelName) => (
        <div key={labelName} className="validator-results__label-block">
          <div className="inspector__section-title">
            True vs Predicted – {labelName}
          </div>
          <div
            className="validator-results__scatter-grid"
            style={{
              gridTemplateColumns: `repeat(${Math.min(
                activeResults.length,
                plotsPerRow,
              )}, 1fr)`,
            }}
          >
            {activeResults.map((mr: MultiModelEntry) => {
              const labelResult = mr.per_label.find((pl) => pl.label === labelName);
              if (!labelResult?.plot) return null;
              return (
                <div key={mr.model_name} className="validator-results__scatter-cell">
                  <img
                    className="validator-results__plot"
                    src={`data:image/png;base64,${labelResult.plot}`}
                    alt={`${mr.model_name} – ${labelName}`}
                  />
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Tabbed hyperparameter wrapper ───────────────────────────────────────────

function TabbedHyperParams({
  hyperparams,
  onChange,
}: {
  hyperparams: HyperParams;
  onChange: (key: string, value: string | number | boolean) => void;
}) {
  const hasAdvanced = Object.keys(hyperparams).some((k) => advancedKeys.has(k));
  const [tab, setTab] = useState<'basic' | 'advanced'>('basic');

  if (!hasAdvanced) {
    return <HyperParamsEditor hyperparams={hyperparams} onChange={onChange} />;
  }

  return (
    <>
      <div className="inspector__tabs">
        <button
          className={`inspector__tab ${tab === 'basic' ? 'inspector__tab--active' : ''}`}
          onClick={() => setTab('basic')}
        >
          Basic
        </button>
        <button
          className={`inspector__tab ${tab === 'advanced' ? 'inspector__tab--active' : ''}`}
          onClick={() => setTab('advanced')}
        >
          Advanced
        </button>
      </div>
      <HyperParamsEditor
        hyperparams={hyperparams}
        onChange={onChange}
        filterKeys={
          tab === 'basic'
            ? (k) => !advancedKeys.has(k)
            : (k) => advancedKeys.has(k)
        }
      />
    </>
  );
}

// ─── Agent-Based HP Tuner UI ────────────────────────────────────────────────

/** Default numeric ranges for common hyperparameters */
const defaultNumericRanges: Record<string, { min: number; max: number; step?: number }> = {
  learning_rate: { min: 0.00001, max: 0.1, step: 0.00001 },
  epochs: { min: 10, max: 1000, step: 1 },
  hidden_layers: { min: 1, max: 10, step: 1 },
  neurons_per_layer: { min: 8, max: 512, step: 8 },
  batch_size: { min: 4, max: 512, step: 4 },
  dropout: { min: 0.0, max: 0.8, step: 0.05 },
  alpha: { min: 0.001, max: 100, step: 0.001 },
  gamma: { min: 0.001, max: 10, step: 0.001 },
  degree: { min: 1, max: 10, step: 1 },
  weight_decay: { min: 0, max: 0.1, step: 0.001 },
  C: { min: 0.01, max: 100, step: 0.01 },
  n_estimators: { min: 10, max: 500, step: 10 },
  max_depth: { min: 1, max: 50, step: 1 },
  hidden_size: { min: 8, max: 512, step: 8 },
  num_layers: { min: 1, max: 10, step: 1 },
  gradient_clipping: { min: 0, max: 10, step: 0.1 },
  latent_dim: { min: 2, max: 128, step: 1 },
  n_components: { min: 1, max: 50, step: 1 },
  n_neighbors: { min: 1, max: 50, step: 1 },
  min_samples_split: { min: 2, max: 20, step: 1 },
  min_samples_leaf: { min: 1, max: 20, step: 1 },
  subsample: { min: 0.1, max: 1.0, step: 0.05 },
  max_iter: { min: 50, max: 2000, step: 50 },
  lr_scheduler_step_size: { min: 1, max: 50, step: 1 },
  lr_scheduler_gamma: { min: 0.01, max: 1.0, step: 0.01 },
  lr_scheduler_patience: { min: 1, max: 50, step: 1 },
  early_stopping_patience: { min: 1, max: 50, step: 1 },
  physics_loss_weight: { min: 0.001, max: 10, step: 0.001 },
  output_dim: { min: 1, max: 256, step: 1 },
};

/**
 * Traverse the pipeline graph backwards from a given node to find all
 * upstream Input nodes and any TrainTestSplit node, then collect training
 * data metadata (feature/label names, input kind, holdout ratio, etc.).
 */
function collectDataInfo(
  startNodeId: string,
  nodes: import('../types').SurroNode[],
  edges: import('../types').SurroEdge[],
): HPTuningDataInfo | undefined {
  // Build adjacency: target → sources (upstream direction)
  const upstreamMap = new Map<string, Set<string>>();
  for (const e of edges) {
    if (!upstreamMap.has(e.target)) upstreamMap.set(e.target, new Set());
    upstreamMap.get(e.target)!.add(e.source);
  }

  // BFS / DFS backwards from startNodeId
  const visited = new Set<string>();
  const queue = [startNodeId];
  const inputNodes: import('../types').SurroNode[] = [];
  let holdoutRatio: number | undefined;

  while (queue.length > 0) {
    const current = queue.pop()!;
    if (visited.has(current)) continue;
    visited.add(current);

    const node = nodes.find((n) => n.id === current);
    if (node) {
      const d = node.data as Record<string, unknown>;
      const cat = d.category as string | undefined;

      if (cat === 'input') {
        inputNodes.push(node);
      }

      // Check for TrainTestSplit to extract holdout_ratio
      if (
        cat === 'feature_engineering' &&
        (d as FeatureEngineeringNodeData).method === 'TrainTestSplit'
      ) {
        const hp = (d as FeatureEngineeringNodeData).hyperparams ?? {};
        const hr = Number(hp.holdout_ratio);
        if (!Number.isNaN(hr)) holdoutRatio = hr;
      }
    }

    // Enqueue upstream neighbours
    const parents = upstreamMap.get(current);
    if (parents) {
      for (const pid of parents) queue.push(pid);
    }
  }

  if (inputNodes.length === 0) return undefined;

  // Aggregate features / labels across all input nodes
  const allFeatures: string[] = [];
  const allLabels: string[] = [];
  let inputKind = 'scalar';
  let fileName = '';
  let source = '';

  for (const inp of inputNodes) {
    const d = inp.data as InputNodeData;
    allFeatures.push(...(d.features ?? []));
    allLabels.push(...(d.labels ?? []));
    if (d.inputKind) inputKind = d.inputKind;
    if (d.fileName) fileName = d.fileName;
    if (d.source) source = d.source;
  }

  return {
    n_features: allFeatures.length,
    n_labels: allLabels.length,
    feature_names: allFeatures,
    label_names: allLabels,
    input_kind: inputKind,
    file_name: fileName,
    source,
    holdout_ratio: holdoutRatio,
  };
}

function AgentBasedTunerUI({
  nodeId,
  data,
  nodes,
  edges,
  updateNodeData,
  globalSeed,
}: {
  nodeId: string;
  data: HPTunerNodeData;
  nodes: import('../types').SurroNode[];
  edges: import('../types').SurroEdge[];
  updateNodeData: (id: string, partial: Partial<SurroNodeData>) => void;
  globalSeed: number | null;
}) {
  const [tuningRunning, setTuningRunning] = useState(false);
  // Local raw-text state for discrete value inputs so semicolons aren't
  // immediately stripped by the parse→join round-trip on every keystroke.
  const [discreteRawText, setDiscreteRawText] = useState<Record<string, string>>({});
  // Visual feedback after applying best config
  const [appliedFeedback, setAppliedFeedback] = useState(false);

  // ── Find connected predictor ──────────────────────────────────────────
  const findConnectedPredictor = useCallback(() => {
    const connectedIds = new Set<string>();
    for (const e of edges) {
      if (e.source === nodeId) connectedIds.add(e.target);
      if (e.target === nodeId) connectedIds.add(e.source);
    }
    for (const n of nodes) {
      if (connectedIds.has(n.id)) {
        const cat = (n.data as any).category;
        if (cat === 'regressor' || cat === 'classifier') return n;
      }
    }
    return null;
  }, [nodeId, nodes, edges]);

  // ── Load HPs from connected predictor ─────────────────────────────────
  const loadPredictorHPs = useCallback(() => {
    const predictor = findConnectedPredictor();
    if (!predictor) {
      alert('No regressor or classifier connected. Please connect one to this HP Tuner node.');
      return;
    }

    const predData = predictor.data as RegressorNodeData | ClassifierNodeData;
    const hp = predData.hyperparams ?? {};

    const tunableParams: HPTuneParam[] = Object.entries(hp).map(([key, value]) => {
      const type: HPTuneParam['type'] =
        typeof value === 'number' ? 'number'
        : typeof value === 'boolean' ? 'boolean'
        : 'string';

      const param: HPTuneParam = {
        key,
        type,
        currentValue: value,
        selected: false,
      };

      if (type === 'number') {
        const range = defaultNumericRanges[key];
        if (range) {
          param.min = range.min;
          param.max = range.max;
          param.step = range.step;
        } else {
          const v = value as number;
          if (Number.isInteger(v) && v > 0) {
            param.min = Math.max(1, Math.floor(v / 5));
            param.max = v * 5;
            param.step = 1;
          } else if (v > 0) {
            param.min = Math.max(1e-8, v / 10);
            param.max = v * 10;
            param.step = v / 100;
          } else {
            param.min = 0;
            param.max = 1;
            param.step = 0.01;
          }
        }
      }

      if (type === 'string') {
        if (selectOptions[key]) {
          param.options = selectOptions[key].map((o) => o.value);
        } else {
          // Fall back to a single-element list with the current value
          param.options = [String(value)];
        }
      }

      return param;
    });

    updateNodeData(nodeId, {
      connectedPredictorId: predictor.id,
      tunableParams,
      tuningStatus: 'idle',
      tuningResults: undefined,
      bestConfig: undefined,
      bestScore: undefined,
      tuningError: undefined,
    } as Partial<SurroNodeData>);
  }, [findConnectedPredictor, nodeId, updateNodeData]);

  // ── Toggle HP selection ───────────────────────────────────────────────
  const toggleParamSelection = useCallback(
    (key: string) => {
      const params = data.tunableParams ?? [];
      const updated = params.map((p) =>
        p.key === key ? { ...p, selected: !p.selected } : p,
      );
      updateNodeData(nodeId, { tunableParams: updated } as Partial<SurroNodeData>);
    },
    [data.tunableParams, nodeId, updateNodeData],
  );

  // ── Update param range ────────────────────────────────────────────────
  const updateParamRange = useCallback(
    (key: string, field: 'min' | 'max' | 'step', value: number) => {
      const params = data.tunableParams ?? [];
      const updated = params.map((p) =>
        p.key === key ? { ...p, [field]: value } : p,
      );
      updateNodeData(nodeId, { tunableParams: updated } as Partial<SurroNodeData>);
    },
    [data.tunableParams, nodeId, updateNodeData],
  );

  // ── Toggle range ↔ discrete mode ─────────────────────────────────────
  const toggleDiscreteMode = useCallback(
    (key: string) => {
      const params = data.tunableParams ?? [];
      const updated = params.map((p) =>
        p.key === key ? { ...p, useDiscreteValues: !p.useDiscreteValues } : p,
      );
      updateNodeData(nodeId, { tunableParams: updated } as Partial<SurroNodeData>);
    },
    [data.tunableParams, nodeId, updateNodeData],
  );

  // ── Update discrete values (local raw text + commit on blur) ─────────
  const onDiscreteTextChange = useCallback(
    (key: string, raw: string) => {
      setDiscreteRawText((prev) => ({ ...prev, [key]: raw }));
    },
    [],
  );

  const commitDiscreteValues = useCallback(
    (key: string) => {
      const raw = discreteRawText[key];
      if (raw === undefined) return;
      const params = data.tunableParams ?? [];
      const parsed = raw
        .split(';')
        .map((s) => s.trim())
        .filter((s) => s.length > 0)
        .map((s) => {
          const n = Number(s);
          return Number.isNaN(n) ? s : n;
        });
      const updated = params.map((p) =>
        p.key === key ? { ...p, discreteValues: parsed } : p,
      );
      updateNodeData(nodeId, { tunableParams: updated } as Partial<SurroNodeData>);
    },
    [discreteRawText, data.tunableParams, nodeId, updateNodeData],
  );

  // ── Toggle a string option on/off for tuning ─────────────────────────
  const toggleStringOption = useCallback(
    (key: string, option: string) => {
      const params = data.tunableParams ?? [];
      const updated = params.map((p) => {
        if (p.key !== key) return p;
        const allOptions = p.options ?? [];
        const current = p.selectedOptions ?? [...allOptions]; // default: all selected
        const next = current.includes(option)
          ? current.filter((o) => o !== option)
          : [...current, option];
        // Don't allow deselecting everything — keep at least one
        if (next.length === 0) return p;
        return { ...p, selectedOptions: next };
      });
      updateNodeData(nodeId, { tunableParams: updated } as Partial<SurroNodeData>);
    },
    [data.tunableParams, nodeId, updateNodeData],
  );

  // ── Start tuning ─────────────────────────────────────────────────────
  const startTuning = useCallback(async () => {
    const selected = (data.tunableParams ?? []).filter((p) => p.selected);
    if (selected.length === 0) {
      alert('Please select at least one hyperparameter to tune.');
      return;
    }

    const predictorId = data.connectedPredictorId;
    if (!predictorId) {
      alert('No connected predictor. Please load HPs first.');
      return;
    }

    setTuningRunning(true);
    updateNodeData(nodeId, {
      tuningStatus: 'running',
      tuningError: undefined,
    } as Partial<SurroNodeData>);

    try {
      const storeState = useStore.getState();

      // ── Collect training data metadata from upstream Input nodes ───
      const dataInfo = collectDataInfo(
        predictorId,
        storeState.nodes,
        storeState.edges,
      );

      const payload = {
        nodes: storeState.nodes.map((n) => ({
          id: n.id,
          type: n.type,
          data: n.data,
        })),
        edges: storeState.edges.map((e) => ({
          source: e.source,
          target: e.target,
          sourceHandle: e.sourceHandle,
          targetHandle: e.targetHandle,
        })),
        tuner_node_id: nodeId,
        predictor_node_id: predictorId,
        selected_params: selected.map((p) => ({
          key: p.key,
          type: p.type,
          currentValue: p.currentValue,
          min: p.min,
          max: p.max,
          step: p.step,
          options: p.selectedOptions ?? p.options,
          discreteValues: p.useDiscreteValues ? p.discreteValues : undefined,
        })),
        n_iterations: Number(data.hyperparams.n_iterations) || 50,
        exploration_rate: Number(data.hyperparams.exploration_rate) || 0.1,
        scoring_metric: String(data.hyperparams.scoring_metric || 'r2'),
        seed: globalSeed,
        data_info: dataInfo,
      };

      const result = await runAgentHPTuning(payload);

      if (result.ok) {
        updateNodeData(nodeId, {
          tuningStatus: 'done',
          tuningResults: result.history as HPTuningIterationResult[],
          bestConfig: result.best_config as Record<string, string | number | boolean>,
          bestScore: result.best_score,
          tuningError: undefined,
        } as Partial<SurroNodeData>);
      } else {
        updateNodeData(nodeId, {
          tuningStatus: 'error',
          tuningError: result.error ?? 'Unknown error',
        } as Partial<SurroNodeData>);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      updateNodeData(nodeId, {
        tuningStatus: 'error',
        tuningError: msg,
      } as Partial<SurroNodeData>);
    } finally {
      setTuningRunning(false);
    }
  }, [data, nodeId, updateNodeData, globalSeed]);

  // ── Apply best config to predictor ────────────────────────────────────
  const applyBestConfig = useCallback(() => {
    const predictorId = data.connectedPredictorId;
    const bestConfig = data.bestConfig;
    if (!predictorId || !bestConfig) return;

    const predictor = nodes.find((n) => n.id === predictorId);
    if (!predictor) return;

    const predData = predictor.data as RegressorNodeData | ClassifierNodeData;
    const updatedHP = { ...predData.hyperparams, ...bestConfig };
    updateNodeData(predictorId, { hyperparams: updatedHP } as Partial<SurroNodeData>);

    setAppliedFeedback(true);
    setTimeout(() => setAppliedFeedback(false), 2000);
  }, [data.connectedPredictorId, data.bestConfig, nodes, updateNodeData]);

  const tunableParams = data.tunableParams ?? [];
  const selectedCount = tunableParams.filter((p) => p.selected).length;
  const tuningStatus = data.tuningStatus ?? 'idle';

  return (
    <>
      <div className="inspector__section-title" style={{ marginTop: 12 }}>
        Agent-Based Tuning
      </div>

      {/* ── Load HPs button ──────────────────────────────────────────── */}
      <button
        className="btn btn--load-columns"
        onClick={loadPredictorHPs}
        style={{ marginBottom: 8, width: '100%' }}
      >
        📋 Load HPs from Connected Predictor
      </button>

      {data.connectedPredictorId && (
        <p className="inspector__hint" style={{ fontSize: '0.72rem', opacity: 0.6, margin: '0 0 6px' }}>
          Connected to: {nodes.find((n) => n.id === data.connectedPredictorId)?.data?.label ?? data.connectedPredictorId}
        </p>
      )}

      {/* ── Tunable parameters list ──────────────────────────────────── */}
      {tunableParams.length > 0 && (
        <>
          <div className="inspector__section-title">
            Select HPs to Tune ({selectedCount}/{tunableParams.length})
          </div>
          <div className="hp-tune-list">
            {tunableParams.map((p) => (
              <div
                key={p.key}
                className={`hp-tune-card${p.selected ? ' hp-tune-card--active' : ''}`}
              >
                {/* ── Header row: checkbox + name + current value ── */}
                <label className="hp-tune-card__header">
                  <input
                    type="checkbox"
                    checked={p.selected}
                    onChange={() => toggleParamSelection(p.key)}
                  />
                  <span className="hp-tune-card__name">
                    {p.key.replace(/_/g, ' ')}
                  </span>
                  <span className="hp-tune-card__current">
                    {p.type === 'boolean'
                      ? (p.currentValue ? 'true' : 'false')
                      : String(p.currentValue)}
                  </span>
                </label>

                {/* ── Expanded controls for selected numeric params ── */}
                {p.selected && p.type === 'number' && (
                  <div className="hp-tune-card__body">
                    {/* Mode toggle: Range / Discrete */}
                    <div className="hp-tune-card__mode-toggle">
                      <button
                        className={`hp-tune-card__mode-btn${!p.useDiscreteValues ? ' hp-tune-card__mode-btn--active' : ''}`}
                        onClick={() => p.useDiscreteValues && toggleDiscreteMode(p.key)}
                      >
                        Range
                      </button>
                      <button
                        className={`hp-tune-card__mode-btn${p.useDiscreteValues ? ' hp-tune-card__mode-btn--active' : ''}`}
                        onClick={() => !p.useDiscreteValues && toggleDiscreteMode(p.key)}
                      >
                        Discrete
                      </button>
                    </div>

                    {!p.useDiscreteValues ? (
                      /* ── Range mode: min / max / step ── */
                      <div className="hp-tune-card__range-grid">
                        <div className="inspector__field">
                          <span>min</span>
                          <input
                            type="number"
                            value={p.min ?? 0}
                            step={p.step ?? 0.01}
                            onChange={(e) =>
                              updateParamRange(p.key, 'min', parseFloat(e.target.value) || 0)
                            }
                          />
                        </div>
                        <div className="inspector__field">
                          <span>max</span>
                          <input
                            type="number"
                            value={p.max ?? 1}
                            step={p.step ?? 0.01}
                            onChange={(e) =>
                              updateParamRange(p.key, 'max', parseFloat(e.target.value) || 1)
                            }
                          />
                        </div>
                        <div className="inspector__field">
                          <span>step</span>
                          <input
                            type="number"
                            value={p.step ?? 0.01}
                            onChange={(e) =>
                              updateParamRange(p.key, 'step', parseFloat(e.target.value) || 0.01)
                            }
                          />
                        </div>
                      </div>
                    ) : (
                      /* ── Discrete mode: semicolon-separated values ── */
                      <div className="inspector__field">
                        <span>values (separated by ;)</span>
                        <input
                          type="text"
                          placeholder="e.g. 32; 64; 128; 256"
                          value={
                            discreteRawText[p.key] !== undefined
                              ? discreteRawText[p.key]
                              : (p.discreteValues ?? []).join('; ')
                          }
                          onChange={(e) => onDiscreteTextChange(p.key, e.target.value)}
                          onBlur={() => commitDiscreteValues(p.key)}
                        />
                      </div>
                    )}
                  </div>
                )}

                {/* ── Expanded controls for selected string params ── */}
                {p.selected && p.type === 'string' && p.options && (
                  <div className="hp-tune-card__body">
                    <div className="hp-tune-card__options">
                      {p.options.map((opt) => {
                        const sel = p.selectedOptions ?? p.options ?? [];
                        const isActive = sel.includes(opt);
                        return (
                          <button
                            key={opt}
                            className={`hp-tune-card__option-chip${isActive ? ' hp-tune-card__option-chip--active' : ''}`}
                            onClick={() => toggleStringOption(p.key, opt)}
                            type="button"
                          >
                            {opt}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* ── Boolean: no extra controls needed ── */}
              </div>
            ))}
          </div>

          {/* ── Run / status ─────────────────────────────────────────── */}
          <button
            className="btn"
            onClick={startTuning}
            disabled={tuningRunning || selectedCount === 0}
            style={{
              width: '100%',
              marginBottom: 6,
              background: tuningRunning ? '#555' : '#2dd4bf',
              color: '#000',
              fontWeight: 600,
            }}
          >
            {tuningRunning ? '⏳ Tuning in progress…' : '▶ Start Tuning'}
          </button>

          {tuningStatus === 'error' && data.tuningError && (
            <p style={{ color: '#f87171', fontSize: '0.78rem', margin: '4px 0' }}>
              Error: {data.tuningError}
            </p>
          )}
        </>
      )}

      {/* ── Results summary (compact — full results in Output panel) ─── */}
      {data.tuningResults && data.tuningResults.length > 0 && (
        <>
          {data.bestConfig && data.bestScore != null && (
            <div style={{ background: 'rgba(45,212,191,0.12)', borderRadius: 6, padding: '6px 8px', marginTop: 8, marginBottom: 4 }}>
              <div style={{ fontWeight: 600, fontSize: '0.82rem', marginBottom: 4 }}>
                🏆 Best: {data.hyperparams.scoring_metric} = {data.bestScore.toFixed(6)}
              </div>
              <div style={{ fontSize: '0.72rem', opacity: 0.8 }}>
                {Object.entries(data.bestConfig).map(([k, v]) => (
                  <span key={k} style={{ marginRight: 8 }}>{k.replace(/_/g, ' ')}: <b>{String(v)}</b></span>
                ))}
              </div>
              <button
                className="btn"
                onClick={applyBestConfig}
                style={{
                  marginTop: 6,
                  fontSize: '0.75rem',
                  padding: '3px 10px',
                  background: appliedFeedback ? '#22c55e' : undefined,
                  transition: 'background 0.3s',
                }}
              >
                {appliedFeedback ? '✅ Applied!' : '✅ Apply Best to Predictor'}
              </button>
            </div>
          )}
          <p className="inspector__hint" style={{ fontSize: '0.68rem', opacity: 0.5, margin: '2px 0 0' }}>
            Full results &amp; analytics available in the Output panel below.
          </p>
        </>
      )}
    </>
  );
}

// ─── Inspector Panel ────────────────────────────────────────────────────────

export default function Inspector() {
  const { nodes, selectedNodeId, updateNodeData, deleteNode, setSelectedNode, nodeResults } = useStore();
  const [csvLoading, setCsvLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dropActive, setDropActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── All hooks MUST be above any early return ──────────────────────────────

  // Handle file upload (from drop or file picker)
  const handleFileUpload = useCallback(async (file: File) => {
    if (!selectedNodeId) return;
    setUploading(true);
    try {
      const result = await uploadFile(file);
      if (result.ok && result.fileId) {
        // Derive a flat column list from the structure for the column picker
        const struct: DataStructure | undefined = result.structure;
        let columns: string[] = result.columns ?? [];
        if (struct?.format === 'h5' && struct.groups) {
          // Flatten HDF5 datasets into full paths
          columns = [];
          for (const [grpPath, grpInfo] of Object.entries(struct.groups)) {
            for (const dsName of Object.keys(grpInfo.datasets ?? {})) {
              const full = grpPath === '/' ? `/${dsName}` : `${grpPath}/${dsName}`;
              columns.push(full);
            }
          }
        }
        updateNodeData(selectedNodeId, {
          source: result.fileId,
          fileName: result.originalName ?? file.name,
          columns,
          features: [],
          labels: [],
          structure: struct,
        } as Partial<SurroNodeData>);
      } else {
        alert(`Upload error: ${result.error}`);
      }
    } catch {
      alert('Cannot reach backend. Is the server running?');
    } finally {
      setUploading(false);
    }
  }, [selectedNodeId, updateNodeData]);

  // Load file structure from backend (re-fetch for already-uploaded files)
  const loadColumns = useCallback(async () => {
    if (!selectedNodeId) return;
    const node = nodes.find((n) => n.id === selectedNodeId);
    if (!node) return;
    const inp = node.data as InputNodeData;
    if (!inp.source) return;
    setCsvLoading(true);
    try {
      const res = await fetchStructure(inp.source);
      if (res.ok && res.structure) {
        const struct = res.structure;
        let columns: string[] = [];
        if (struct.format === 'csv') {
          columns = struct.columns ?? [];
        } else if (struct.format === 'h5' && struct.groups) {
          for (const [grpPath, grpInfo] of Object.entries(struct.groups)) {
            for (const dsName of Object.keys(grpInfo.datasets ?? {})) {
              const full = grpPath === '/' ? `/${dsName}` : `${grpPath}/${dsName}`;
              columns.push(full);
            }
          }
        }
        updateNodeData(selectedNodeId, { columns, structure: struct } as Partial<SurroNodeData>);
      } else {
        alert(`Error: ${res.error}`);
      }
    } catch {
      alert('Cannot reach backend. Is the server running?');
    } finally {
      setCsvLoading(false);
    }
  }, [nodes, selectedNodeId, updateNodeData]);

  // Toggle column role
  const handleColumnToggle = useCallback(
    (col: string, role: 'feature' | 'label' | 'none') => {
      if (!selectedNodeId) return;
      const node = nodes.find((n) => n.id === selectedNodeId);
      if (!node) return;
      const inp = node.data as InputNodeData;
      let features = [...(inp.features || [])];
      let labels = [...(inp.labels || [])];

      // Remove from both first
      features = features.filter((c) => c !== col);
      labels = labels.filter((c) => c !== col);

      if (role === 'feature') features.push(col);
      if (role === 'label') labels.push(col);

      updateNodeData(selectedNodeId, { features, labels } as Partial<SurroNodeData>);
    },
    [nodes, selectedNodeId, updateNodeData],
  );

  // ── Early returns (after all hooks) ───────────────────────────────────────

  if (!selectedNodeId) {
    return (
      <aside className="inspector inspector--empty">
        <p>Select a node to inspect its properties.</p>
      </aside>
    );
  }

  const node = nodes.find((n) => n.id === selectedNodeId);
  if (!node) return null;

  const data = node.data as SurroNodeData;
  const accent = categoryColor[data.category];

  const handleChange = (key: string, value: string) => {
    updateNodeData(selectedNodeId, { [key]: value } as Partial<SurroNodeData>);
  };

  const handleHyperparamChange = (key: string, value: string | number | boolean) => {
    const withHP = data as RegressorNodeData | ClassifierNodeData | FeatureEngineeringNodeData;
    updateNodeData(selectedNodeId, {
      hyperparams: { ...withHP.hyperparams, [key]: value },
    } as Partial<SurroNodeData>);
  };

  // Check for validator results
  const rawResult = nodeResults[selectedNodeId];
  const isMultiModel = rawResult && 'multi_model' in rawResult && (rawResult as MultiModelValidatorResult).multi_model;
  const validatorResult = isMultiModel ? undefined : rawResult as ValidatorResult | undefined;
  const multiModelResult = isMultiModel ? rawResult as MultiModelValidatorResult : undefined;
  const hasValidatorPlots = (validatorResult && 'per_label' in validatorResult) || isMultiModel;

  const categoryLabel: Record<string, string> = {
    input: 'input',
    regressor: 'regressor',
    classifier: 'classifier',
    validator: 'validator',
    feature_engineering: 'feat. eng.',
    inference: 'inference',
    hp_tuner: 'hp tuner',
  };

  return (
    <aside className="inspector">
      <div className="inspector__header" style={{ borderBottomColor: accent }}>
        <h3>{data.label}</h3>
        <span className="inspector__badge" style={{ background: accent }}>
          {categoryLabel[data.category] ?? data.category}
        </span>
      </div>

      <div className="inspector__body">
        {/* Label (common) */}
        <label className="inspector__field">
          <span>Label</span>
          <input
            type="text"
            value={data.label}
            onChange={(e) => handleChange('label', e.target.value)}
          />
        </label>

        {/* ── Input ────────────────────────────────────────────────────── */}
        {data.category === 'input' && (
          <>
            <label className="inspector__field">
              <span>Data type</span>
              <input type="text" value={(data as InputNodeData).inputKind?.replace('_', ' ')} disabled />
            </label>

            {/* ── File upload drop zone ── */}
            <div className="inspector__section-title">Data File</div>
            <div
              className={`inspector__drop-zone ${dropActive ? 'inspector__drop-zone--active' : ''} ${(data as InputNodeData).fileName ? 'inspector__drop-zone--has-file' : ''}`}
              onDragOver={(e) => {
                if (e.dataTransfer.types.includes('Files')) {
                  e.preventDefault();
                  e.stopPropagation();
                  setDropActive(true);
                }
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setDropActive(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setDropActive(false);
                const files = e.dataTransfer.files;
                if (files.length > 0) handleFileUpload(files[0]);
              }}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.tsv,.txt,.h5,.hdf5"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFileUpload(file);
                  e.target.value = '';
                }}
              />
              {uploading ? (
                <span className="inspector__drop-zone-text">Uploading…</span>
              ) : (data as InputNodeData).fileName ? (
                <>
                  <span className="inspector__drop-zone-icon">📄</span>
                  <span className="inspector__drop-zone-filename">{(data as InputNodeData).fileName}</span>
                  <span className="inspector__drop-zone-hint">Click or drop to replace</span>
                </>
              ) : (
                <>
                  <span className="inspector__drop-zone-icon">📁</span>
                  <span className="inspector__drop-zone-text">Drop a data file here</span>
                  <span className="inspector__drop-zone-hint">CSV or HDF5 — click to browse</span>
                </>
              )}
            </div>

            {(data as InputNodeData).source && (data as InputNodeData).columns?.length === 0 && !uploading && (
              <button
                className="btn btn--load-columns"
                onClick={loadColumns}
                disabled={csvLoading}
              >
                {csvLoading ? 'Loading…' : '📋 Reload Structure'}
              </button>
            )}

            {(data as InputNodeData).columns?.length > 0 && (
              <>
                <div className="inspector__section-title">
                  {(data as InputNodeData).structure?.format === 'h5' ? 'Dataset Roles' : 'Column Roles'}
                </div>
                <ColumnPicker
                  columns={(data as InputNodeData).columns}
                  features={(data as InputNodeData).features || []}
                  labels={(data as InputNodeData).labels || []}
                  onToggle={handleColumnToggle}
                />
              </>
            )}
          </>
        )}

        {/* ── Regressor ────────────────────────────────────────────────── */}
        {data.category === 'regressor' && (
          <>
            <label className="inspector__field">
              <span>Model</span>
              <input type="text" value={(data as RegressorNodeData).model} disabled />
            </label>
            <label className="inspector__field">
              <span>Role</span>
              <select
                value={(data as RegressorNodeData).role ?? 'final'}
                onChange={(e) =>
                  updateNodeData(selectedNodeId, {
                    role: e.target.value as 'transform' | 'final',
                  } as Partial<SurroNodeData>)
                }
              >
                <option value="final">Final (predict labels)</option>
                <option value="transform">Transform (pass-through)</option>
              </select>
            </label>
            <div className="inspector__section-title">Hyperparameters</div>
            <TabbedHyperParams
              hyperparams={(data as RegressorNodeData).hyperparams}
              onChange={handleHyperparamChange}
            />
          </>
        )}

        {/* ── Classifier ───────────────────────────────────────────────── */}
        {data.category === 'classifier' && (
          <>
            <label className="inspector__field">
              <span>Model</span>
              <input type="text" value={(data as ClassifierNodeData).model} disabled />
            </label>
            <div className="inspector__section-title">Hyperparameters</div>
            <TabbedHyperParams
              hyperparams={(data as ClassifierNodeData).hyperparams}
              onChange={handleHyperparamChange}
            />
          </>
        )}

        {/* ── Feature Engineering ───────────────────────────────────────── */}
        {data.category === 'feature_engineering' && (
          <>
            <label className="inspector__field">
              <span>Method</span>
              <input
                type="text"
                value={(data as FeatureEngineeringNodeData).method}
                disabled
              />
            </label>
            <div className="inspector__section-title">Parameters</div>
            <TabbedHyperParams
              hyperparams={(data as FeatureEngineeringNodeData).hyperparams}
              onChange={handleHyperparamChange}
            />
          </>
        )}

        {/* ── Validator ────────────────────────────────────────────────── */}
        {data.category === 'validator' && (
          <>
            <label className="inspector__field">
              <span>Validator kind</span>
              <select
                value={(data as ValidatorNodeData).validatorKind}
                onChange={(e) => handleChange('validatorKind', e.target.value)}
              >
                <option value="classifier_validator">Classifier Validator</option>
                <option value="regressor_validator">Regressor Validator</option>
                <option value="relation_seeker">Relation Seeker</option>
              </select>
            </label>

            <label className="inspector__field">
              <span>Plots per row</span>
              <input
                type="number"
                min={1}
                max={10}
                value={(data as ValidatorNodeData).plotsPerRow ?? 4}
                onChange={(e) =>
                  updateNodeData(selectedNodeId, {
                    plotsPerRow: Math.max(1, parseInt(e.target.value, 10) || 4),
                  } as Partial<SurroNodeData>)
                }
              />
            </label>

            {hasValidatorPlots && !isMultiModel && validatorResult && (
              <ValidatorResultsView result={validatorResult} />
            )}

            {isMultiModel && multiModelResult && (
              <MultiModelResultsView
                result={multiModelResult}
                plotsPerRow={(data as ValidatorNodeData).plotsPerRow ?? 4}
              />
            )}
          </>
        )}

        {/* ── Inference ────────────────────────────────────────────────── */}
        {data.category === 'inference' && (
          <>
            <label className="inspector__field">
              <span>Model source</span>
              <input
                type="text"
                placeholder="ID or name of trained model node"
                value={(data as InferenceNodeData).modelSource}
                onChange={(e) => handleChange('modelSource', e.target.value)}
              />
            </label>
            <label className="inspector__field">
              <span>Batch size</span>
              <input
                type="number"
                value={(data as InferenceNodeData).batchSize}
                min={1}
                onChange={(e) =>
                  updateNodeData(selectedNodeId, {
                    batchSize: parseInt(e.target.value, 10) || 1,
                  } as Partial<SurroNodeData>)
                }
              />
            </label>
          </>
        )}

        {/* ── HP Tuner ─────────────────────────────────────────────────── */}
        {data.category === 'hp_tuner' && (
          <>
            <label className="inspector__field">
              <span>Method</span>
              <input
                type="text"
                value={(data as HPTunerNodeData).method}
                disabled
              />
            </label>
            <div className="inspector__section-title">Parameters</div>
            <HyperParamsEditor
              hyperparams={(data as HPTunerNodeData).hyperparams}
              onChange={handleHyperparamChange}
            />

            {/* ── AgentBased-specific UI ─────────────────────────────────── */}
            {(data as HPTunerNodeData).method === 'AgentBased' && (
              <AgentBasedTunerUI
                nodeId={selectedNodeId}
                data={data as HPTunerNodeData}
                nodes={nodes}
                edges={useStore.getState().edges}
                updateNodeData={updateNodeData}
                globalSeed={useStore.getState().globalSeed}
              />
            )}
          </>
        )}

        {/* ── RBL ──────────────────────────────────────────────────────── */}
        {data.category === 'rbl' && (
          <>
            <div className="inspector__section-title">Loss Weights</div>
            <label className="inspector__field">
              <span>λ kernel (MSE(z, y))</span>
              <input
                type="number"
                step="0.01"
                min={0}
                value={(data as RBLNodeData).lambda_kernel ?? 1.0}
                onChange={(e) =>
                  updateNodeData(selectedNodeId, {
                    lambda_kernel: parseFloat(e.target.value) || 0,
                  } as Partial<SurroNodeData>)
                }
              />
            </label>
            <label className="inspector__field">
              <span>λ residual (mean(r²))</span>
              <input
                type="number"
                step="0.001"
                min={0}
                value={(data as RBLNodeData).lambda_residual ?? 0.01}
                onChange={(e) =>
                  updateNodeData(selectedNodeId, {
                    lambda_residual: parseFloat(e.target.value) || 0,
                  } as Partial<SurroNodeData>)
                }
              />
            </label>
            <p className="inspector__hint" style={{ fontSize: '0.75rem', opacity: 0.7, margin: '0.5rem 0' }}>
              loss = MSE(ŷ, y) + λ_kernel · MSE(z, y) + λ_residual · mean(r²)
            </p>
          </>
        )}

        {/* ── RBL Aggregator ───────────────────────────────────────────── */}
        {data.category === 'rbl_aggregator' && (
          <p className="inspector__hint" style={{ fontSize: '0.8rem', opacity: 0.7, margin: '0.5rem 0' }}>
            Computes ŷ = z + r (primary prediction + residual correction).
          </p>
        )}
      </div>

      <div className="inspector__actions">
        <button className="btn btn--danger" onClick={() => deleteNode(selectedNodeId)}>
          Delete Node
        </button>
        <button className="btn" onClick={() => setSelectedNode(null)}>
          Close
        </button>
      </div>
    </aside>
  );
}
