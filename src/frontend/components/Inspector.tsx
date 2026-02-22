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
  HyperParams,
  ValidatorResult,
  MultiModelValidatorResult,
  MultiModelEntry,
} from '../types';
import { categoryColor, advancedKeys } from '../utils';
import { uploadFile, fetchStructure } from '../api';
import type { DataStructure } from '../api';

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
  return (
    <div className="validator-results">
      <div className="inspector__section-title">Overall Metrics</div>
      <div className="validator-results__metrics">
        {Object.entries(result.metrics).map(([key, val]) => (
          <div key={key} className="validator-results__metric">
            <span className="validator-results__metric-key">{key.toUpperCase()}</span>
            <span className="validator-results__metric-val">{(val as number).toFixed(4)}</span>
          </div>
        ))}
      </div>

      {result.per_label.map((pl) => (
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
  // Collect all unique label names across models
  const labelNames = result.model_results.length > 0
    ? result.model_results[0].per_label.map((pl) => pl.label)
    : [];

  return (
    <div className="validator-results">
      {/* ── Comparison bar chart ─────────────────────────────────────── */}
      <div className="inspector__section-title">Metrics Comparison</div>
      {result.comparison_bar_plot && (
        <img
          className="validator-results__plot"
          src={`data:image/png;base64,${result.comparison_bar_plot}`}
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
              {Object.keys(result.model_results[0]?.metrics ?? {}).map((k) => (
                <th key={k}>{k.toUpperCase()}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.model_results.map((mr: MultiModelEntry) => (
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
                result.model_results.length,
                plotsPerRow,
              )}, 1fr)`,
            }}
          >
            {result.model_results.map((mr: MultiModelEntry) => {
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
          </>
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
