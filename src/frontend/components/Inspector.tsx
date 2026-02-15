import useStore from '../store';
import type {
  SurroNodeData,
  InputNodeData,
  RegressorNodeData,
  ClassifierNodeData,
  ValidatorNodeData,
} from '../types';
import { categoryColor } from '../utils';

// ─── Inspector Panel ────────────────────────────────────────────────────────
// Shown on the right side when a node is selected.

export default function Inspector() {
  const { nodes, selectedNodeId, updateNodeData, deleteNode, setSelectedNode } = useStore();

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

  return (
    <aside className="inspector">
      <div className="inspector__header" style={{ borderBottomColor: accent }}>
        <h3>{data.label}</h3>
        <span className="inspector__badge" style={{ background: accent }}>
          {data.category}
        </span>
      </div>

      <div className="inspector__body">
        {/* Label field (common to all) */}
        <label className="inspector__field">
          <span>Label</span>
          <input
            type="text"
            value={data.label}
            onChange={(e) => handleChange('label', e.target.value)}
          />
        </label>

        {/* Input-specific fields */}
        {data.category === 'input' && (
          <>
            <label className="inspector__field">
              <span>Data kind</span>
              <select
                value={(data as InputNodeData).dataKind}
                onChange={(e) => handleChange('dataKind', e.target.value)}
              >
                <option value="scalar">scalar</option>
                <option value="2d_field">2d field</option>
                <option value="3d_field">3d field</option>
                <option value="time_series">time series</option>
                <option value="step">step</option>
              </select>
            </label>
            <label className="inspector__field">
              <span>Source</span>
              <input
                type="text"
                placeholder="e.g. path/to/data.csv"
                value={(data as InputNodeData).source}
                onChange={(e) => handleChange('source', e.target.value)}
              />
            </label>
          </>
        )}

        {/* Regressor-specific fields */}
        {data.category === 'regressor' && (
          <>
            <label className="inspector__field">
              <span>Data kind</span>
              <select
                value={(data as RegressorNodeData).dataKind}
                onChange={(e) => handleChange('dataKind', e.target.value)}
              >
                <option value="scalar">scalar</option>
                <option value="2d_field">2d field</option>
                <option value="3d_field">3d field</option>
                <option value="time_series">time series</option>
              </select>
            </label>
            <label className="inspector__field">
              <span>Method</span>
              <input
                type="text"
                value={(data as RegressorNodeData).method}
                onChange={(e) => handleChange('method', e.target.value)}
              />
            </label>
          </>
        )}

        {/* Classifier-specific fields */}
        {data.category === 'classifier' && (
          <>
            <label className="inspector__field">
              <span>Data kind</span>
              <select
                value={(data as ClassifierNodeData).dataKind}
                onChange={(e) => handleChange('dataKind', e.target.value)}
              >
                <option value="scalar">scalar</option>
                <option value="2d_field">2d field</option>
                <option value="3d_field">3d field</option>
                <option value="time_series">time series</option>
              </select>
            </label>
            <label className="inspector__field">
              <span>Method</span>
              <input
                type="text"
                value={(data as ClassifierNodeData).method}
                onChange={(e) => handleChange('method', e.target.value)}
              />
            </label>
          </>
        )}

        {/* Validator-specific fields */}
        {data.category === 'validator' && (
          <label className="inspector__field">
            <span>Validator kind</span>
            <select
              value={(data as ValidatorNodeData).validatorKind}
              onChange={(e) => handleChange('validatorKind', e.target.value)}
            >
              <option value="classifier_validator">classifier validator</option>
              <option value="regressor_validator">regressor validator</option>
              <option value="relation_seeker">relation seeker</option>
            </select>
          </label>
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
