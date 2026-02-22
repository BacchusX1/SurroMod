import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { FeatureEngineeringNodeData } from '../../types';
import { categoryColor } from '../../utils';

export default function FeatureEngineeringNode({ data, selected }: NodeProps) {
  const d = data as unknown as FeatureEngineeringNodeData;
  const accent = categoryColor.feature_engineering;

  const hp = d.hyperparams;

  const isDataSplitter = d.method === 'DataSplitter';
  const nOutputs = isDataSplitter ? Number(hp.n_outputs ?? 3) : 1;
  const splitMode = isDataSplitter ? String(hp.split_mode ?? 'channel') : '';

  // Build a compact preview string.
  const previewEntries = Object.entries(hp).slice(0, 2);
  const preview = previewEntries
    .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v}`)
    .join(' · ');

  const methodIcons: Record<string, string> = {
    PCA: '📉',
    GeometrySampler: '📐',
    Scaler: '⚖️',
    DataSplitter: '🔀',
    Autoencoder: '🧬',
  };

  return (
    <div
      className={`surro-node ${selected ? 'selected' : ''}`}
      style={{ borderColor: accent }}
    >
      <div className="surro-node__header" style={{ background: accent }}>
        <span className="surro-node__icon">{methodIcons[d.method] ?? '🔧'}</span>
        <span className="surro-node__title">{d.label}</span>
      </div>
      <div className="surro-node__body">
        <span className="surro-node__tag">{d.method}</span>
        {preview && <span className="surro-node__detail">{preview}</span>}
      </div>
      <Handle type="target" position={Position.Left} className="surro-handle" />
      {isDataSplitter ? (
        Array.from({ length: nOutputs }, (_, i) => (
          <Handle
            key={`out-${i}`}
            type="source"
            position={Position.Right}
            id={`channel-${i}`}
            className="surro-handle"
            style={{ top: `${((i + 1) / (nOutputs + 1)) * 100}%` }}
          />
        ))
      ) : (
        <Handle type="source" position={Position.Right} className="surro-handle" />
      )}
    </div>
  );
}
