import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { ClassifierNodeData } from '../../types';
import { categoryColor } from '../../utils';

export default function ClassifierNode({ data, selected }: NodeProps) {
  const d = data as unknown as ClassifierNodeData;
  const accent = categoryColor.classifier;

  const hp = d.hyperparams;
  const preview = Object.entries(hp)
    .slice(0, 2)
    .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v}`)
    .join(' · ');

  return (
    <div
      className={`surro-node ${selected ? 'selected' : ''}`}
      style={{ borderColor: accent }}
    >
      <div className="surro-node__header" style={{ background: accent }}>
        <span className="surro-node__icon">🏷️</span>
        <span className="surro-node__title">{d.label}</span>
      </div>
      <div className="surro-node__body">
        <span className="surro-node__tag">{d.model}</span>
        {preview && <span className="surro-node__detail">{preview}</span>}
      </div>
      <Handle type="target" position={Position.Left} className="surro-handle" />
      <Handle type="source" position={Position.Right} className="surro-handle" />
    </div>
  );
}
