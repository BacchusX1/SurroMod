import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { InferenceNodeData } from '../../types';
import { categoryColor } from '../../utils';

export default function InferenceNode({ data, selected }: NodeProps) {
  const d = data as unknown as InferenceNodeData;
  const accent = categoryColor.inference;

  return (
    <div
      className={`surro-node ${selected ? 'selected' : ''}`}
      style={{ borderColor: accent }}
    >
      <div className="surro-node__header" style={{ background: accent }}>
        <span className="surro-node__icon">🚀</span>
        <span className="surro-node__title">{d.label}</span>
      </div>
      <div className="surro-node__body">
        <span className="surro-node__tag">batch: {d.batchSize}</span>
        {d.modelSource && <span className="surro-node__detail">src: {d.modelSource}</span>}
      </div>
      <Handle type="target" position={Position.Left} className="surro-handle" />
      <Handle type="source" position={Position.Right} className="surro-handle" />
    </div>
  );
}
