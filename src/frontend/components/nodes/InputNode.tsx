import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { InputNodeData } from '../../types';
import { categoryColor } from '../../utils';

export default function InputNode({ data, selected }: NodeProps) {
  const d = data as unknown as InputNodeData;
  const accent = categoryColor.input;

  return (
    <div
      className={`surro-node ${selected ? 'selected' : ''}`}
      style={{ borderColor: accent }}
    >
      <div className="surro-node__header" style={{ background: accent }}>
        <span className="surro-node__icon">📂</span>
        <span className="surro-node__title">{d.label}</span>
      </div>
      <div className="surro-node__body">
        <span className="surro-node__tag">{d.dataKind.replace('_', ' ')}</span>
        {d.source && <span className="surro-node__detail">{d.source}</span>}
      </div>
      {/* Input nodes only have an output handle */}
      <Handle type="source" position={Position.Right} className="surro-handle" />
    </div>
  );
}
