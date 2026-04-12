import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { RBLAggregatorNodeData } from '../../types';
import { categoryColor } from '../../utils';

export default function RBLAggregatorNode({ data, selected }: NodeProps) {
  const d = data as unknown as RBLAggregatorNodeData;
  const accent = categoryColor.rbl_aggregator;

  return (
    <div
      className={`surro-node ${selected ? 'selected' : ''}`}
      style={{ borderColor: accent }}
    >
      <div className="surro-node__header" style={{ background: accent }}>
        <span className="surro-node__title">{d.label}</span>
      </div>
      <div className="surro-node__body">
        <span className="surro-node__detail">ŷ = z + r</span>
      </div>
      {/* Left handle: residual prediction from upstream regressor */}
      <Handle type="target" position={Position.Left} className="surro-handle" title="residual" />
      {/* Right handle: output to validator */}
      <Handle type="source" position={Position.Right} className="surro-handle" title="prediction" />
    </div>
  );
}
