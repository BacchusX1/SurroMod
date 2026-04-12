import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { RBLNodeData } from '../../types';
import { categoryColor } from '../../utils';

export default function RBLNode({ data, selected }: NodeProps) {
  const d = data as unknown as RBLNodeData;
  const accent = categoryColor.rbl;

  return (
    <div
      className={`surro-node ${selected ? 'selected' : ''}`}
      style={{ borderColor: accent }}
    >
      <div className="surro-node__header" style={{ background: accent }}>
        <span className="surro-node__title">{d.label}</span>
      </div>
      <div className="surro-node__body">
        <span className="surro-node__detail">
          λ_kernel: {d.lambda_kernel ?? 1.0}
        </span>
        <span className="surro-node__detail">
          λ_residual: {d.lambda_residual ?? 0.01}
        </span>
      </div>
      {/* Left handle: primary prediction z */}
      <Handle type="target" position={Position.Left} id="default" className="surro-handle" title="z" />
      {/* Top handle: representation inputs h_i */}
      <Handle
        type="target"
        position={Position.Top}
        id="representations"
        className="surro-handle"
        style={{ left: '50%' }}
        title="repr"
      />
      {/* Right handle: output to downstream regressor */}
      <Handle type="source" position={Position.Right} className="surro-handle" title="out" />
    </div>
  );
}
