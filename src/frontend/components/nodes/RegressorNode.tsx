import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { RegressorNodeData } from '../../types';
import { categoryColor } from '../../utils';

export default function RegressorNode({ data, selected }: NodeProps) {
  const d = data as unknown as RegressorNodeData;
  const accent = categoryColor.regressor;
  const role = d.role ?? 'final';

  // Show a few key hyperparams on the node face
  const hp = d.hyperparams;
  const preview = Object.entries(hp)
    .filter(([k]) => k !== 'output_dim' && k !== 'lambda_kernel' && k !== 'lambda_residual')
    .slice(0, 2)
    .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v}`)
    .join(' · ');

  return (
    <div
      className={`surro-node ${selected ? 'selected' : ''}`}
      style={{ borderColor: accent }}
    >
      <div className="surro-node__header" style={{ background: accent }}>
        <span className="surro-node__icon">📈</span>
        <span className="surro-node__title">{d.label}</span>
        {role === 'transform' && (
          <span className="surro-node__role-badge" title="Transform (pass-through)">T</span>
        )}
      </div>
      <div className="surro-node__body">
        <span className="surro-node__tag">{d.model}</span>
        {role === 'transform' && hp.output_dim !== undefined && Number(hp.output_dim) > 0 && (
          <span className="surro-node__detail">out: {hp.output_dim}</span>
        )}
        {preview && <span className="surro-node__detail">{preview}</span>}
      </div>
      <Handle type="target" position={Position.Left} className="surro-handle" />
      <Handle type="source" position={Position.Right} className="surro-handle" />
    </div>
  );
}
