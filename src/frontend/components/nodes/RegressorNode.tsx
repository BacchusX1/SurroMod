import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { RegressorNodeData } from '../../types';
import { categoryColor } from '../../utils';

export default function RegressorNode({ data, selected }: NodeProps) {
  const d = data as unknown as RegressorNodeData;
  const accent = categoryColor.regressor;
  const role = d.role ?? 'final';
  const isGFF = d.model === 'GraphFlowForecaster';

  // Show a few key hyperparams on the node face
  const hp = d.hyperparams;
  const preview = isGFF
    ? `latent: ${hp.latent_dim ?? '?'}  layers: ${hp.num_message_passing_layers ?? '?'}  ${hp.aggregation_mode ?? 'mean'}`
    : Object.entries(hp)
        .filter(([k]) => k !== 'output_dim' && k !== 'lambda_kernel' && k !== 'lambda_residual')
        .slice(0, 2)
        .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v}`)
        .join(' · ');

  // GraphFlowForecaster has custom input ports to show the branched architecture
  const gffInputPorts = isGFF
    ? [
        { id: 'features', label: 'features' },
        { id: 'graph', label: 'graph' },
        { id: 'data', label: 'data' },
      ]
    : [];

  return (
    <div
      className={`surro-node ${selected ? 'selected' : ''}`}
      style={{ borderColor: accent }}
    >
      <div className="surro-node__header" style={{ background: accent }}>
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
      {isGFF ? (
        gffInputPorts.map((port, i) => (
          <Handle
            key={`in-${port.id}`}
            type="target"
            position={Position.Left}
            id={port.id}
            className="surro-handle"
            style={{ top: `${((i + 1) / (gffInputPorts.length + 1)) * 100}%` }}
            title={port.label}
          />
        ))
      ) : (
        <Handle type="target" position={Position.Left} className="surro-handle" title="in" />
      )}
      <Handle type="source" position={Position.Right} className="surro-handle" title="out" />
    </div>
  );
}
