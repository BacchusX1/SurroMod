import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { HPTunerNodeData } from '../../types';
import { categoryColor } from '../../utils';

export default function HPTunerNode({ data, selected }: NodeProps) {
  const d = data as unknown as HPTunerNodeData;
  const accent = categoryColor.hp_tuner;

  const hp = d.hyperparams;
  const preview = Object.entries(hp)
    .slice(0, 2)
    .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v}`)
    .join(' · ');

  // Status badge
  const statusBadge = (() => {
    switch (d.tuningStatus) {
      case 'running':
        return <span className="hpnode-status hpnode-status--running">Running…</span>;
      case 'done':
        return <span className="hpnode-status hpnode-status--done">Done</span>;
      case 'stopped':
        return <span className="hpnode-status hpnode-status--stopped">Stopped</span>;
      case 'error':
        return <span className="hpnode-status hpnode-status--error">Error</span>;
      default:
        return null;
    }
  })();

  return (
    <div
      className={`surro-node ${selected ? 'selected' : ''}`}
      style={{ borderColor: accent }}
    >
      <div className="surro-node__header" style={{ background: accent }}>
        <span className="surro-node__title">{d.label}</span>
        {statusBadge}
      </div>

      <div className="surro-node__body">
        <span className="surro-node__tag">{d.method}</span>
        {preview && <span className="surro-node__detail">{preview}</span>}
        {d.tuningResults && d.tuningResults.length > 0 && d.bestScore != null && (
          <span className="surro-node__detail" style={{ color: '#2dd4bf' }}>
            Best: {d.bestScore.toFixed(6)}
          </span>
        )}
      </div>

      <Handle type="target" position={Position.Left} className="surro-handle" title="predictor" />
      <Handle type="source" position={Position.Right} className="surro-handle" title="tuned" />
    </div>
  );
}
