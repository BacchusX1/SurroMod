import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { HPTunerNodeData } from '../../types';
import { categoryColor } from '../../utils';

const methodIcon: Record<string, string> = {
  GridSearch: '🔍',
  AgentBased: '🤖',
  OptimiserBased: '⚙️',
};

export default function HPTunerNode({ data, selected }: NodeProps) {
  const d = data as unknown as HPTunerNodeData;
  const accent = categoryColor.hp_tuner;

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
        <span className="surro-node__icon">{methodIcon[d.method] ?? '🎯'}</span>
        <span className="surro-node__title">{d.label}</span>
      </div>
      <div className="surro-node__body">
        <span className="surro-node__tag">{d.method}</span>
        {preview && <span className="surro-node__detail">{preview}</span>}
      </div>
      <Handle type="target" position={Position.Left} className="surro-handle" />
      <Handle type="source" position={Position.Right} className="surro-handle" />
    </div>
  );
}
