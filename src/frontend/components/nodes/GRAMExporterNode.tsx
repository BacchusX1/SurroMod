import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { GRAMExporterNodeData } from '../../types';
import { categoryColor } from '../../utils';

export default function GRAMExporterNode({ data, selected }: NodeProps) {
  const d = data as unknown as GRAMExporterNodeData;
  const accent = categoryColor.gram_exporter;

  const statusBadge = (() => {
    switch (d.exportStatus) {
      case 'exporting':
        return <span className="hpnode-status hpnode-status--running">Submitting…</span>;
      case 'done':
        return <span className="hpnode-status hpnode-status--done">Exported</span>;
      case 'error':
        return <span className="hpnode-status hpnode-status--error">Error</span>;
      default:
        return null;
    }
  })();

  return (
    <div
      className={`surro-node ${selected ? 'selected' : ''}`}
      style={{ borderColor: accent, minWidth: 160 }}
    >
      <div className="surro-node__header" style={{ background: accent }}>
        <span className="surro-node__title">{d.label}</span>
        {statusBadge}
      </div>
      <div className="surro-node__body">
        <span className="surro-node__tag" style={{ color: accent }}>
          GRaM @ ICLR 2026
        </span>
        <span className="surro-node__detail" style={{ fontSize: '0.72rem', opacity: 0.75 }}>
          {d.hyperparams?.model_name ?? 'surromod_gff'}
        </span>
        {d.exportStatus === 'done' && d.exportDir && (
          <span className="surro-node__detail" style={{ color: accent, fontSize: '0.7rem' }}>
            ✔ {d.exportDir}
          </span>
        )}
        {d.exportStatus === 'done' && d.prUrl && (
          <span className="surro-node__detail" style={{ color: accent, fontSize: '0.7rem' }}>
            PR: {d.prUrl}
          </span>
        )}
        {d.exportStatus === 'error' && d.exportError && (
          <span className="surro-node__detail" style={{ color: '#f87171', fontSize: '0.7rem' }}>
            {d.exportError}
          </span>
        )}
      </div>
      {/* Accepts connection from any upstream node (GFF output) */}
      <Handle type="target" position={Position.Left} className="surro-handle" title="model" />
    </div>
  );
}
