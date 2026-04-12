import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { CodeExporterNodeData } from '../../types';
import { categoryColor } from '../../utils';

export default function CodeExporterNode({ data, selected }: NodeProps) {
  const d = data as unknown as CodeExporterNodeData;
  const accent = categoryColor.code_exporter;

  const statusBadge = (() => {
    switch (d.exportStatus) {
      case 'exporting':
        return <span className="hpnode-status hpnode-status--running">Exporting…</span>;
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
      style={{ borderColor: accent }}
    >
      <div className="surro-node__header" style={{ background: accent }}>
        <span className="surro-node__title">{d.label}</span>
        {statusBadge}
      </div>
      <div className="surro-node__body">
        <span className="surro-node__tag">📄 {d.outputFilename ?? 'train.py'}</span>
        {d.exportStatus === 'done' && d.exportPath && (
          <span className="surro-node__detail" style={{ color: accent }}>
            {d.exportPath}
          </span>
        )}
        {d.exportStatus === 'error' && d.exportError && (
          <span className="surro-node__detail" style={{ color: '#f87171' }}>
            {d.exportError}
          </span>
        )}
      </div>
      <Handle type="target" position={Position.Left} className="surro-handle" title="pipeline" />
    </div>
  );
}
