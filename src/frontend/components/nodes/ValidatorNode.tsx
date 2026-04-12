import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { ValidatorNodeData } from '../../types';
import { categoryColor } from '../../utils';

export default function ValidatorNode({ data, selected }: NodeProps) {
  const d = data as unknown as ValidatorNodeData;
  const accent = categoryColor.validator;

  return (
    <div
      className={`surro-node ${selected ? 'selected' : ''}`}
      style={{ borderColor: accent }}
    >
      <div className="surro-node__header" style={{ background: accent }}>
        <span className="surro-node__title">{d.label}</span>
      </div>
      <div className="surro-node__body">
        <span className="surro-node__tag">{d.validatorKind.replace(/_/g, ' ')}</span>
        {d.validatorKind === 'regressor_validator' && (
          <span className="surro-node__tag surro-node__tag--dim">
            {d.plotsPerRow ?? 4}/row
          </span>
        )}
      </div>
      <Handle type="target" position={Position.Left} className="surro-handle" title="in" />
    </div>
  );
}
