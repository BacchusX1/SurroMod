import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { PostprocessingNodeData } from '../../types';
import { categoryColor } from '../../utils';

const kindLabels: Record<string, string> = {
  field_slice_plot: 'Field Slice Plot',
  flow_metrics_summary: 'Flow Metrics Summary',
  prediction_comparison_report: 'Comparison Report',
};

export default function PostProcessingNode({ data, selected }: NodeProps) {
  const d = data as unknown as PostprocessingNodeData;
  const accent = categoryColor.postprocessing;

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
        <span className="surro-node__title">{d.label}</span>
      </div>
      <div className="surro-node__body">
        <span className="surro-node__tag">
          {kindLabels[d.postprocessingKind] ?? d.postprocessingKind}
        </span>
        {preview && <span className="surro-node__detail">{preview}</span>}
      </div>
      <Handle type="target" position={Position.Left} className="surro-handle" title="in" />
    </div>
  );
}
