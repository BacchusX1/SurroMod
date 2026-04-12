import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { FeatureEngineeringNodeData } from '../../types';
import { categoryColor } from '../../utils';

export default function FeatureEngineeringNode({ data, selected }: NodeProps) {
  const d = data as unknown as FeatureEngineeringNodeData;
  const accent = categoryColor.feature_engineering;

  const hp = d.hyperparams;

  // Determine custom input/output ports for specialised nodes
  const customInputPorts: { id: string; label: string }[] = [];
  const customOutputPorts: { id: string; label: string }[] = [];

  if (d.method === 'SpatialGraphBuilder') {
    customInputPorts.push({ id: 'pos', label: 'pos' });
    customOutputPorts.push({ id: 'edge_index', label: 'edge_idx' });
    customOutputPorts.push({ id: 'edge_attr', label: 'edge_attr' });
  } else if (d.method === 'SurfaceDistanceFeature') {
    customInputPorts.push({ id: 'pos', label: 'pos' });
    customInputPorts.push({ id: 'surface_info', label: 'surface' });
    customOutputPorts.push({ id: 'dist_to_surface', label: 'dist' });
    customOutputPorts.push({ id: 'geometry_mask', label: 'mask' });
    if (hp.return_vector) {
      customOutputPorts.push({ id: 'nearest_surface_vec', label: 'vec' });
    }
  } else if (d.method === 'TemporalStackFlatten') {
    const srcField = (hp.source_field as string) || 'velocity_in';
    const outField = (hp.output_field as string) || 'velocity_history_features';
    customInputPorts.push({ id: srcField, label: srcField.replace(/_/g, ' ') });
    customOutputPorts.push({ id: outField, label: outField.replace(/_/g, ' ') });
  } else if (d.method === 'PointFeatureFusion') {
    customInputPorts.push({ id: 'pos', label: 'pos' });
    customInputPorts.push({ id: 'velocity_history_features', label: 'vel_hist' });
    customInputPorts.push({ id: 'geometry_mask', label: 'mask' });
    customInputPorts.push({ id: 'dist_to_surface', label: 'dist' });
    if (hp.include_nearest_surface_vec) {
      customInputPorts.push({ id: 'nearest_surface_vec', label: 'vec' });
    }
    if (hp.include_pressure) {
      customInputPorts.push({ id: 'pressure', label: 'pressure' });
    }
    if (hp.include_low_freq) {
      customInputPorts.push({ id: 'vel_low_freq_features', label: 'low_freq' });
    }
    if (hp.include_high_freq) {
      customInputPorts.push({ id: 'vel_high_freq_features', label: 'high_freq' });
    }
    customOutputPorts.push({ id: 'point_features', label: 'features' });
  } else if (d.method === 'FeatureNormalizer') {
    customInputPorts.push({ id: 'point_features', label: 'features' });
    customOutputPorts.push({ id: 'point_features', label: 'features' });
  } else if (d.method === 'SpectralDecomposer') {
    customInputPorts.push({ id: 'velocity_in', label: 'vel_in' });
    customOutputPorts.push({ id: 'vel_low_freq', label: 'low_freq' });
    customOutputPorts.push({ id: 'vel_high_freq', label: 'high_freq' });
  } else if (d.method === 'HierarchicalGraphBuilder') {
    customInputPorts.push({ id: 'pos', label: 'pos' });
    customOutputPorts.push({ id: 'edge_index_fine', label: 'fine edges' });
    customOutputPorts.push({ id: 'edge_index_coarse', label: 'coarse edges' });
    customOutputPorts.push({ id: 'coarse_indices', label: 'coarse idx' });
    customOutputPorts.push({ id: 'unpool_ftc_idx', label: 'unpool idx' });
    customOutputPorts.push({ id: 'dist_to_af', label: 'dist_af' });
  } else if (d.method === 'TemporalXLSTMEncoder') {
    customInputPorts.push({ id: 'velocity_in', label: 'vel_in' });
    if (hp.include_pressure) {
      customInputPorts.push({ id: 'pressure_in', label: 'pressure' });
    }
    customOutputPorts.push({ id: 'temporal_features', label: 'temporal_features' });
  }

  const hasCustomPorts = customInputPorts.length > 0 || customOutputPorts.length > 0;

  // Build a compact preview string.
  const previewEntries = Object.entries(hp).slice(0, 2);
  const preview = previewEntries
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
        <span className="surro-node__tag">{d.method}</span>
        {d.method === 'DatasetSplit' && (
          <span className="surro-node__detail" style={{ fontWeight: 500 }}>
            {hp.data_kind === '3d_field' ? '3D Field' : 'Scalar'}
          </span>
        )}
        {d.method !== 'DatasetSplit' && preview && <span className="surro-node__detail">{preview}</span>}
      </div>

      {/* ── Input handles ── */}
      {hasCustomPorts ? (
        customInputPorts.map((port, i) => (
          <Handle
            key={`in-${port.id}`}
            type="target"
            position={Position.Left}
            id={port.id}
            className="surro-handle"
            style={{ top: `${((i + 1) / (customInputPorts.length + 1)) * 100}%` }}
            title={port.label}
          />
        ))
      ) : (
        <Handle type="target" position={Position.Left} className="surro-handle" title="in" />
      )}

      {/* ── Output handles ── */}
      {hasCustomPorts ? (
        customOutputPorts.map((port, i) => (
          <Handle
            key={`out-${port.id}`}
            type="source"
            position={Position.Right}
            id={port.id}
            className="surro-handle"
            style={{ top: `${((i + 1) / (customOutputPorts.length + 1)) * 100}%` }}
            title={port.label}
          />
        ))
      ) : (
        <Handle type="source" position={Position.Right} className="surro-handle" title="out" />
      )}
    </div>
  );
}
