import { useState, useCallback } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { InputNodeData } from '../../types';
import { categoryColor } from '../../utils';
import { uploadFile } from '../../api';
import useStore from '../../store';

export default function InputNode({ id, data, selected }: NodeProps) {
  const d = data as unknown as InputNodeData;
  const accent = categoryColor.input;
  const updateNodeData = useStore((s) => s.updateNodeData);

  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);

  const fileName = d.fileName || '';

  const handleFileDrop = useCallback(
    async (file: File) => {
      setUploading(true);
      try {
        const result = await uploadFile(file);
        if (result.ok && result.fileId) {
          const struct = result.structure;
          let columns: string[] = result.columns ?? [];
          if (struct?.format === 'h5' && struct.groups) {
            columns = [];
            for (const [grpPath, grpInfo] of Object.entries(struct.groups)) {
              for (const dsName of Object.keys(grpInfo.datasets ?? {})) {
                const full = grpPath === '/' ? `/${dsName}` : `${grpPath}/${dsName}`;
                columns.push(full);
              }
            }
          }
          updateNodeData(id, {
            source: result.fileId,
            fileName: result.originalName ?? file.name,
            columns,
            features: [],
            labels: [],
            structure: struct,
          } as Partial<InputNodeData>);
        }
      } catch {
        console.error('Upload failed');
      } finally {
        setUploading(false);
      }
    },
    [id, updateNodeData],
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    // Only handle native file drops, not sidebar node drags
    if (e.dataTransfer.types.includes('Files')) {
      e.preventDefault();
      e.stopPropagation();
      e.dataTransfer.dropEffect = 'copy';
      setDragOver(true);
    }
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOver(false);
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        handleFileDrop(files[0]);
      }
    },
    [handleFileDrop],
  );

  const isTemporalPointCloud = d.inputKind === '3d_field' && d.hyperparams?.format_mode === 'Temporal Point Cloud Field';
  const isPointCloudSurfaceMask = d.inputKind === '3d_geometry' && d.hyperparams?.format_mode === 'Point Cloud Surface Mask';
  const isBatchMode = isTemporalPointCloud && Boolean(d.hyperparams?.batch_dir);

  // Determine output ports for specialised formats
  const outputPorts: { id: string; label: string }[] = [];
  if (isTemporalPointCloud) {
    // Named ports for explicitly connected data flows
    outputPorts.push({ id: 'pos', label: 'pos' });
    outputPorts.push({ id: 'velocity_in', label: 'vel_in' });
    outputPorts.push({ id: 'surface_info', label: 'surface' });
    // General data output carries full dataset (samples, vel_out, t, etc.)
    outputPorts.push({ id: 'data', label: 'data' });
  } else if (isPointCloudSurfaceMask) {
    outputPorts.push({ id: 'geometry_mask', label: 'mask' });
    outputPorts.push({ id: 'surface_points', label: 'surf_pts' });
  }
  const hasMultiOutput = outputPorts.length > 0;

  return (
    <div
      className={`surro-node surro-node--compact ${selected ? 'selected' : ''} ${dragOver ? 'surro-node--drop-active' : ''}`}
      style={{ borderColor: accent, background: accent }}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <div className="surro-node__header surro-node__header--compact" style={{ background: accent }}>
        <span className="surro-node__title">{d.label}</span>
      </div>
      {isTemporalPointCloud && (
        <div className="surro-node__tag" style={{ fontSize: '0.65rem', opacity: 0.85, padding: '1px 4px' }}>
          {isBatchMode ? 'GRAM Batch' : 'Temporal Point Cloud'}
        </div>
      )}
      {isPointCloudSurfaceMask && (
        <div className="surro-node__tag" style={{ fontSize: '0.65rem', opacity: 0.85, padding: '1px 4px' }}>
          Surface Mask
        </div>
      )}
      {uploading && (
        <div className="surro-node__upload-status">Uploading\u2026</div>
      )}
      {!uploading && isBatchMode && (
        <div className="surro-node__file-badge" title={d.hyperparams?.batch_dir as string}>
          {(() => {
            const dir = String(d.hyperparams?.batch_dir ?? '');
            const parts = dir.split('/').filter(Boolean);
            const short = parts.length > 0 ? parts[parts.length - 1] : dir;
            return short.length > 18 ? short.slice(0, 16) + '\u2026' : short;
          })()}
        </div>
      )}
      {!uploading && !isBatchMode && fileName && (
        <div className="surro-node__file-badge" title={fileName}>
          {fileName.length > 20 ? fileName.slice(0, 18) + '\u2026' : fileName}
        </div>
      )}
      {!uploading && !isBatchMode && !fileName && (
        <div className="surro-node__drop-hint">
          Drop file here
        </div>
      )}
      {isBatchMode && (
        <div className="surro-node__detail" style={{ fontSize: '0.6rem', opacity: 0.7, padding: '0 4px' }}>
          {d.hyperparams?.max_files ? `max ${d.hyperparams.max_files} files` : 'all files'}
          {d.hyperparams?.max_points ? ` \u00b7 ${d.hyperparams.max_points} pts` : ''}
        </div>
      )}
      {hasMultiOutput ? (
        outputPorts.map((port, i) => (
          <Handle
            key={port.id}
            type="source"
            position={Position.Right}
            id={port.id}
            className="surro-handle"
            style={{ top: `${((i + 1) / (outputPorts.length + 1)) * 100}%` }}
            title={port.label}
          />
        ))
      ) : (
        <Handle type="source" position={Position.Right} className="surro-handle" title="data" />
      )}
    </div>
  );
}
