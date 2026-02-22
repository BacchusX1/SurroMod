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

  const kindIcons: Record<string, string> = {
    scalar: '\u{1F522}',
    time_series: '\u{1F4C8}',
    '2d_field': '\u{1F5FA}\uFE0F',
    '3d_field': '\u{1F310}',
    '2d_geometry': '\u{1F4D0}',
    '3d_geometry': '\u{1F4D0}',
  };

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

  return (
    <div
      className={`surro-node surro-node--compact ${selected ? 'selected' : ''} ${dragOver ? 'surro-node--drop-active' : ''}`}
      style={{ borderColor: accent, background: accent }}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <div className="surro-node__header surro-node__header--compact" style={{ background: accent }}>
        <span className="surro-node__icon">{kindIcons[d.inputKind] ?? '\u{1F4C2}'}</span>
        <span className="surro-node__title">{d.label}</span>
      </div>
      {uploading && (
        <div className="surro-node__upload-status">Uploading\u2026</div>
      )}
      {!uploading && fileName && (
        <div className="surro-node__file-badge" title={fileName}>
          {'\u{1F4C4}'} {fileName.length > 20 ? fileName.slice(0, 18) + '\u2026' : fileName}
        </div>
      )}
      {!uploading && !fileName && (
        <div className="surro-node__drop-hint">
          Drop file here
        </div>
      )}
      <Handle type="source" position={Position.Right} className="surro-handle" />
    </div>
  );
}
