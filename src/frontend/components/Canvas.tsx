import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  SelectionMode,
  type ReactFlowInstance,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import useStore from '../store';
import type { SurroNode, SurroEdge, SurroNodeData, NodeCategory } from '../types';
import { paletteItems, categoryColor } from '../utils';

import InputNode from './nodes/InputNode';
import RegressorNode from './nodes/RegressorNode';
import ClassifierNode from './nodes/ClassifierNode';
import ValidatorNode from './nodes/ValidatorNode';
import FeatureEngineeringNode from './nodes/FeatureEngineeringNode';
import InferenceNode from './nodes/InferenceNode';
import HPTunerNode from './nodes/HPTunerNode';

// Register custom node types (keys must match node.type stored in the store)
const nodeTypes = {
  input: InputNode,
  regressor: RegressorNode,
  classifier: ClassifierNode,
  validator: ValidatorNode,
  feature_engineering: FeatureEngineeringNode,
  inference: InferenceNode,
  hp_tuner: HPTunerNode,
};

// ─── Sidebar ────────────────────────────────────────────────────────────────

function Sidebar() {
  const [collapsed, setCollapsed] = useState<Record<NodeCategory, boolean>>({
    input: false,
    regressor: false,
    classifier: false,
    validator: false,
    feature_engineering: false,
    inference: false,
    hp_tuner: false,
  });

  const toggle = (cat: NodeCategory) =>
    setCollapsed((prev) => ({ ...prev, [cat]: !prev[cat] }));

  const onDragStart = (event: React.DragEvent, data: SurroNodeData) => {
    event.dataTransfer.setData('application/surro-node', JSON.stringify(data));
    event.dataTransfer.effectAllowed = 'move';
  };

  const categories: { key: NodeCategory; title: string }[] = [
    { key: 'input', title: '📂 Data Input' },
    { key: 'feature_engineering', title: '🔧 Feature Engineering' },
    { key: 'regressor', title: '📈 Regressors' },
    { key: 'classifier', title: '🏷️ Classifiers' },
    { key: 'hp_tuner', title: '🎯 HP Tuner' },
    { key: 'validator', title: '✅ Validators' },
    { key: 'inference', title: '🚀 Inference' },
  ];

  return (
    <aside className="sidebar">
      <h2 className="sidebar__title">Building Blocks</h2>
      {categories.map(({ key, title }) => (
        <div key={key} className="sidebar__group">
          <button
            className="sidebar__group-header"
            onClick={() => toggle(key)}
            style={{ borderLeftColor: categoryColor[key] }}
          >
            {title}
            <span className="sidebar__chevron">
              {collapsed[key] ? '▸' : '▾'}
            </span>
          </button>
          {!collapsed[key] && (
            <div className="sidebar__items">
              {paletteItems
                .filter((p) => p.category === key)
                .map((item) => (
                  <div
                    key={item.label}
                    className="sidebar__item"
                    style={{ borderLeftColor: categoryColor[key] }}
                    draggable
                    onDragStart={(e) => onDragStart(e, item.defaultData)}
                  >
                    {item.label}
                  </div>
                ))}
            </div>
          )}
        </div>
      ))}
    </aside>
  );
}

// ─── Flow Canvas ────────────────────────────────────────────────────────────

function FlowCanvas() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [rfInstance, setRfInstance] = useState<ReactFlowInstance<SurroNode, SurroEdge> | null>(null);

  const {
    nodes, edges, onNodesChange, onEdgesChange, onConnect,
    addNode, setSelectedNode, copySelected, pasteClipboard,
    undo, redo,
  } = useStore();

  const onInit = useCallback((instance: ReactFlowInstance<SurroNode, SurroEdge>) => {
    setRfInstance(instance);
  }, []);

  // ── Keyboard shortcuts (Ctrl+C / Ctrl+V / Ctrl+Z / Shift+Ctrl+Z) ──────

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // Skip if user is typing in an input / textarea / select
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      const mod = e.ctrlKey || e.metaKey;

      if (mod && e.key === 'c') {
        e.preventDefault();
        copySelected();
      }

      if (mod && e.key === 'v') {
        e.preventDefault();
        pasteClipboard();
      }

      // Undo: Ctrl+Z (without Shift)
      if (mod && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
      }

      // Redo: Shift+Ctrl+Z  (also handle Ctrl+Y)
      if (mod && ((e.key === 'z' && e.shiftKey) || (e.key === 'Z' && e.shiftKey) || e.key === 'y')) {
        e.preventDefault();
        redo();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [copySelected, pasteClipboard, undo, redo]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const raw = event.dataTransfer.getData('application/surro-node');
      if (!raw || !rfInstance) return;

      const data: SurroNodeData = JSON.parse(raw);
      const position = rfInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      addNode(data, position);
    },
    [rfInstance, addNode],
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: { id: string }) => {
      setSelectedNode(node.id);
    },
    [setSelectedNode],
  );

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  return (
    <div className="canvas-wrapper" ref={reactFlowWrapper}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={onInit}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        deleteKeyCode={['Backspace', 'Delete']}
        selectionOnDrag
        panOnDrag={[1, 2]}
        selectionMode={SelectionMode.Partial}
        multiSelectionKeyCode="Shift"
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={20} size={1} />
        <Controls />
        <MiniMap
          nodeColor={(n) => categoryColor[(n.data as SurroNodeData).category] ?? '#999'}
          maskColor="rgba(0,0,0,0.15)"
        />
      </ReactFlow>
    </div>
  );
}

// ─── Exported Canvas (wrapped in provider) ──────────────────────────────────

export default function Canvas() {
  return (
    <ReactFlowProvider>
      <div className="canvas-container">
        <Sidebar />
        <FlowCanvas />
      </div>
    </ReactFlowProvider>
  );
}
