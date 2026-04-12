import { useState, useCallback, useRef, useEffect } from 'react';
import Canvas from './components/Canvas';
import Inspector from './components/Inspector';
import OutputPanel from './components/OutputPanel';
import useStore from './store';

// ─── Inline SVG Logo ────────────────────────────────────────────────────────

function Logo({ theme }: { theme: 'dark' | 'light' }) {
  if (theme === 'dark') {
    return (
      <svg className="app__logo-svg" viewBox="0 0 720 200" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="gradCurveDark" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#4F46E5" />
            <stop offset="50%" stopColor="#3B82F6" />
            <stop offset="100%" stopColor="#60A5FA" />
          </linearGradient>
          <linearGradient id="gradFillDark" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#60A5FA" stopOpacity="0.18" />
            <stop offset="100%" stopColor="#60A5FA" stopOpacity="0.02" />
          </linearGradient>
          <filter id="softGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        {[[60,120],[120,80],[180,150],[240,100],[300,130],[360,90],[420,120],[480,80],[540,140],[600,100]].map(([cx,cy],i)=>(
          <circle key={i} cx={cx} cy={cy} r={4} fill="#93C5FD" filter="url(#softGlow)" />
        ))}
        <path fill="url(#gradFillDark)" d="M50 140 C120 80,240 160,360 90 C480 40,600 140,670 100 L670 180 L50 180 Z" />
        <path fill="none" stroke="url(#gradCurveDark)" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" filter="url(#softGlow)"
          d="M50 140 C120 80,240 160,360 90 C480 40,600 140,670 100" />
        <text x="50" y="195" fill="#F1F5F9" fontFamily="system-ui,-apple-system,sans-serif" fontWeight="500" fontSize="46" letterSpacing="1.5">SurroMod</text>
      </svg>
    );
  }
  return (
    <svg className="app__logo-svg" viewBox="0 0 720 200" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="gradCurveLight" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#0F172A" />
          <stop offset="100%" stopColor="#2563EB" />
        </linearGradient>
        <linearGradient id="gradFillLight" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#2563EB" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#0F172A" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[[60,120],[120,80],[180,150],[240,100],[300,130],[360,90],[420,120],[480,80],[540,140],[600,100]].map(([cx,cy],i)=>(
        <circle key={i} cx={cx} cy={cy} r={4} fill="#0F172A" />
      ))}
      <path fill="url(#gradFillLight)" d="M50 140 C120 80,240 160,360 90 C480 40,600 140,670 100 L670 180 L50 180 Z" />
      <path fill="none" stroke="url(#gradCurveLight)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"
        d="M50 140 C120 80,240 160,360 90 C480 40,600 140,670 100" />
      <text x="50" y="195" fill="#0F172A" fontFamily="system-ui,-apple-system,sans-serif" fontWeight="500" fontSize="46" letterSpacing="1.5">SurroMod</text>
    </svg>
  );
}

// ─── Tab Bar (Chrome-style) ─────────────────────────────────────────────────

function TabBar() {
  const { tabs, activeTabId, addTab, removeTab, setActiveTab, renameTab } = useStore();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');

  const startRename = (id: string, name: string) => {
    setEditingId(id);
    setEditValue(name);
  };

  const commitRename = () => {
    if (editingId && editValue.trim()) {
      renameTab(editingId, editValue.trim());
    }
    setEditingId(null);
  };

  return (
    <div className="tab-bar">
      {tabs.map((tab) => (
        <div
          key={tab.id}
          className={`tab-bar__tab ${tab.id === activeTabId ? 'tab-bar__tab--active' : ''}`}
          onClick={() => setActiveTab(tab.id)}
          onDoubleClick={() => startRename(tab.id, tab.name)}
        >
          {editingId === tab.id ? (
            <input
              className="tab-bar__input"
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onBlur={commitRename}
              onKeyDown={(e) => {
                if (e.key === 'Enter') commitRename();
                if (e.key === 'Escape') setEditingId(null);
              }}
              autoFocus
              onClick={(e) => e.stopPropagation()}
            />
          ) : (
            <span className="tab-bar__label">{tab.name}</span>
          )}
          {tabs.length > 1 && (
            <button
              className="tab-bar__close"
              onClick={(e) => {
                e.stopPropagation();
                removeTab(tab.id);
              }}
            >
              ×
            </button>
          )}
        </div>
      ))}
      <button className="tab-bar__add" onClick={addTab} title="New tab">
        +
      </button>
    </div>
  );
}

// ─── Settings Modal ─────────────────────────────────────────────────────────

function SettingsModal() {
  const { theme, setTheme, settingsOpen, toggleSettings, globalSeed, setGlobalSeed } = useStore();

  if (!settingsOpen) return null;

  return (
    <div className="settings-overlay" onClick={toggleSettings}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="settings-modal__header">
          <h3>Settings</h3>
          <button className="settings-modal__close" onClick={toggleSettings}>
            ×
          </button>
        </div>
        <div className="settings-modal__body">
          <label className="settings-modal__field">
            <span>Theme</span>
            <select
              value={theme}
              onChange={(e) => setTheme(e.target.value as 'dark' | 'light')}
            >
              <option value="dark">Dark</option>
              <option value="light">Light</option>
            </select>
          </label>
          <label className="settings-modal__field">
            <span>Random Seed</span>
            <div className="settings-modal__control-row">
              <input
                type="number"
                min={0}
                step={1}
                placeholder="None (non-deterministic)"
                value={globalSeed ?? ''}
                onChange={(e) => {
                  const val = e.target.value.trim();
                  setGlobalSeed(val === '' ? null : parseInt(val, 10));
                }}
                className="settings-modal__seed-input"
              />
              {globalSeed !== null && (
                <button
                  className="settings-modal__clear-seed"
                  onClick={() => setGlobalSeed(null)}
                  title="Clear seed"
                >
                  Clear
                </button>
              )}
            </div>
          </label>
          <p className="settings-modal__hint">
            Set a seed to make all training runs reproducible.
          </p>
        </div>
      </div>
    </div>
  );
}

// ─── Resizable Inspector wrapper ────────────────────────────────────────────

function ResizableInspector() {
  const { inspectorWidth, setInspectorWidth } = useStore();
  const isResizing = useRef(false);

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      isResizing.current = true;

      const onMouseMove = (ev: MouseEvent) => {
        if (!isResizing.current) return;
        const newWidth = window.innerWidth - ev.clientX;
        setInspectorWidth(newWidth);
      };

      const onMouseUp = () => {
        isResizing.current = false;
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };

      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    },
    [setInspectorWidth],
  );

  return (
    <>
      <div className="resize-handle" onMouseDown={onMouseDown} />
      <div style={{ width: inspectorWidth, minWidth: 200, flexShrink: 0 }}>
        <Inspector />
      </div>
    </>
  );
}

// ─── App ────────────────────────────────────────────────────────────────────

export default function App() {
  const { theme, toggleSettings, runPipeline, stopEverything, pipelineRunning, pipelineError, toggleOutputPanel, showConnectionLabels, toggleConnectionLabels } = useStore();
  const loadInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (showConnectionLabels) {
      document.documentElement.classList.add('show-connection-labels');
    } else {
      document.documentElement.classList.remove('show-connection-labels');
    }
  }, [showConnectionLabels]);

  const handleSave = useCallback(() => {
    const name = prompt('Workflow name:', useStore.getState().tabs.find(
      (t) => t.id === useStore.getState().activeTabId
    )?.name ?? 'My Workflow');
    if (name) {
      useStore.getState().saveCurrentWorkflow(name);
    }
  }, []);

  const handleLoad = useCallback(() => {
    loadInputRef.current?.click();
  }, []);

  const handleLoadFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      useStore.getState().loadWorkflow(file);
    }
    e.target.value = '';
  }, []);

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__logo">
          <Logo theme={theme} />
        </h1>
        <div className="app__spacer" />
        <button
          className={`app__btn${showConnectionLabels ? ' app__btn--active' : ''}`}
          onClick={toggleConnectionLabels}
          title="Toggle connection labels"
        >
          ⌗ Labels
        </button>
        <button
          className="app__btn"
          onClick={handleSave}
          title="Save workflow to file"
        >
          💾 Save
        </button>
        <button
          className="app__btn"
          onClick={handleLoad}
          title="Load workflow from file"
        >
          📂 Load
        </button>
        <input
          ref={loadInputRef}
          type="file"
          accept=".pkl,.pickle"
          style={{ display: 'none' }}
          onChange={handleLoadFile}
        />
        <div className="app__separator" />
        <button
          className="app__btn app__btn--run"
          title="Run pipeline"
          onClick={runPipeline}
          disabled={pipelineRunning}
        >
          {pipelineRunning ? '⏳ Running…' : '▶ Run'}
        </button>
        <button
          className="app__btn app__btn--stop"
          title="Stop all running operations"
          onClick={stopEverything}
          disabled={!pipelineRunning}
        >
          ⏹ Stop
        </button>
        <button className="app__btn app__btn--debug" title="Debug (step-by-step)">
          🐛 Debug
        </button>
        <button
          className="app__btn"
          onClick={toggleOutputPanel}
          title="Toggle output panel"
        >
          📋 Output
        </button>
        <button
          className="app__btn app__btn--settings"
          onClick={toggleSettings}
          title="Settings"
        >
          ⚙
        </button>
      </header>

      <TabBar />

      <div className="app__content">
        <main className="app__main">
          <Canvas />
          <ResizableInspector />
        </main>

        <OutputPanel />
      </div>

      {pipelineError && (
        <div className="pipeline-error" onClick={() => useStore.getState().clearResults()}>
          <span>⚠ Pipeline Error: {pipelineError}</span>
          <button className="pipeline-error__close">×</button>
        </div>
      )}

      <SettingsModal />
    </div>
  );
}
