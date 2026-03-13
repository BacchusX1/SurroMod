/**
 * HPTunerAnalytics – Interactive analytics plots for agent-based HP tuning.
 *
 * Plots:
 * 1. Metric (train + holdout) vs total model parameters
 * 2. Metric (train + holdout) over iteration
 * 3. Metric vs each numeric hyperparameter (scatter)
 * 4. 3D surface: metric over PCA-reduced numeric HP space (2 principal components)
 */
import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import type { HPTuningIterationResult, HPTuneParam } from '../types';

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Filter out penalty / failed iterations (extreme sentinel values). */
function validIterations(history: HPTuningIterationResult[]): HPTuningIterationResult[] {
  return history.filter((h) => Math.abs(h.score) < 1e9);
}

/** Common dark-theme layout defaults for Plotly. */
function baseDarkLayout(title: string, extra: Partial<Plotly.Layout> = {}): Partial<Plotly.Layout> {
  return {
    title: { text: title, font: { size: 13, color: '#e2e8f0' } },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(30,30,40,0.55)',
    font: { color: '#cbd5e1', size: 11 },
    margin: { l: 52, r: 20, t: 36, b: 44 },
    legend: { font: { size: 10 }, bgcolor: 'rgba(0,0,0,0)' },
    ...extra,
  };
}

// ─── PCA (minimal, no deps) ────────────────────────────────────────────────

/** Center + PCA → project N×D matrix to N×2. Returns {projected, explainedLabels}. */
function pcaProject(
  matrix: number[][],
): { projected: number[][]; pc1Label: string; pc2Label: string } | null {
  const N = matrix.length;
  if (N < 3) return null;
  const D = matrix[0].length;
  if (D < 2) return null;

  // Center
  const mean = new Array(D).fill(0);
  for (let i = 0; i < N; i++) for (let j = 0; j < D; j++) mean[j] += matrix[i][j];
  for (let j = 0; j < D; j++) mean[j] /= N;
  const centered = matrix.map((row) => row.map((v, j) => v - mean[j]));

  // Standardise (unit variance per column)
  const std = new Array(D).fill(0);
  for (let i = 0; i < N; i++) for (let j = 0; j < D; j++) std[j] += centered[i][j] ** 2;
  for (let j = 0; j < D; j++) std[j] = Math.sqrt(std[j] / N) || 1;
  for (let i = 0; i < N; i++) for (let j = 0; j < D; j++) centered[i][j] /= std[j];

  // Covariance matrix (D×D)
  const cov: number[][] = Array.from({ length: D }, () => new Array(D).fill(0));
  for (let i = 0; i < N; i++)
    for (let a = 0; a < D; a++)
      for (let b = a; b < D; b++) {
        cov[a][b] += centered[i][a] * centered[i][b];
      }
  for (let a = 0; a < D; a++)
    for (let b = a; b < D; b++) {
      cov[a][b] /= N - 1;
      cov[b][a] = cov[a][b];
    }

  // Power iteration for top-2 eigenvectors (sufficient for visualisation)
  const eigenvectors: number[][] = [];
  const eigenvalues: number[] = [];
  const deflated = cov.map((r) => [...r]);

  for (let ev = 0; ev < Math.min(2, D); ev++) {
    let v = Array.from({ length: D }, () => Math.random() - 0.5);
    let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    v = v.map((x) => x / norm);

    for (let iter = 0; iter < 200; iter++) {
      const next = new Array(D).fill(0);
      for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) next[i] += deflated[i][j] * v[j];
      norm = Math.sqrt(next.reduce((s, x) => s + x * x, 0));
      if (norm < 1e-12) break;
      v = next.map((x) => x / norm);
    }

    const eigenvalue = v.reduce((s, _, i) => {
      const Av_i = deflated[i].reduce((acc, c, j) => acc + c * v[j], 0);
      return s + v[i] * Av_i;
    }, 0);
    eigenvalues.push(eigenvalue);
    eigenvectors.push(v);

    // Deflate
    for (let i = 0; i < D; i++)
      for (let j = 0; j < D; j++)
        deflated[i][j] -= eigenvalue * v[i] * v[j];
  }

  if (eigenvectors.length < 2) return null;

  const totalVar = eigenvalues.reduce((a, b) => a + Math.abs(b), 0) || 1;
  const pct1 = ((Math.abs(eigenvalues[0]) / totalVar) * 100).toFixed(1);
  const pct2 = ((Math.abs(eigenvalues[1]) / totalVar) * 100).toFixed(1);

  // Project
  const projected = centered.map((row) => [
    row.reduce((acc, v, j) => acc + v * eigenvectors[0][j], 0),
    row.reduce((acc, v, j) => acc + v * eigenvectors[1][j], 0),
  ]);

  return { projected, pc1Label: `PC1 (${pct1}%)`, pc2Label: `PC2 (${pct2}%)` };
}

// ─── Component ──────────────────────────────────────────────────────────────

interface Props {
  history: HPTuningIterationResult[];
  tunableParams: HPTuneParam[];
  scoringMetric: string;
}

export default function HPTunerAnalytics({ history, tunableParams, scoringMetric }: Props) {
  const valid = useMemo(() => validIterations(history), [history]);
  const metricLabel = scoringMetric.toUpperCase();
  const hasHoldout = valid.some((h) => h.holdout_score != null);
  const hasNParams = valid.some((h) => h.n_params != null);

  // ── Numeric HP keys that were tuned ─────────────────────────────────────
  const numericKeys = useMemo(
    () =>
      tunableParams
        .filter((p) => p.selected && p.type === 'number')
        .map((p) => p.key),
    [tunableParams],
  );

  // ── Plot 1: Metric vs Model Parameters ──────────────────────────────────
  const paramPlot = useMemo(() => {
    if (!hasNParams) return null;
    const items = valid.filter((h) => h.n_params != null);
    const x = items.map((h) => h.n_params!);
    const yTrain = items.map((h) => h.train_score ?? h.score);
    const yHoldout = items.map((h) => h.holdout_score ?? null);

    const traces: Plotly.Data[] = [
      {
        x,
        y: yTrain,
        mode: 'markers',
        name: 'Train',
        marker: { color: '#6366f1', size: 7 },
        type: 'scatter',
      },
    ];
    if (hasHoldout) {
      traces.push({
        x,
        y: yHoldout as number[],
        mode: 'markers',
        name: 'Holdout',
        marker: { color: '#f59e0b', size: 7 },
        type: 'scatter',
      });
    }
    return {
      data: traces,
      layout: baseDarkLayout(`${metricLabel} vs Model Parameters`, {
        xaxis: { title: { text: 'Total Parameters' }, color: '#94a3b8' },
        yaxis: { title: { text: metricLabel }, color: '#94a3b8' },
      }),
    };
  }, [valid, hasNParams, hasHoldout, metricLabel]);

  // ── Plot 2: Metric over Iteration ───────────────────────────────────────
  const iterPlot = useMemo(() => {
    const x = valid.map((h) => h.iteration);
    const yTrain = valid.map((h) => h.train_score ?? h.score);
    const yHoldout = valid.map((h) => h.holdout_score ?? null);

    // Running best
    const runningBestTrain: number[] = [];
    let bestSoFar = -Infinity;
    for (const v of yTrain) { bestSoFar = Math.max(bestSoFar, v); runningBestTrain.push(bestSoFar); }

    const traces: Plotly.Data[] = [
      {
        x,
        y: yTrain,
        mode: 'lines+markers',
        name: 'Train',
        line: { color: '#6366f1', width: 1.5 },
        marker: { size: 4 },
        type: 'scatter',
      },
      {
        x,
        y: runningBestTrain,
        mode: 'lines',
        name: 'Best (train)',
        line: { color: '#6366f1', width: 2, dash: 'dash' },
        type: 'scatter',
      },
    ];
    if (hasHoldout) {
      traces.push({
        x,
        y: yHoldout as number[],
        mode: 'lines+markers',
        name: 'Holdout',
        line: { color: '#f59e0b', width: 1.5 },
        marker: { size: 4 },
        type: 'scatter',
      });
    }
    return {
      data: traces,
      layout: baseDarkLayout(`${metricLabel} over Iteration`, {
        xaxis: { title: { text: 'Iteration' }, color: '#94a3b8' },
        yaxis: { title: { text: metricLabel }, color: '#94a3b8' },
      }),
    };
  }, [valid, hasHoldout, metricLabel]);

  // ── Plot 3: Metric vs each numeric HP ───────────────────────────────────
  const perHPPlots = useMemo(() => {
    return numericKeys.map((key) => {
      const x = valid.map((h) => Number(h.config[key]));
      const yTrain = valid.map((h) => h.train_score ?? h.score);
      const yHoldout = valid.map((h) => h.holdout_score ?? null);

      const traces: Plotly.Data[] = [
        {
          x,
          y: yTrain,
          mode: 'markers',
          name: 'Train',
          marker: { color: '#6366f1', size: 6, opacity: 0.75 },
          type: 'scatter',
        },
      ];
      if (hasHoldout) {
        traces.push({
          x,
          y: yHoldout as number[],
          mode: 'markers',
          name: 'Holdout',
          marker: { color: '#f59e0b', size: 6, opacity: 0.75 },
          type: 'scatter',
        });
      }
      return {
        key,
        data: traces,
        layout: baseDarkLayout(`${metricLabel} vs ${key.replace(/_/g, ' ')}`, {
          xaxis: { title: { text: key.replace(/_/g, ' ') }, color: '#94a3b8' },
          yaxis: { title: { text: metricLabel }, color: '#94a3b8' },
          showlegend: false,
        }),
      };
    });
  }, [valid, numericKeys, hasHoldout, metricLabel]);

  // ── Plot 4: 3D surface – PCA of numeric HPs → metric ───────────────────
  const pcaPlot = useMemo(() => {
    if (numericKeys.length < 2 || valid.length < 5) return null;

    // Build matrix: each row = [hp1, hp2, ...]
    const matrix = valid.map((h) =>
      numericKeys.map((k) => Number(h.config[k]) || 0),
    );
    const scores = valid.map((h) => h.train_score ?? h.score);

    const pca = pcaProject(matrix);
    if (!pca) return null;

    const pc1 = pca.projected.map((p) => p[0]);
    const pc2 = pca.projected.map((p) => p[1]);

    // ── Build a grid for the surface via simple 2D interpolation ──────
    const GRID = 30;
    const pc1Min = Math.min(...pc1);
    const pc1Max = Math.max(...pc1);
    const pc2Min = Math.min(...pc2);
    const pc2Max = Math.max(...pc2);
    const dx = (pc1Max - pc1Min) / (GRID - 1) || 1;
    const dy = (pc2Max - pc2Min) / (GRID - 1) || 1;

    const xGrid = Array.from({ length: GRID }, (_, i) => pc1Min + i * dx);
    const yGrid = Array.from({ length: GRID }, (_, i) => pc2Min + i * dy);

    // Inverse-distance-weighted interpolation
    const zGrid: number[][] = [];
    for (let iy = 0; iy < GRID; iy++) {
      const row: number[] = [];
      for (let ix = 0; ix < GRID; ix++) {
        const gx = xGrid[ix];
        const gy = yGrid[iy];
        let wSum = 0;
        let vSum = 0;
        for (let k = 0; k < pc1.length; k++) {
          const d = Math.sqrt((pc1[k] - gx) ** 2 + (pc2[k] - gy) ** 2) + 1e-8;
          const w = 1 / (d * d);
          wSum += w;
          vSum += w * scores[k];
        }
        row.push(vSum / wSum);
      }
      zGrid.push(row);
    }

    const traces: Plotly.Data[] = [
      {
        type: 'surface',
        x: xGrid,
        y: yGrid,
        z: zGrid,
        colorscale: 'Viridis',
        opacity: 0.85,
        showscale: true,
        colorbar: { title: metricLabel, titlefont: { size: 11 }, tickfont: { size: 10 } },
      } as any,
      {
        type: 'scatter3d',
        x: pc1,
        y: pc2,
        z: scores,
        mode: 'markers',
        name: 'Observations',
        marker: { size: 3, color: '#f59e0b', opacity: 0.9 },
      } as any,
    ];

    return {
      data: traces,
      layout: {
        ...baseDarkLayout(`${metricLabel} over PCA HP Space`, {
          margin: { l: 0, r: 0, t: 36, b: 0 },
          showlegend: false,
        }),
        scene: {
          xaxis: { title: { text: pca.pc1Label }, color: '#94a3b8', gridcolor: 'rgba(100,100,120,0.3)' },
          yaxis: { title: { text: pca.pc2Label }, color: '#94a3b8', gridcolor: 'rgba(100,100,120,0.3)' },
          zaxis: { title: { text: metricLabel }, color: '#94a3b8', gridcolor: 'rgba(100,100,120,0.3)' },
          bgcolor: 'rgba(30,30,40,0.55)',
        },
      } as any,
    };
  }, [valid, numericKeys, metricLabel]);

  // ── Render ──────────────────────────────────────────────────────────────
  if (valid.length === 0) {
    return <p className="inspector__empty-tab">No valid iterations to analyse.</p>;
  }

  const plotConfig: Partial<Plotly.Config> = {
    displayModeBar: true,
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d'] as any[],
  };

  return (
    <div className="hp-analytics">
      {/* Plot 2: over iteration (most useful → first) */}
      <div className="hp-analytics__chart">
        <Plot
          data={iterPlot.data}
          layout={iterPlot.layout}
          config={plotConfig}
          useResizeHandler
          style={{ width: '100%', height: 280 }}
        />
      </div>

      {/* Plot 1: vs model parameters */}
      {paramPlot && (
        <div className="hp-analytics__chart">
          <Plot
            data={paramPlot.data}
            layout={paramPlot.layout}
            config={plotConfig}
            useResizeHandler
            style={{ width: '100%', height: 280 }}
          />
        </div>
      )}

      {/* Plot 3: per-HP scatter */}
      {perHPPlots.length > 0 && (
        <>
          <div className="hp-analytics__section-title">
            {metricLabel} vs Individual Hyperparameters
            <span className="hp-analytics__caveat">
              (does not account for other HPs – interpret with care)
            </span>
          </div>
          <div className="hp-analytics__grid">
            {perHPPlots.map((p) => (
              <div key={p.key} className="hp-analytics__chart hp-analytics__chart--half">
                <Plot
                  data={p.data}
                  layout={p.layout}
                  config={plotConfig}
                  useResizeHandler
                  style={{ width: '100%', height: 240 }}
                />
              </div>
            ))}
          </div>
        </>
      )}

      {/* Plot 4: PCA 3D surface */}
      {pcaPlot && (
        <div className="hp-analytics__chart">
          <div className="hp-analytics__section-title">
            {metricLabel} over PCA-Reduced HP Space
            <span className="hp-analytics__caveat">
              (numeric HPs projected to 2 principal components)
            </span>
          </div>
          <Plot
            data={pcaPlot.data}
            layout={pcaPlot.layout}
            config={plotConfig}
            useResizeHandler
            style={{ width: '100%', height: 400 }}
          />
        </div>
      )}
    </div>
  );
}
