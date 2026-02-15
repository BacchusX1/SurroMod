import Canvas from './components/Canvas';
import Inspector from './components/Inspector';

export default function App() {
  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__logo">⚙️ SurroMod</h1>
        <span className="app__subtitle">Surrogate Model Builder</span>
      </header>
      <main className="app__main">
        <Canvas />
        <Inspector />
      </main>
    </div>
  );
}
