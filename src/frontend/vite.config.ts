import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: true,
    proxy: {
      '/api/logs/stream': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // SSE needs these to avoid buffering
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['cache-control'] = 'no-cache';
            proxyRes.headers['x-accel-buffering'] = 'no';
          });
        },
      },
      '/api/': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        timeout: 1800000,  // 30 min — HP tuning is long-running
      },
    },
  },
});
