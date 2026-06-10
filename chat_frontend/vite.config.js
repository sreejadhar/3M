import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The DataChat UI is served by the orchestrator (8005) at the same origin and
// calls backend routes at the root (/sources, /sessions, /auth/...). We expose
// them under /api (stripped) and /auth (preserved), and the React API client
// uses the /api prefix.
const ORCH = process.env.ORCHESTRATOR_URL || 'http://localhost:8005';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      '/api': { target: ORCH, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, '') },
      '/auth': { target: ORCH, changeOrigin: true },
    },
  },
});
