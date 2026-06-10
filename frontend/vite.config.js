import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Replicates tech_ui_server.py's proxying to the orchestrator (chat-ui, 8005):
//   /api/{path}  -> {ORCH}/{path}        (strip the /api layer)
//   /auth/{path} -> {ORCH}/auth/{path}   (preserved, incl. SSE index-events)
const ORCH = process.env.ORCHESTRATOR_URL || 'http://localhost:8005';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: ORCH,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/auth': {
        target: ORCH,
        changeOrigin: true,
      },
    },
  },
});
