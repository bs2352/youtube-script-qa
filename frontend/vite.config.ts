import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/summary': 'http://localhost:8080',
      '/qa': 'http://localhost:8080',
      '/transcript': 'http://localhost:8080',
      "/sample": 'http://localhost:8080'
    }
  }
})
