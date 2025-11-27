import { useEffect, useState } from 'react';
import { apiFetch } from '../utils/fetch';
import { ROAMPAL_CONFIG } from '../config/roampal';

export const useBackendAutoStart = () => {
  const [backendStatus, setBackendStatus] = useState<'checking' | 'starting' | 'ready' | 'error'>('checking');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const initBackend = async () => {
      const isTauri = window.__TAURI__ !== undefined;

      console.log('[BackendAutoStart] Is Tauri?', isTauri);
      console.log('[BackendAutoStart] API URL:', ROAMPAL_CONFIG.apiUrl);

      if (!isTauri) {
        // In dev mode, assume backend is already running
        setBackendStatus('ready');
        return;
      }

      try {
        // First check if backend is already accessible via HTTP
        console.log('[BackendAutoStart] Checking if backend is already running via HTTP...');
        try {
          const healthCheck = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/health`, {
            method: 'GET',
            signal: AbortSignal.timeout(2000)
          });

          if (healthCheck.ok) {
            console.log('[BackendAutoStart] Backend is already running!');
            setBackendStatus('ready');
            return;
          }
        } catch (e) {
          console.log('[BackendAutoStart] Backend not running, will attempt to start...');
        }

        // Import Tauri invoke dynamically
        console.log('[BackendAutoStart] Importing Tauri API...');
        const { invoke } = await import('@tauri-apps/api/tauri');
        console.log('[BackendAutoStart] Tauri API imported successfully');

        // Try to start backend
        console.log('[BackendAutoStart] Starting backend...');
        setBackendStatus('starting');
        const result = await invoke('start_backend') as string;
        console.log('[BackendAutoStart] Backend start result:', result);

        // Wait for backend to be ready (max 120 seconds for first-time initialization)
        let attempts = 0;
        const maxAttempts = 240; // 120 seconds (500ms intervals)

        const checkInterval = setInterval(async () => {
          attempts++;
          console.log(`[BackendAutoStart] Checking backend readiness (attempt ${attempts}/${maxAttempts})...`);

          try {
            const isReady = await invoke('check_backend') as boolean;

            if (isReady) {
              console.log('[BackendAutoStart] Backend is ready!');
              clearInterval(checkInterval);
              setBackendStatus('ready');
            } else if (attempts >= maxAttempts) {
              console.error('[BackendAutoStart] Backend failed to start within 120 seconds');
              clearInterval(checkInterval);
              setBackendStatus('error');
              setErrorMessage('Backend failed to start within 120 seconds. This may be due to slow first-time initialization. Try restarting the app.');
            }
          } catch (err: any) {
            console.error('[BackendAutoStart] Error checking backend:', err);
            if (attempts >= maxAttempts) {
              clearInterval(checkInterval);
              setBackendStatus('error');
              setErrorMessage(`Failed to check backend status: ${err.message || err}`);
            }
          }
        }, 500);

      } catch (err: any) {
        console.error('[BackendAutoStart] Fatal error:', err);
        setBackendStatus('error');
        setErrorMessage(`Failed to start backend: ${err.message || err}`);
      }
    };

    initBackend();
  }, []);

  return { backendStatus, errorMessage };
};
