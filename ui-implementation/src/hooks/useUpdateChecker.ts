/**
 * Update Checker Hook - v0.2.8
 * Checks for available updates on app startup and provides update notification state.
 */
import { useEffect, useState } from 'react';
import { open } from '@tauri-apps/api/shell';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface UpdateInfo {
  available: boolean;
  version?: string;
  notes?: string;
  download_url?: string;
  is_critical?: boolean;
  current_version?: string;
}

export function useUpdateChecker() {
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [dismissed, setDismissed] = useState(false);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    // Check on mount, with 5s delay to not block startup
    const timer = setTimeout(async () => {
      setChecking(true);
      try {
        const response = await fetch(`${ROAMPAL_CONFIG.apiUrl}/api/check-update`);
        const data = await response.json();
        if (data.available) {
          setUpdateInfo(data);
        }
      } catch (error) {
        // Fail silently - update check is non-critical
        console.debug('[UPDATE] Check failed:', error);
      } finally {
        setChecking(false);
      }
    }, 5000);

    return () => clearTimeout(timer);
  }, []);

  const dismiss = () => setDismissed(true);

  const openDownload = async () => {
    if (updateInfo?.download_url) {
      try {
        // Use Tauri's shell API for native browser open
        await open(updateInfo.download_url);
      } catch {
        // Fallback to window.open
        window.open(updateInfo.download_url, '_blank');
      }
    }
  };

  return {
    updateInfo: dismissed ? null : updateInfo,
    checking,
    dismiss,
    openDownload
  };
}
