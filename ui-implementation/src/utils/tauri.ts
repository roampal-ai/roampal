// Tauri utils stub

export const isTauri = () => {
  return typeof window !== 'undefined' && window.__TAURI__ !== undefined;
};

export const getPlatformBadge = () => {
  if (!isTauri()) return null;
  return 'Desktop';
};