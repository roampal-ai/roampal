/**
 * Update Banner Component - v0.2.8
 * Shows update notification when a new version is available.
 * Directs users to Gumroad for download.
 */
import React from 'react';
import { useUpdateChecker } from '../hooks/useUpdateChecker';

export const UpdateBanner: React.FC = () => {
  const { updateInfo, dismiss, openDownload } = useUpdateChecker();

  if (!updateInfo) return null;

  return (
    <div
      className={`fixed bottom-4 right-4 z-50 max-w-md p-4 rounded-lg shadow-lg border ${
        updateInfo.is_critical
          ? 'bg-red-900/90 border-red-600'
          : 'bg-zinc-800/95 border-zinc-600'
      }`}
    >
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className={`flex-shrink-0 p-2 rounded-lg ${
          updateInfo.is_critical ? 'bg-red-600/20' : 'bg-blue-600/20'
        }`}>
          <svg
            className={`w-5 h-5 ${updateInfo.is_critical ? 'text-red-400' : 'text-blue-400'}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
            />
          </svg>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-semibold text-zinc-100">
            {updateInfo.is_critical ? 'Critical Update Required' : 'Update Available'}
          </h4>
          <p className="text-xs text-zinc-400 mt-1">
            Roampal {updateInfo.version} is available
            {updateInfo.notes && ` - ${updateInfo.notes}`}
          </p>

          {/* Buttons */}
          <div className="flex items-center gap-2 mt-3">
            <button
              onClick={openDownload}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                updateInfo.is_critical
                  ? 'bg-red-600 hover:bg-red-500 text-white'
                  : 'bg-blue-600 hover:bg-blue-500 text-white'
              }`}
            >
              Download
            </button>
            {!updateInfo.is_critical && (
              <button
                onClick={dismiss}
                className="px-3 py-1.5 text-xs font-medium text-zinc-400 hover:text-zinc-200 rounded-md hover:bg-zinc-700 transition-colors"
              >
                Later
              </button>
            )}
          </div>
        </div>

        {/* Close button (only for non-critical) */}
        {!updateInfo.is_critical && (
          <button
            onClick={dismiss}
            className="flex-shrink-0 p-1 text-zinc-500 hover:text-zinc-300 rounded transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
};
