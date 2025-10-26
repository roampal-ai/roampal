/**
 * One-time migration of legacy localStorage keys to new format
 * Maps old keys to new per-shard namespacing
 */

export function migrateLegacyStorage() {
  try {
    // Check if migration already done
    if (localStorage.getItem('roampal_migration_v2_complete')) {
      return;
    }

    console.log('[Migration] Starting localStorage migration...');

    // Legacy keys that might exist
    const legacyKeys = [
      'chat_history',
      'active_shard',
      'user_session',
      'roampal-chat', // Old Neural UI key
      'VITE_ENABLE_NEURAL_UI', // Remove old flag
    ];

    // Migrate old chat history
    const oldChat = localStorage.getItem('chat_history');
    if (oldChat) {
      try {
        const data = JSON.parse(oldChat);
        // Convert to new format
        const newData = {
          sessionId: data.session_id || `session-${Date.now()}`,
          activeShard: data.active_shard || 'roampal',
          messages: data.messages || [],
        };
        localStorage.setItem('roampal-chat', JSON.stringify(newData));
        localStorage.removeItem('chat_history');
        console.log('[Migration] Migrated chat history');
      } catch (e) {
        console.warn('[Migration] Failed to migrate chat history:', e);
      }
    }

    // Migrate old Neural UI data if different format
    const oldNeuralData = localStorage.getItem('roampal-chat');
    if (oldNeuralData) {
      try {
        const data = JSON.parse(oldNeuralData);
        // Ensure it has the right structure
        if (!data.state) {
          // Old format, wrap it
          const wrapped = {
            state: data,
            version: 0,
          };
          localStorage.setItem('roampal-chat', JSON.stringify(wrapped));
          console.log('[Migration] Updated Neural UI data format');
        }
      } catch (e) {
        console.warn('[Migration] Failed to update Neural UI data:', e);
      }
    }

    // Clean up old feature flags
    localStorage.removeItem('VITE_ENABLE_NEURAL_UI');
    localStorage.removeItem('VITE_ENABLE_MOCK_MODE');

    // Mark migration complete
    localStorage.setItem('roampal_migration_v2_complete', 'true');
    console.log('[Migration] localStorage migration complete');

  } catch (error) {
    console.error('[Migration] Error during migration:', error);
  }
}

// Run migration on import
migrateLegacyStorage();