import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface TransparencyPreferences {
  // Transparency settings
  transparencyLevel: 'none' | 'summary' | 'detailed';
  autoExpandThinking: boolean;
  thinkingPosition: 'inline' | 'sidebar';
  showConfidence: boolean;
  showAlternatives: boolean;
  
  // Actions
  setTransparencyLevel: (level: 'none' | 'summary' | 'detailed') => void;
  setAutoExpand: (expand: boolean) => void;
  setPosition: (position: 'inline' | 'sidebar') => void;
  setShowConfidence: (show: boolean) => void;
  setShowAlternatives: (show: boolean) => void;
  resetToDefaults: () => void;
}

const defaultPreferences = {
  transparencyLevel: 'summary' as const,
  autoExpandThinking: false,
  thinkingPosition: 'inline' as const,
  showConfidence: true,
  showAlternatives: true,
};

export const usePreferenceStore = create<TransparencyPreferences>()(
  persist(
    (set) => ({
      ...defaultPreferences,
      
      setTransparencyLevel: (level) => {
        set({ transparencyLevel: level });
        console.log('[PreferenceStore] Transparency level set to:', level);
      },
      
      setAutoExpand: (expand) => {
        set({ autoExpandThinking: expand });
      },
      
      setPosition: (position) => {
        set({ thinkingPosition: position });
      },
      
      setShowConfidence: (show) => {
        set({ showConfidence: show });
      },
      
      setShowAlternatives: (show) => {
        set({ showAlternatives: show });
      },
      
      resetToDefaults: () => {
        set(defaultPreferences);
        console.log('[PreferenceStore] Reset to default preferences');
      }
    }),
    {
      name: 'loopsmith-transparency-preferences',
      version: 1,
      partialize: (state) => ({
        transparencyLevel: state.transparencyLevel,
        autoExpandThinking: state.autoExpandThinking,
        thinkingPosition: state.thinkingPosition,
        showConfidence: state.showConfidence,
        showAlternatives: state.showAlternatives,
      })
    }
  )
);