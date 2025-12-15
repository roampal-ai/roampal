import React from 'react';
import ReactDOM from 'react-dom/client';
import { ConnectedChat } from './components/ConnectedChat';
import { UpdateBanner } from './components/UpdateBanner';
import { useBackendAutoStart } from './hooks/useBackendAutoStart';
import './index.css';

// App wrapper with backend auto-start
const App = () => {
  const { backendStatus, errorMessage } = useBackendAutoStart();

  if (backendStatus === 'checking' || backendStatus === 'starting') {
    return (
      <div className="h-screen flex items-center justify-center bg-black">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-white mb-2">
            {backendStatus === 'checking' ? 'Checking backend...' : 'Starting Roampal backend...'}
          </h2>
          <p className="text-zinc-400">This may take a few seconds</p>
        </div>
      </div>
    );
  }

  if (backendStatus === 'error') {
    return (
      <div className="h-screen flex items-center justify-center bg-black p-8">
        <div className="text-center max-w-2xl">
          <h1 className="text-2xl font-bold text-red-500 mb-4">Backend Failed to Start</h1>
          <div className="bg-zinc-900 p-4 rounded-lg mb-4 text-left">
            <p className="text-zinc-400 text-sm whitespace-pre-wrap font-mono">{errorMessage}</p>
          </div>
          <div className="text-zinc-500 text-sm mb-4">
            <p>If the backend files are missing, try reinstalling Roampal</p>
          </div>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      <ConnectedChat />
      <UpdateBanner />
    </>
  );
};

// Error boundary
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }
  
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }
  
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('App error:', error, errorInfo);
  }
  
  render() {
    if (this.state.hasError) {
      return (
        <div className="h-screen flex items-center justify-center bg-black">
          <div className="text-center max-w-md">
            <h1 className="text-2xl font-bold text-red-500 mb-4">Something went wrong</h1>
            <p className="text-zinc-400 mb-4">{this.state.error?.message}</p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white"
            >
              Reload App
            </button>
          </div>
        </div>
      );
    }
    
    return this.props.children;
  }
}

// Render app
ReactDOM.createRoot(document.getElementById('root')!).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);