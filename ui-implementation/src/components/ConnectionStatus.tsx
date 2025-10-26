import React from 'react';

interface ConnectionStatusProps {
  status: 'connecting' | 'connected' | 'disconnected' | 'error';
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ status }) => {
  const getStatusConfig = () => {
    switch (status) {
      case 'connected':
        return {
          color: 'bg-green-500',
          text: 'Connected',
          animation: '',
        };
      case 'connecting':
        return {
          color: 'bg-yellow-500',
          text: 'Connecting...',
          animation: 'animate-pulse',
        };
      case 'disconnected':
        return {
          color: 'bg-zinc-500',
          text: 'Disconnected',
          animation: '',
        };
      case 'error':
        return {
          color: 'bg-red-500',
          text: 'Connection Error',
          animation: '',
        };
    }
  };
  
  const config = getStatusConfig();
  
  return (
    <div className="flex items-center gap-2">
      <div className={`w-2 h-2 rounded-full ${config.color} ${config.animation}`} />
      <span className="text-xs text-zinc-400">{config.text}</span>
    </div>
  );
};