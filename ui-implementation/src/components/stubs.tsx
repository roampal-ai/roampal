import React from 'react';

// Stub components to get app loading

export const ImageAnalysis: React.FC<any> = ({ onClose }) => {
  return (
    <div className="p-4 bg-zinc-900 rounded-lg">
      <h3>Image Analysis</h3>
      <button onClick={onClose}>Close</button>
    </div>
  );
};

export const ShardCreationModal: React.FC<any> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
      <div className="bg-zinc-900 p-4 rounded-lg">
        <h3>Create Shard</h3>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
};

export const ShardBooksModal: React.FC<any> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
      <div className="bg-zinc-900 p-4 rounded-lg">
        <h3>Shard Books</h3>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
};

// Re-export BookProcessorModal as ShardManagementModal for backwards compatibility
export { BookProcessorModal as ShardManagementModal } from './BookProcessorModal';

export const LoginModal: React.FC<any> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
      <div className="bg-zinc-900 p-4 rounded-lg">
        <h3>Login</h3>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
};

export const ProcessingBubble: React.FC<any> = () => {
  return <div className="animate-pulse">Processing...</div>;
};

export const VoiceConversationModal: React.FC<any> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
      <div className="bg-zinc-900 p-4 rounded-lg">
        <h3>Voice Conversation</h3>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
};

export const VoiceSettingsModal: React.FC<any> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
      <div className="bg-zinc-900 p-4 rounded-lg">
        <h3>Voice Settings</h3>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
};

export const ProfileSettings: React.FC<any> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
      <div className="bg-zinc-900 p-4 rounded-lg">
        <h3>Profile Settings</h3>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
};

export const SleepMode: React.FC<any> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
      <div className="bg-zinc-900 p-4 rounded-lg">
        <h3>Sleep Mode</h3>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
};

export default ProfileSettings;