import React from 'react';

interface LoadingSpinnerProps {
  message?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ message = 'Loading...' }) => {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="text-white text-center">
        <div className="animate-spin text-4xl mb-4">⚡</div>
        <p>{message}</p>
      </div>
    </div>
  );
}; 