import React from 'react';

interface ErrorDisplayProps {
  error: string;
  onDismiss?: () => void;
}

export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ error, onDismiss }) => {
  return (
    <div className="fixed top-4 right-4 bg-red-600 text-white p-4 rounded-lg shadow-lg z-50 max-w-md">
      <div className="flex items-start space-x-3">
        <span className="text-xl">⚠️</span>
        <div className="flex-1">
          <h4 className="font-semibold mb-1">Error</h4>
          <p className="text-sm">{error}</p>
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="text-white hover:text-gray-200 text-lg leading-none"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
}; 