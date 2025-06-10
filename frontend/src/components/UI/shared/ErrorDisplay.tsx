import React from 'react';

interface ErrorDisplayProps {
  error: string;
  className?: string;
}

export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ error, className }) => {
  return (
    <div className={`absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-red-400 text-base z-50 bg-black/90 px-10 py-5 rounded-lg text-center border border-red-400 max-w-4/5 ${className || ''}`}>
      Error: {error}
    </div>
  );
}; 