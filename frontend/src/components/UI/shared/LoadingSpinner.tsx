import React from 'react';

interface LoadingSpinnerProps {
  message?: string;
  className?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  message = 'Loading...', 
  className 
}) => {
  return (
    <div className={`absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-white text-lg z-50 bg-black/80 px-10 py-5 rounded-lg text-center backdrop-blur-md ${className || ''}`}>
      {message}
    </div>
  );
}; 