import React from 'react';
import { InfoPanel } from './InfoPanel';

interface InfoPanelLayoutProps {
  className?: string;
}

export const InfoPanelLayout: React.FC<InfoPanelLayoutProps> = ({ className }) => {
  return (
    <div className={`absolute top-4 left-4 z-30 ${className || ''}`}>
      <InfoPanel className="!relative !top-0 !left-0" />
    </div>
  );
}; 