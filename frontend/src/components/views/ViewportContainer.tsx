import React, { useState } from 'react';
import { FlameGraph3DView } from './FlameGraph3DView';
import { FlameGraph2DView } from './FlameGraph2DView';
import { DataTableView } from './DataTableView';
import { LineChartView } from './LineChartView';

export type ViewType = '3d-flame' | '2d-flame' | 'data-table' | 'line-chart';

interface ViewportContainerProps {
  className?: string;
}

export const ViewportContainer: React.FC<ViewportContainerProps> = ({ className }) => {
  const [activeView, setActiveView] = useState<ViewType>('3d-flame');

  // Listen for view changes from the control panel
  React.useEffect(() => {
    const handleViewChange = (event: Event) => {
      const customEvent = event as CustomEvent<ViewType>;
      setActiveView(customEvent.detail);
    };

    // Listen for view type changes
    const radioButtons = document.querySelectorAll('input[name="viewType"]');
    radioButtons.forEach(radio => {
      radio.addEventListener('change', (e) => {
        const target = e.target as HTMLInputElement;
        if (target.checked) {
          setActiveView(target.value as ViewType);
        }
      });
    });

    return () => {
      radioButtons.forEach(radio => {
        radio.removeEventListener('change', () => {});
      });
    };
  }, []);

  const renderView = () => {
    switch (activeView) {
      case '3d-flame':
        return <FlameGraph3DView />;
      case '2d-flame':
        return <FlameGraph2DView />;
      case 'data-table':
        return <DataTableView />;
      case 'line-chart':
        return <LineChartView />;
      default:
        return <FlameGraph3DView />;
    }
  };

  return (
    <div className={`flex-1 flex flex-col ${className || ''}`}>
      {/* Current View */}
      <div className="flex-1 relative">
        {renderView()}
      </div>
    </div>
  );
}; 