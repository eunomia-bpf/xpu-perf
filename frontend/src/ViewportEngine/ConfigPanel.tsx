import React from 'react';

interface ConfigFieldProps {
  label: string;
  type: 'range' | 'select' | 'checkbox' | 'number' | 'text';
  value: any;
  onChange: (value: any) => void;
  options?: Array<{ value: any; label: string }>;
  min?: number;
  max?: number;
  step?: number;
}

const ConfigField: React.FC<ConfigFieldProps> = ({ 
  label, 
  type, 
  value, 
  onChange, 
  options, 
  min, 
  max, 
  step 
}) => {
  const renderInput = () => {
    switch (type) {
      case 'range':
        return (
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-300">{min}</span>
              <span className="text-xs text-white font-medium">{value}</span>
              <span className="text-xs text-gray-300">{max}</span>
            </div>
            <input
              type="range"
              min={min}
              max={max}
              step={step}
              value={value}
              onChange={(e) => onChange(Number(e.target.value))}
              className="w-full h-1 bg-gray-600 rounded appearance-none cursor-pointer slider"
            />
          </div>
        );
      
      case 'select':
        return (
          <select
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="w-full bg-gray-600 text-white rounded px-2 py-1 text-xs border border-gray-500 focus:border-blue-400 focus:outline-none"
          >
            {options?.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        );
      
      case 'checkbox':
        return (
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={value}
              onChange={(e) => onChange(e.target.checked)}
              className="w-4 h-4 text-blue-600 bg-gray-600 border-gray-500 rounded focus:ring-blue-500 focus:ring-2"
            />
            <span className="text-xs text-gray-300">{label}</span>
          </label>
        );
      
      case 'number':
        return (
          <input
            type="number"
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
            min={min}
            max={max}
            step={step}
            className="w-full bg-gray-600 text-white rounded px-2 py-1 text-xs border border-gray-500 focus:border-blue-400 focus:outline-none"
          />
        );
      
      case 'text':
        return (
          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="w-full bg-gray-600 text-white rounded px-2 py-1 text-xs border border-gray-500 focus:border-blue-400 focus:outline-none"
          />
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="space-y-1">
      {type !== 'checkbox' && (
        <label className="text-xs text-gray-300">{label}:</label>
      )}
      {renderInput()}
    </div>
  );
};

interface ConfigPanelProps {
  title?: string;
  fields: ConfigFieldProps[];
  className?: string;
}

export const ConfigPanel: React.FC<ConfigPanelProps> = ({ 
  title, 
  fields, 
  className 
}) => {
  return (
    <div className={`bg-gray-800 border-t border-gray-700 p-3 ${className || ''}`}>
      {title && (
        <h4 className="text-sm font-medium text-white mb-3">{title}</h4>
      )}
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {fields.map((field, index) => (
          <ConfigField key={index} {...field} />
        ))}
      </div>
    </div>
  );
}; 