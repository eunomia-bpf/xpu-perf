import React, { useMemo } from 'react';
import { Text } from '@react-three/drei';

interface ThreadLabelProps {
  threadName: string;
  position: [number, number, number];
}

export const ThreadLabel: React.FC<ThreadLabelProps> = ({ threadName, position }) => {
  const textConfig = useMemo(() => ({
    fontSize: 1.5,
    color: '#ffffff',
    anchorX: 'left' as const,
    anchorY: 'middle' as const
  }), []);

  return (
    <Text
      position={position}
      {...textConfig}
    >
      {threadName}
    </Text>
  );
}; 