import React from 'react';
import { useFlameGraphStore } from '@/stores';
import { FlameGraphDataLoader } from '@/utils/flameDataLoader';
import { FlameBlocks } from './FlameBlocks';
import { ThreadLabel } from './ThreadLabel';

export const FlameGraphContent: React.FC = () => {
  const { data, config } = useFlameGraphStore();
  const loader = new FlameGraphDataLoader();

  if (Object.keys(data).length === 0) {
    return null;
  }

  let zOffset = 0;
  const threads = Object.keys(data);

  return (
    <group>
      {threads.map((threadName) => {
        const threadData = data[threadName];
        const filteredData = threadData?.filter(entry => entry.count >= config.minCount) || [];
        
        if (filteredData.length === 0) {
          return null;
        }

        const tree = loader.buildFlameTree(filteredData);
        const currentZOffset = zOffset;
        zOffset += config.zSpacing;

        return (
          <group key={threadName} position={[0, 0, currentZOffset]}>
            <FlameBlocks
              tree={tree}
              threadName={threadName}
              maxDepth={config.maxDepth}
              colorSchemeIndex={config.colorSchemeIndex}
            />
            <ThreadLabel
              threadName={threadName}
              position={[-10, -2, 0]}
            />
          </group>
        );
      })}
    </group>
  );
}; 