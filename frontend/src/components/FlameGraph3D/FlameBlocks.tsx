import React, { useState } from 'react';
import * as THREE from 'three';
import { FlameTreeNode } from '@/types/flame.types';
import { useFlameGraphStore } from '@/stores/flameGraphStore';
import { getColorForFunction } from '@/utils/colorSchemes';

interface FlameBlocksProps {
  tree: Record<string, FlameTreeNode>;
  threadName: string;
  maxDepth: number;
  colorSchemeIndex: number;
}

export const FlameBlocks: React.FC<FlameBlocksProps> = ({
  tree,
  threadName,
  maxDepth,
  colorSchemeIndex
}) => {
  return (
    <group>
      <FlameBlocksRecursive
        tree={tree}
        threadName={threadName}
        x={0}
        y={0}
        depth={0}
        maxDepth={maxDepth}
        colorSchemeIndex={colorSchemeIndex}
      />
    </group>
  );
};

interface FlameBlocksRecursiveProps {
  tree: Record<string, FlameTreeNode>;
  threadName: string;
  x: number;
  y: number;
  depth: number;
  maxDepth: number;
  colorSchemeIndex: number;
}

const FlameBlocksRecursive: React.FC<FlameBlocksRecursiveProps> = ({
  tree,
  threadName,
  x,
  y,
  depth,
  maxDepth,
  colorSchemeIndex
}) => {
  if (depth >= maxDepth) {
    return null;
  }

  let currentX = x;
  const sortedFunctions = Object.keys(tree).sort((a, b) => (tree[b]?.count || 0) - (tree[a]?.count || 0));

  return (
    <>
      {sortedFunctions.map((funcName) => {
        const data = tree[funcName];
        if (!data) return null;
        
        const width = Math.max(data.count / 50, 0.8);
        const blockX = currentX + width / 2;
        
        const result = (
          <group key={`${funcName}-${depth}-${currentX}`}>
            <FlameBlock
              funcName={funcName}
              count={data.count}
              threadName={threadName}
              position={[blockX, y, 0]}
              width={width}
              depth={depth}
              colorSchemeIndex={colorSchemeIndex}
            />
            
            {/* Render children */}
            {Object.keys(data.children).length > 0 && (
              <FlameBlocksRecursive
                tree={data.children}
                threadName={threadName}
                x={currentX}
                y={y + 1.0}
                depth={depth + 1}
                maxDepth={maxDepth}
                colorSchemeIndex={colorSchemeIndex}
              />
            )}
          </group>
        );
        
        currentX += width;
        return result;
      })}
    </>
  );
};

interface FlameBlockProps {
  funcName: string;
  count: number;
  threadName: string;
  position: [number, number, number];
  width: number;
  depth: number;
  colorSchemeIndex: number;
}

const FlameBlock: React.FC<FlameBlockProps> = ({
  funcName,
  count,
  threadName,
  position,
  width,
  depth,
  colorSchemeIndex
}) => {
  const [hovered, setHovered] = useState(false);
  const { setHoveredBlock } = useFlameGraphStore();
  
  const color = getColorForFunction(funcName, colorSchemeIndex);
  const originalColor = new THREE.Color(color);
  const hoverColor = new THREE.Color(0xffffff);
  
  const height = 0.8;
  const blockDepth = 0.8;

  const handlePointerEnter = () => {
    setHovered(true);
    
    setHoveredBlock({
      funcName,
      count,
      threadName,
      depth,
      originalColor,
      width
    });
  };

  const handlePointerLeave = () => {
    setHovered(false);
  };

  return (
    <mesh
      position={position}
      castShadow
      receiveShadow
      onPointerEnter={handlePointerEnter}
      onPointerLeave={handlePointerLeave}
    >
      <boxGeometry args={[width, height, blockDepth]} />
      <meshLambertMaterial
        color={hovered ? hoverColor : originalColor}
        transparent
        opacity={hovered ? 1.0 : 0.85}
      />
    </mesh>
  );
}; 