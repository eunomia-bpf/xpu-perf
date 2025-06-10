import React, { Suspense, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import * as THREE from 'three';
import { useFlameGraphStore } from '@/stores/flameGraphStore';
import { FlameGraphDataLoader } from '@/utils/flameDataLoader';
import { FlameBlocks } from './FlameBlocks';
import { ThreadLabel } from './ThreadLabel';

interface FlameGraph3DProps {
  className?: string;
}

export const FlameGraph3D: React.FC<FlameGraph3DProps> = ({ className }) => {
  const controlsRef = useRef<any>(null);
  const {
    data,
    config,
    setHoveredBlock,
    updateStats
  } = useFlameGraphStore();

  // Update statistics when data changes
  useEffect(() => {
    if (Object.keys(data).length > 0) {
      const loader = new FlameGraphDataLoader();
      const stats = loader.getSummaryStats(data);
      updateStats(stats);
    }
  }, [data, updateStats]);

  // Handle auto-rotation
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = config.autoRotate;
      controlsRef.current.autoRotateSpeed = 0.5;
    }
  }, [config.autoRotate]);

  const handlePointerMissed = () => {
    setHoveredBlock(null);
  };

  const cameraPosition = new THREE.Vector3(60, 40, 60);

  return (
    <div className={className} style={{ width: '100%', height: '100vh', background: '#111111' }}>
      <Canvas
        camera={{ 
          position: cameraPosition,
          fov: 75,
          near: 0.1,
          far: 1000
        }}
        shadows
        onPointerMissed={handlePointerMissed}
      >
        <Suspense fallback={null}>
          {/* Lighting */}
          <ambientLight intensity={0.8} color="#404040" />
          <directionalLight
            position={[100, 100, 50]}
            intensity={1.0}
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
            shadow-camera-near={0.5}
            shadow-camera-far={500}
            shadow-camera-left={-100}
            shadow-camera-right={100}
            shadow-camera-top={100}
            shadow-camera-bottom={-100}
          />
          <pointLight position={[-50, 50, 50]} intensity={0.6} />
          
          {/* Environment for better lighting */}
          <Environment preset="city" />
          
          {/* Controls */}
          <OrbitControls
            ref={controlsRef}
            enableDamping
            dampingFactor={0.05}
            maxDistance={200}
            minDistance={10}
            autoRotate={config.autoRotate}
            autoRotateSpeed={0.5}
          />
          
          {/* Flame Graph Content */}
          <FlameGraphContent />
        </Suspense>
      </Canvas>
    </div>
  );
};

const FlameGraphContent: React.FC = () => {
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