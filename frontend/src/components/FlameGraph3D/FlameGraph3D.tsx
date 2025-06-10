import React, { Suspense, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useFlameGraphStore } from '@/stores';
import { FlameGraphDataLoader } from '@/utils/flameDataLoader';
import { FlameGraphContent } from './FlameGraphContent';
import { LightingSystem } from './LightingSystem';

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
          {/* Lighting System */}
          <LightingSystem />
          
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