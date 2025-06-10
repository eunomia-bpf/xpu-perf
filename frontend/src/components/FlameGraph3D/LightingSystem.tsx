import React from 'react';
import { Environment } from '@react-three/drei';

export const LightingSystem: React.FC = () => {
  return (
    <>
      {/* Ambient lighting */}
      <ambientLight intensity={0.8} color="#404040" />
      
      {/* Main directional light with shadows */}
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
      
      {/* Point light for additional illumination */}
      <pointLight position={[-50, 50, 50]} intensity={0.6} />
      
      {/* Environment for better lighting */}
      <Environment preset="city" />
    </>
  );
}; 