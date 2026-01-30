/**
 * 3D Vector Space visualization for AI/ML scene
 * Enhanced version with better clustering and visual effects
 */

import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface VectorNode {
  position: [number, number, number];
  color: string;
  size: number;
  cluster: number;
}

interface VectorSpace3DProps {
  frame: number;
}

const VectorNode: React.FC<{ node: VectorNode; time: number }> = ({ node, time }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (!meshRef.current) return;

    // Gentle floating
    const offset = time + node.position[0];
    meshRef.current.position.y = node.position[1] + Math.sin(offset * 0.3) * 0.5;

    // Subtle pulse
    const pulse = 1 + Math.sin(offset * 2) * 0.1;
    meshRef.current.scale.setScalar(pulse);
  });

  return (
    <mesh ref={meshRef} position={node.position}>
      <sphereGeometry args={[node.size, 24, 24]} />
      <meshStandardMaterial
        color={node.color}
        emissive={node.color}
        emissiveIntensity={0.8}
        metalness={0.5}
        roughness={0.3}
      />
    </mesh>
  );
};

const Connections: React.FC<{ nodes: VectorNode[]; time: number }> = ({ nodes, time }) => {
  const connections = useMemo(() => {
    const lines: [VectorNode, VectorNode][] = [];
    const maxDistance = 6;

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        if (nodes[i].cluster !== nodes[j].cluster) continue;

        const dx = nodes[i].position[0] - nodes[j].position[0];
        const dy = nodes[i].position[1] - nodes[j].position[1];
        const dz = nodes[i].position[2] - nodes[j].position[2];
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (distance < maxDistance) {
          lines.push([nodes[i], nodes[j]]);
        }
      }
    }

    return lines;
  }, [nodes]);

  return (
    <>
      {connections.map(([n1, n2], i) => {
        const points = [
          new THREE.Vector3(...n1.position),
          new THREE.Vector3(...n2.position),
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const opacity = 0.4 + Math.sin(time + i * 0.2) * 0.2;

        return (
          <line key={i} geometry={geometry}>
            <lineBasicMaterial
              color={n1.color}
              opacity={opacity}
              transparent
              linewidth={1.5}
            />
          </line>
        );
      })}
    </>
  );
};

export const VectorSpace3D: React.FC<VectorSpace3DProps> = ({ frame }) => {
  const time = frame / 30;

  const nodes = useMemo<VectorNode[]>(() => {
    const count = 30;
    const nodes: VectorNode[] = [];
    const colors = ['#8B5CF6', '#EC4899', '#3B82F6'];
    const clusterCenters = [
      [-8, 3, 2],
      [0, -4, -2],
      [8, 2, 3],
    ];

    for (let i = 0; i < count; i++) {
      const cluster = Math.floor(i / (count / 3));
      const center = clusterCenters[cluster];

      nodes.push({
        position: [
          center[0] + (Math.random() - 0.5) * 4,
          center[1] + (Math.random() - 0.5) * 4,
          center[2] + (Math.random() - 0.5) * 4,
        ],
        color: colors[cluster],
        size: 0.2 + Math.random() * 0.15,
        cluster,
      });
    }

    return nodes;
  }, []);

  return (
    <Canvas camera={{ position: [0, 0, 25], fov: 50 }}>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={0.8} color="#8B5CF6" />
      <pointLight position={[-10, -10, -10]} intensity={0.6} color="#EC4899" />
      <pointLight position={[0, 10, -10]} intensity={0.5} color="#3B82F6" />

      {nodes.map((node, i) => (
        <VectorNode key={i} node={node} time={time} />
      ))}

      <Connections nodes={nodes} time={time} />
    </Canvas>
  );
};
