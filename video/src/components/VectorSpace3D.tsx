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
  // Statische Nodes - keine Animation mehr
  return (
    <mesh position={node.position}>
      <sphereGeometry args={[node.size, 24, 24]} />
      <meshStandardMaterial
        color={node.color}
        emissive={node.color}
        emissiveIntensity={1.2}
        metalness={0.6}
        roughness={0.2}
      />
    </mesh>
  );
};

const Connections: React.FC<{ nodes: VectorNode[]; time: number }> = ({ nodes, time }) => {
  const connections = useMemo(() => {
    const lines: [VectorNode, VectorNode][] = [];
    const maxDistance = 10; // Increased to match larger clusters (was 6)

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

        return (
          <line key={i} geometry={geometry}>
            <lineBasicMaterial
              color={n1.color}
              opacity={0.4}
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

  // Seeded random for consistent node positions across frames
  const seededRandom = (seed: number) => {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  };

  const nodes = useMemo<VectorNode[]>(() => {
    const count = 45; // More nodes
    const nodes: VectorNode[] = [];
    // Brighter, more contrasting colors
    const colors = ['#A78BFA', '#F472B6', '#60A5FA'];
    const clusterCenters = [
      [-6, 0, 1],   // links
      [1, -3, -1],  // mitte: nach rechts, nach unten
      [7, -1, 2],   // rechts: passt in Box
    ];

    for (let i = 0; i < count; i++) {
      const cluster = Math.floor(i / (count / 3));
      const center = clusterCenters[cluster];

      // Use seeded random so positions stay consistent across frames
      nodes.push({
        position: [
          center[0] + (seededRandom(i * 3) - 0.5) * 8,
          center[1] + (seededRandom(i * 3 + 1) - 0.5) * 8,
          center[2] + (seededRandom(i * 3 + 2) - 0.5) * 8,
        ],
        color: colors[cluster],
        size: 0.4 + seededRandom(i * 7) * 0.3,
        cluster,
      });
    }

    return nodes;
  }, []);

  return (
    <Canvas camera={{ position: [0, 0, 26], fov: 50 }}>
      {/* Brighter ambient light for better visibility */}
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1.2} color="#A78BFA" />
      <pointLight position={[-10, -10, -10]} intensity={1.0} color="#F472B6" />
      <pointLight position={[0, 10, -10]} intensity={0.8} color="#60A5FA" />

      {/* Sanfte Kamera-Rotation für subtile Bewegung */}
      <group rotation={[0, time * 0.05, 0]}>
        {nodes.map((node, i) => (
          <VectorNode key={i} node={node} time={time} />
        ))}

        <Connections nodes={nodes} time={time} />
      </group>
    </Canvas>
  );
};
