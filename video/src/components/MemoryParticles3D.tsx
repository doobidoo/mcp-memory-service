import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface Particle {
  position: [number, number, number];
  color: string;
  size: number;
}

interface MemoryParticles3DProps {
  frame: number;
}

/**
 * 3D particle system showing memory nodes connecting with lines
 * Used in HeroIntro scene background
 */
export const MemoryParticles3D: React.FC<MemoryParticles3DProps> = ({ frame }) => {
  // Generate particles (20-30 nodes in space)
  const particles = useMemo<Particle[]>(() => {
    const count = 25;
    const particles: Particle[] = [];

    for (let i = 0; i < count; i++) {
      particles.push({
        position: [
          (Math.random() - 0.5) * 20,
          (Math.random() - 0.5) * 15,
          (Math.random() - 0.5) * 15,
        ],
        color: ['#8B5CF6', '#EC4899', '#3B82F6'][Math.floor(Math.random() * 3)],
        size: 0.15 + Math.random() * 0.1,
      });
    }

    return particles;
  }, []);

  // Calculate connections between nearby particles
  const connections = useMemo(() => {
    const maxDistance = 8;
    const lines: [Particle, Particle][] = [];

    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const p1 = particles[i];
        const p2 = particles[j];

        const dx = p1.position[0] - p2.position[0];
        const dy = p1.position[1] - p2.position[1];
        const dz = p1.position[2] - p2.position[2];
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (distance < maxDistance) {
          lines.push([p1, p2]);
        }
      }
    }

    return lines;
  }, [particles]);

  // Slow rotation based on frame
  const rotation = (frame / 30) * 0.002; // Very slow rotation

  return (
    <Canvas camera={{ position: [0, 0, 25], fov: 50 }}>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={0.5} />
      <pointLight position={[-10, -10, -10]} intensity={0.3} />

      <group rotation={[0, rotation, 0]}>
        {/* Render particles as spheres */}
        {particles.map((particle, i) => (
          <mesh key={i} position={particle.position}>
            <sphereGeometry args={[particle.size, 16, 16]} />
            <meshStandardMaterial
              color={particle.color}
              emissive={particle.color}
              emissiveIntensity={0.5}
            />
          </mesh>
        ))}

        {/* Render connections as lines */}
        {connections.map(([p1, p2], i) => {
          const points = [
            new THREE.Vector3(...p1.position),
            new THREE.Vector3(...p2.position),
          ];
          const geometry = new THREE.BufferGeometry().setFromPoints(points);

          return (
            <line key={`line-${i}`} geometry={geometry}>
              <lineBasicMaterial color="#8B5CF6" opacity={0.3} transparent />
            </line>
          );
        })}
      </group>

      {/* Optional: Auto-rotate camera */}
      <OrbitControls
        enableZoom={false}
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.5}
      />
    </Canvas>
  );
};
