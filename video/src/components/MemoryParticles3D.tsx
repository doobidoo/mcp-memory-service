import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface Particle {
  position: [number, number, number];
  velocity: [number, number, number];
  color: string;
  size: number;
}

interface MemoryParticles3DProps {
  frame: number;
}

/**
 * Individual particle mesh with animation
 */
const AnimatedParticle: React.FC<{
  particle: Particle;
  frame: number;
}> = ({ particle, frame }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (!meshRef.current) return;

    // Gentle floating motion
    const time = frame / 30; // Convert frames to seconds
    meshRef.current.position.x =
      particle.position[0] + Math.sin(time * 0.5 + particle.position[0]) * 0.3;
    meshRef.current.position.y =
      particle.position[1] + Math.cos(time * 0.4 + particle.position[1]) * 0.3;
    meshRef.current.position.z =
      particle.position[2] + Math.sin(time * 0.3 + particle.position[2]) * 0.2;

    // Gentle pulse
    const scale = 1 + Math.sin(time * 2 + particle.position[0]) * 0.1;
    meshRef.current.scale.setScalar(scale);
  });

  return (
    <mesh ref={meshRef} position={particle.position}>
      <sphereGeometry args={[particle.size, 32, 32]} />
      <meshStandardMaterial
        color={particle.color}
        emissive={particle.color}
        emissiveIntensity={0.6}
        metalness={0.3}
        roughness={0.4}
      />
    </mesh>
  );
};

/**
 * Connection lines between particles
 */
const Connections: React.FC<{
  particles: Particle[];
  frame: number;
}> = ({ particles, frame }) => {
  const connections = useMemo(() => {
    const maxDistance = 7;
    const lines: [Particle, Particle, number][] = [];

    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const p1 = particles[i];
        const p2 = particles[j];

        const dx = p1.position[0] - p2.position[0];
        const dy = p1.position[1] - p2.position[1];
        const dz = p1.position[2] - p2.position[2];
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (distance < maxDistance) {
          lines.push([p1, p2, distance]);
        }
      }
    }

    return lines;
  }, [particles]);

  return (
    <>
      {connections.map(([p1, p2, distance], i) => {
        // Opacity based on distance
        const opacity = THREE.MathUtils.lerp(0.6, 0.1, distance / 7);

        // Animated opacity pulsing
        const time = frame / 30;
        const pulse = Math.sin(time * 2 + i * 0.5) * 0.2 + 0.8;
        const finalOpacity = opacity * pulse;

        const points = [
          new THREE.Vector3(...p1.position),
          new THREE.Vector3(...p2.position),
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        return (
          <line key={`line-${i}`} geometry={geometry}>
            <lineBasicMaterial
              color="#8B5CF6"
              opacity={finalOpacity}
              transparent
              linewidth={2}
            />
          </line>
        );
      })}
    </>
  );
};

/**
 * 3D particle system showing memory nodes connecting with lines
 * Enhanced with dynamic movement and better distribution
 */
export const MemoryParticles3D: React.FC<MemoryParticles3DProps> = ({ frame }) => {
  // Generate particles with better distribution (more particles, clustered)
  const particles = useMemo<Particle[]>(() => {
    const count = 40; // Increased from 25
    const particles: Particle[] = [];
    const colors = ['#8B5CF6', '#EC4899', '#3B82F6', '#10B981'];

    // Create 4 clusters
    const clusterCenters = [
      [-8, 5, 3],
      [8, -4, 2],
      [-5, -6, -4],
      [6, 6, -3],
    ];

    for (let i = 0; i < count; i++) {
      const clusterIndex = Math.floor(i / (count / 4));
      const center = clusterCenters[clusterIndex];

      // Random offset from cluster center
      const spread = 5;
      particles.push({
        position: [
          center[0] + (Math.random() - 0.5) * spread,
          center[1] + (Math.random() - 0.5) * spread,
          center[2] + (Math.random() - 0.5) * spread,
        ],
        velocity: [
          (Math.random() - 0.5) * 0.01,
          (Math.random() - 0.5) * 0.01,
          (Math.random() - 0.5) * 0.01,
        ],
        color: colors[clusterIndex],
        size: 0.2 + Math.random() * 0.15,
      });
    }

    return particles;
  }, []);

  return (
    <Canvas camera={{ position: [0, 0, 30], fov: 50 }}>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[15, 15, 15]} intensity={0.6} color="#8B5CF6" />
      <pointLight position={[-15, -15, -15]} intensity={0.4} color="#EC4899" />
      <pointLight position={[0, 15, -15]} intensity={0.3} color="#3B82F6" />

      {/* Particles with animation */}
      {particles.map((particle, i) => (
        <AnimatedParticle key={i} particle={particle} frame={frame} />
      ))}

      {/* Connection lines */}
      <Connections particles={particles} frame={frame} />

      {/* Camera controls with slow auto-rotation */}
      <OrbitControls
        enableZoom={false}
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.3}
      />
    </Canvas>
  );
};
