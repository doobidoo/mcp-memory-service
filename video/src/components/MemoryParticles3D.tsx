import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface Particle {
  position: [number, number, number];
  velocity: [number, number, number];
  color: string;
  size: number;
  phase: number;
}

interface MemoryParticles3DProps {
  frame: number;
}

/**
 * Individual particle with dramatic effects
 */
const AnimatedParticle: React.FC<{
  particle: Particle;
  time: number;
}> = ({ particle, time }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const lightRef = useRef<THREE.PointLight>(null);

  useFrame(() => {
    if (!meshRef.current) return;

    // Organic floating motion with multiple sine waves
    const offset = time + particle.phase;
    meshRef.current.position.x =
      particle.position[0] +
      Math.sin(offset * 0.3) * 1.5 +
      Math.cos(offset * 0.7) * 0.5;

    meshRef.current.position.y =
      particle.position[1] +
      Math.cos(offset * 0.4) * 1.8 +
      Math.sin(offset * 0.6) * 0.6;

    meshRef.current.position.z =
      particle.position[2] +
      Math.sin(offset * 0.5) * 1.2 +
      Math.cos(offset * 0.8) * 0.4;

    // Dramatic pulsing
    const pulse = Math.sin(offset * 1.5) * 0.3 + 1.0;
    meshRef.current.scale.setScalar(pulse);

    // Update point light
    if (lightRef.current) {
      lightRef.current.intensity = pulse * 1.5;
    }
  });

  return (
    <group>
      <mesh ref={meshRef} position={particle.position}>
        <sphereGeometry args={[particle.size, 32, 32]} />
        <meshStandardMaterial
          color={particle.color}
          emissive={particle.color}
          emissiveIntensity={1.5}
          metalness={0.9}
          roughness={0.1}
        />
      </mesh>

      {/* Point light for glow effect */}
      <pointLight
        ref={lightRef}
        position={particle.position}
        color={particle.color}
        intensity={1.5}
        distance={6}
        decay={2}
      />
    </group>
  );
};

/**
 * Enhanced connection lines
 */
const Connections: React.FC<{
  particles: Particle[];
  time: number;
}> = ({ particles, time }) => {
  const connections = useMemo(() => {
    const maxDistance = 8;
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
        const baseOpacity = THREE.MathUtils.lerp(0.7, 0.15, distance / 8);
        const pulse = Math.sin(time * 2 + i * 0.3) * 0.3 + 0.7;
        const finalOpacity = baseOpacity * pulse;

        const points = [
          new THREE.Vector3(...p1.position),
          new THREE.Vector3(...p2.position),
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        const color = i % 3 === 0 ? '#8B5CF6' : i % 3 === 1 ? '#EC4899' : '#3B82F6';

        return (
          <line key={`line-${i}`} geometry={geometry}>
            <lineBasicMaterial
              color={color}
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
 * Camera animation
 */
const CameraRig: React.FC<{ time: number }> = ({ time }) => {
  useFrame(({ camera }) => {
    const radius = 30;
    const speed = 0.1;
    camera.position.x = Math.sin(time * speed) * radius * 0.3;
    camera.position.y = Math.cos(time * speed * 0.7) * radius * 0.2;
    camera.position.z = 30 + Math.sin(time * speed * 0.5) * 3;
    camera.lookAt(0, 0, 0);
  });

  return null;
};

/**
 * Enhanced 3D particle system with dramatic lighting
 */
export const MemoryParticles3D: React.FC<MemoryParticles3DProps> = ({ frame }) => {
  const time = frame / 30;

  const particles = useMemo<Particle[]>(() => {
    const count = 50;
    const particles: Particle[] = [];
    const colors = ['#8B5CF6', '#EC4899', '#3B82F6', '#10B981', '#F59E0B'];

    const clusterCenters = [
      [-10, 6, 4],
      [10, -5, 3],
      [-6, -8, -5],
      [8, 7, -4],
      [0, 0, 0],
    ];

    for (let i = 0; i < count; i++) {
      const clusterIndex = Math.floor(i / (count / 5));
      const center = clusterCenters[clusterIndex];
      const size = 0.15 + Math.random() * 0.25;
      const spread = 4 + Math.random() * 3;

      particles.push({
        position: [
          center[0] + (Math.random() - 0.5) * spread,
          center[1] + (Math.random() - 0.5) * spread,
          center[2] + (Math.random() - 0.5) * spread,
        ],
        velocity: [
          (Math.random() - 0.5) * 0.02,
          (Math.random() - 0.5) * 0.02,
          (Math.random() - 0.5) * 0.02,
        ],
        color: colors[clusterIndex],
        size,
        phase: Math.random() * Math.PI * 2,
      });
    }

    return particles;
  }, []);

  return (
    <Canvas
      camera={{ position: [0, 0, 30], fov: 60 }}
      gl={{ antialias: true, alpha: true }}
    >
      <ambientLight intensity={0.2} />
      <pointLight position={[20, 20, 20]} intensity={1.2} color="#8B5CF6" />
      <pointLight position={[-20, -20, -20]} intensity={1.0} color="#EC4899" />
      <pointLight position={[0, 20, -20]} intensity={0.8} color="#3B82F6" />
      <pointLight position={[-15, 0, 15]} intensity={0.6} color="#10B981" />

      <CameraRig time={time} />

      {particles.map((particle, i) => (
        <AnimatedParticle key={i} particle={particle} time={time} />
      ))}

      <Connections particles={particles} time={time} />
    </Canvas>
  );
};
