/**
 * Memory Service Logo
 * Uses the brain icon with subtle animations
 */

import { interpolate, spring, useCurrentFrame, useVideoConfig, Img, staticFile } from 'remotion';

interface MemoryLogoProps {
  size?: number;
  opacity?: number;
}

export const MemoryLogo: React.FC<MemoryLogoProps> = ({ size = 140, opacity = 1 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Gentle breathing scale animation
  const breathe = spring({
    frame,
    fps,
    config: {
      damping: 200,
      stiffness: 50,
    },
  });

  const scale = interpolate(breathe, [0, 1], [0.98, 1.02]);

  // Subtle pulsing glow intensity
  const glowIntensity = interpolate(
    Math.sin(frame / 25),
    [-1, 1],
    [20, 35]
  );

  return (
    <div
      style={{
        position: 'relative',
        width: size,
        height: size,
        opacity,
      }}
    >
      {/* Subtle purple glow */}
      <div
        style={{
          position: 'absolute',
          inset: -15,
          borderRadius: '50%',
          background: `radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, transparent 70%)`,
          filter: `blur(${glowIntensity}px)`,
        }}
      />

      {/* Brain icon */}
      <Img
        src={staticFile('assets/brain-icon.png')}
        style={{
          width: size,
          height: size,
          transform: `scale(${scale})`,
          filter: `drop-shadow(0 0 ${glowIntensity * 0.4}px rgba(139, 92, 246, 0.6))`,
          borderRadius: '20%',
        }}
      />
    </div>
  );
};
