/**
 * Memory Service Logo
 * SVG icon representing memory/neural connections
 */

import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';

interface MemoryLogoProps {
  size?: number;
  opacity?: number;
}

export const MemoryLogo: React.FC<MemoryLogoProps> = ({ size = 120, opacity = 1 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Pulse animation
  const pulse = spring({
    frame,
    fps,
    config: {
      damping: 200,
      stiffness: 100,
    },
  });

  const scale = interpolate(pulse, [0, 1], [0.95, 1.05]);

  // Glow intensity
  const glowIntensity = interpolate(
    Math.sin(frame / 15),
    [-1, 1],
    [0.3, 0.7]
  );

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 120 120"
      style={{
        opacity,
        transform: `scale(${scale})`,
        filter: `drop-shadow(0 0 ${30 * glowIntensity}px rgba(139, 92, 246, ${glowIntensity}))`,
      }}
    >
      <defs>
        <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#8B5CF6" />
          <stop offset="50%" stopColor="#EC4899" />
          <stop offset="100%" stopColor="#3B82F6" />
        </linearGradient>
      </defs>

      {/* Outer circle */}
      <circle
        cx="60"
        cy="60"
        r="55"
        fill="none"
        stroke="url(#logoGradient)"
        strokeWidth="3"
        opacity="0.3"
      />

      {/* Memory nodes (small circles) */}
      <circle cx="60" cy="30" r="6" fill="url(#logoGradient)" />
      <circle cx="35" cy="45" r="5" fill="url(#logoGradient)" />
      <circle cx="85" cy="45" r="5" fill="url(#logoGradient)" />
      <circle cx="30" cy="75" r="6" fill="url(#logoGradient)" />
      <circle cx="60" cy="70" r="7" fill="url(#logoGradient)" />
      <circle cx="90" cy="75" r="6" fill="url(#logoGradient)" />
      <circle cx="45" cy="90" r="5" fill="url(#logoGradient)" />
      <circle cx="75" cy="90" r="5" fill="url(#logoGradient)" />

      {/* Connection lines between nodes */}
      <line x1="60" y1="30" x2="35" y2="45" stroke="url(#logoGradient)" strokeWidth="2" opacity="0.5" />
      <line x1="60" y1="30" x2="85" y2="45" stroke="url(#logoGradient)" strokeWidth="2" opacity="0.5" />
      <line x1="35" y1="45" x2="30" y2="75" stroke="url(#logoGradient)" strokeWidth="2" opacity="0.5" />
      <line x1="85" y1="45" x2="90" y2="75" stroke="url(#logoGradient)" strokeWidth="2" opacity="0.5" />
      <line x1="60" y1="70" x2="30" y2="75" stroke="url(#logoGradient)" strokeWidth="2" opacity="0.5" />
      <line x1="60" y1="70" x2="90" y2="75" stroke="url(#logoGradient)" strokeWidth="2" opacity="0.5" />
      <line x1="30" y1="75" x2="45" y2="90" stroke="url(#logoGradient)" strokeWidth="2" opacity="0.5" />
      <line x1="90" y1="75" x2="75" y2="90" stroke="url(#logoGradient)" strokeWidth="2" opacity="0.5" />
      <line x1="60" y1="70" x2="45" y2="90" stroke="url(#logoGradient)" strokeWidth="2" opacity="0.5" />
      <line x1="60" y1="70" x2="75" y2="90" stroke="url(#logoGradient)" strokeWidth="2" opacity="0.5" />

      {/* Center 'M' for Memory */}
      <text
        x="60"
        y="64"
        fontSize="40"
        fontWeight="bold"
        fontFamily="JetBrains Mono, monospace"
        fill="url(#logoGradient)"
        textAnchor="middle"
        dominantBaseline="middle"
      >
        M
      </text>
    </svg>
  );
};
