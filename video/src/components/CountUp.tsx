/**
 * CountUp animation component
 * Animates numbers with spring physics
 */

import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';

interface CountUpProps {
  from?: number;
  to: number;
  suffix?: string;
  prefix?: string;
  fontSize?: number;
  color?: string;
  glowColor?: string;
  delay?: number; // Frames to delay
  label?: string;
  labelSize?: number;
}

export const CountUp: React.FC<CountUpProps> = ({
  from = 0,
  to,
  suffix = '',
  prefix = '',
  fontSize = 64,
  color = '#F8FAFC',
  glowColor,
  delay = 0,
  label,
  labelSize = 20,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const relativeFrame = Math.max(0, frame - delay);

  const progress = spring({
    frame: relativeFrame,
    fps,
    config: {
      damping: 20,
      stiffness: 80,
    },
  });

  const value = interpolate(progress, [0, 1], [from, to]);

  // Format large numbers with commas
  const formatted = Math.floor(value).toLocaleString();

  // Scale animation
  const scale = spring({
    frame: relativeFrame,
    fps,
    from: 0.8,
    to: 1,
    config: {
      damping: 12,
      stiffness: 100,
    },
  });

  // Glow pulse
  const glowIntensity = interpolate(
    Math.sin(frame / 20),
    [-1, 1],
    [20, 40]
  );

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 8,
      }}
    >
      <div
        style={{
          fontSize,
          fontWeight: 'bold',
          color,
          fontFamily: 'Inter',
          transform: `scale(${scale})`,
          textShadow: glowColor
            ? `0 0 ${glowIntensity}px ${glowColor}80`
            : undefined,
        }}
      >
        {prefix}{formatted}{suffix}
      </div>
      {label && (
        <div
          style={{
            fontSize: labelSize,
            color: '#FFFFFF',
            fontFamily: 'Inter',
            fontWeight: '600',
            textShadow: '0 2px 10px rgba(0, 0, 0, 0.9)',
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            padding: '8px 20px',
            borderRadius: 8,
            border: '1px solid rgba(255, 255, 255, 0.2)',
          }}
        >
          {label}
        </div>
      )}
    </div>
  );
};
