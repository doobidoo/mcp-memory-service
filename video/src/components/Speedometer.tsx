/**
 * Speedometer component for performance visualization
 * Shows animated gauge with needle and value
 */

import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';

interface SpeedometerProps {
  maxValue: number;
  currentValue: number;
  label: string;
  color: string;
  unit?: string;
  startFrame?: number;
}

export const Speedometer: React.FC<SpeedometerProps> = ({
  maxValue,
  currentValue,
  label,
  color,
  unit = 'ms',
  startFrame = 0,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const relativeFrame = Math.max(0, frame - startFrame);

  // Needle animation with overshoot
  const progress = spring({
    frame: relativeFrame,
    fps,
    config: {
      damping: 15,
      stiffness: 100,
      overshootClamping: false,
    },
  });

  // Needle angle (-120° to 120°, 240° total range)
  const targetAngle = ((currentValue / maxValue) * 240) - 120;
  const angle = interpolate(progress, [0, 1], [-120, targetAngle]);

  // Value count-up
  const displayValue = Math.floor(progress * currentValue);

  // Glow pulse
  const glowIntensity = interpolate(
    Math.sin(frame / 15),
    [-1, 1],
    [0.3, 0.6]
  );

  return (
    <svg width="400" height="320" viewBox="0 0 400 320">
      <defs>
        <linearGradient id="arcGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="1" />
        </linearGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>

      {/* Background arc */}
      <path
        d="M 50 250 A 150 150 0 0 1 350 250"
        fill="none"
        stroke="#1E293B"
        strokeWidth="20"
        strokeLinecap="round"
      />

      {/* Progress arc */}
      <path
        d="M 50 250 A 150 150 0 0 1 350 250"
        fill="none"
        stroke="url(#arcGradient)"
        strokeWidth="20"
        strokeLinecap="round"
        strokeDasharray={`${progress * 471} 471`}
        filter="url(#glow)"
        style={{
          opacity: glowIntensity + 0.4,
        }}
      />

      {/* Tick marks */}
      {[0, 25, 50, 75, 100].map((tick) => {
        const tickAngle = (tick / 100) * 240 - 120;
        const rad = (tickAngle * Math.PI) / 180;
        const x1 = 200 + Math.sin(rad) * 130;
        const y1 = 250 - Math.cos(rad) * 130;
        const x2 = 200 + Math.sin(rad) * 150;
        const y2 = 250 - Math.cos(rad) * 150;

        return (
          <line
            key={tick}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke="#475569"
            strokeWidth="3"
            strokeLinecap="round"
          />
        );
      })}

      {/* Needle */}
      <g transform={`rotate(${angle} 200 250)`}>
        <line
          x1="200"
          y1="250"
          x2="200"
          y2="110"
          stroke={color}
          strokeWidth="4"
          strokeLinecap="round"
          filter="url(#glow)"
        />
        <circle cx="200" cy="250" r="14" fill={color} filter="url(#glow)" />
        <circle cx="200" cy="250" r="8" fill="#0F172A" />
      </g>

      {/* Center value display */}
      <text
        x="200"
        y="240"
        textAnchor="middle"
        fontSize="56"
        fontWeight="bold"
        fill="#F8FAFC"
        fontFamily="Inter"
      >
        {displayValue}
      </text>
      <text
        x="200"
        y="270"
        textAnchor="middle"
        fontSize="24"
        fill="#94A3B8"
        fontFamily="Inter"
      >
        {unit}
      </text>

      {/* Label */}
      <text
        x="200"
        y="305"
        textAnchor="middle"
        fontSize="18"
        fill="#94A3B8"
        fontFamily="Inter"
      >
        {label}
      </text>
    </svg>
  );
};
