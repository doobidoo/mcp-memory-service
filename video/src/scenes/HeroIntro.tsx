import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';
import { colors } from '../styles/colors';
import { fontFamilies } from '../styles/fonts';
import { MemoryParticles3D } from '../components/MemoryParticles3D';
import { MemoryLogo } from '../components/MemoryLogo';

/**
 * HeroIntro Scene (0-15s / 0-450 frames)
 *
 * Timeline:
 * - 0-3s: Fade in from black → Logo appears center
 * - 3-8s: Title animation: "MCP Memory Service" (word-by-word, 100ms stagger)
 * - 8-12s: Tagline: "Semantic Memory for AI Applications" with glow
 * - 12-15s: 3D particle background - glowing spheres connecting with lines
 *
 * Enhancements:
 * - Professional fonts (JetBrains Mono, Inter)
 * - SVG logo with pulse animation
 * - Fine-tuned spring configs for smoother motion
 * - Enhanced 3D particles (40 nodes, dynamic movement)
 */
export const HeroIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Logo fade-in (0-90 frames / 0-3s) - smoother transition
  const logoOpacity = spring({
    frame,
    fps,
    from: 0,
    to: 1,
    config: {
      damping: 25,
      stiffness: 80,
    },
    durationInFrames: 90,
  });

  // Title words with stagger - improved spring config
  const titleWords = ['MCP', 'Memory', 'Service'];

  // Tagline fade-in (240-330 frames / 8-11s) - longer, smoother fade
  const taglineOpacity = spring({
    frame: Math.max(0, frame - 240),
    fps,
    from: 0,
    to: 1,
    config: {
      damping: 30,
      stiffness: 60,
    },
  });

  // Tagline glow pulse
  const glowIntensity = interpolate(
    Math.sin(frame / 20),
    [-1, 1],
    [20, 40]
  );

  // 3D particles start at frame 360 (12s) - smoother fade
  const particlesOpacity = spring({
    frame: Math.max(0, frame - 360),
    fps,
    from: 0,
    to: 0.7, // Slightly transparent so text remains visible
    config: {
      damping: 40,
      stiffness: 50,
    },
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: colors.background,
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      {/* 3D Particle Background */}
      {frame >= 360 && (
        <div style={{ opacity: particlesOpacity, position: 'absolute', inset: 0 }}>
          <MemoryParticles3D frame={frame - 360} />
        </div>
      )}

      {/* Content Container */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 48,
          zIndex: 1,
        }}
      >
        {/* Logo */}
        <MemoryLogo size={140} opacity={logoOpacity} />

        {/* Title with staggered animation */}
        <div style={{ display: 'flex', gap: 28, alignItems: 'center' }}>
          {titleWords.map((word, index) => {
            const startFrame = 90 + index * 35; // 3s start + slightly longer stagger

            // Improved spring animation
            const progress = spring({
              frame: Math.max(0, frame - startFrame),
              fps,
              config: {
                damping: 18,
                stiffness: 90,
                overshootClamping: false, // Allow slight overshoot for bounce
              },
            });

            const slideY = interpolate(progress, [0, 1], [60, 0]);
            const wordOpacity = interpolate(progress, [0, 0.3, 1], [0, 0.8, 1]);

            return (
              <h1
                key={word}
                style={{
                  fontFamily: fontFamilies.mono,
                  fontSize: 72,
                  fontWeight: 'bold',
                  color: colors.textPrimary,
                  margin: 0,
                  transform: `translateY(${slideY}px)`,
                  opacity: wordOpacity,
                  letterSpacing: '-0.02em',
                }}
              >
                {word}
              </h1>
            );
          })}
        </div>

        {/* Tagline */}
        <p
          style={{
            fontFamily: fontFamilies.sans,
            fontSize: 32,
            fontWeight: '400',
            color: colors.textSecondary,
            margin: 0,
            opacity: taglineOpacity,
            textShadow: `0 0 ${glowIntensity}px ${colors.aiml.from}60`,
            letterSpacing: '0.02em',
          }}
        >
          Semantic Memory for AI Applications
        </p>
      </div>
    </AbsoluteFill>
  );
};
