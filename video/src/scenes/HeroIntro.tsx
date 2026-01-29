import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';
import { colors } from '../styles/colors';
import { MemoryParticles3D } from '../components/MemoryParticles3D';

/**
 * HeroIntro Scene (0-15s / 0-450 frames)
 *
 * Timeline:
 * - 0-3s: Fade in from black → Logo appears center
 * - 3-8s: Title animation: "MCP Memory Service" (word-by-word, 100ms stagger)
 * - 8-12s: Tagline: "Semantic Memory for AI Applications" with glow
 * - 12-15s: 3D particle background - glowing spheres connecting with lines
 */
export const HeroIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Logo fade-in (0-90 frames / 0-3s)
  const logoOpacity = interpolate(frame, [0, 90], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Title words with stagger
  const titleWords = ['MCP', 'Memory', 'Service'];

  // Tagline fade-in (240-300 frames / 8-10s)
  const taglineOpacity = interpolate(frame, [240, 300], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // 3D particles start at frame 360 (12s)
  const particlesOpacity = interpolate(frame, [360, 420], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
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
          gap: 40,
          zIndex: 1,
        }}
      >
        {/* Logo (placeholder - will be replaced with actual logo) */}
        <div
          style={{
            opacity: logoOpacity,
            width: 120,
            height: 120,
            backgroundColor: colors.aiml.from,
            borderRadius: 24,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 64,
            fontWeight: 'bold',
            color: colors.textPrimary,
            boxShadow: `0 0 60px ${colors.aiml.from}80`,
          }}
        >
          M
        </div>

        {/* Title with staggered animation */}
        <div style={{ display: 'flex', gap: 24, alignItems: 'center' }}>
          {titleWords.map((word, index) => {
            const startFrame = 90 + index * 30; // 3s start + 1s stagger per word
            const endFrame = startFrame + 45; // 1.5s slide duration

            const slideY = spring({
              frame: Math.max(0, frame - startFrame),
              fps,
              config: {
                damping: 15,
                stiffness: 100,
              },
              from: 50,
              to: 0,
            });

            const wordOpacity = interpolate(frame, [startFrame, endFrame], [0, 1], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            });

            return (
              <h1
                key={word}
                style={{
                  fontFamily: 'sans-serif',
                  fontSize: 72,
                  fontWeight: 'bold',
                  color: colors.textPrimary,
                  margin: 0,
                  transform: `translateY(${slideY}px)`,
                  opacity: wordOpacity,
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
            fontFamily: 'sans-serif',
            fontSize: 28,
            color: colors.textSecondary,
            margin: 0,
            opacity: taglineOpacity,
            textShadow: `0 0 20px ${colors.aiml.from}40`,
          }}
        >
          Semantic Memory for AI Applications
        </p>
      </div>
    </AbsoluteFill>
  );
};
