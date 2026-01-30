/**
 * Outro Scene (165-180s / 4950-5400 frames)
 *
 * Timeline:
 * - 0-3s: Fade out previous, fade to dark
 * - 3-7s: GitHub URL appears center
 * - 7-12s: Badges orbit around URL
 * - 12-14s: Final tagline
 * - 14-15s: Fade to black
 */

import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';
import { colors } from '../styles/colors';
import { fontFamilies } from '../styles/fonts';
import { projectData } from '../data/realData';

export const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // GitHub URL fade in
  const urlOpacity = spring({
    frame: Math.max(0, frame - 90),
    fps,
    from: 0,
    to: 1,
    config: {
      damping: 20,
      stiffness: 60,
    },
  });

  // Badges
  const badges = [
    { icon: '⭐', text: 'GitHub Stars', value: '1200+' },
    { icon: '✅', text: 'Tests', value: projectData.testCount },
    { icon: '🚀', text: 'Version', value: `v${projectData.version}` },
    { icon: '💾', text: 'Storage', value: 'Hybrid' },
    { icon: '📦', text: 'PyPI', value: 'Published' },
  ];

  // Tagline
  const taglineOpacity = interpolate(frame, [360, 390], [0, 1], {
    extrapolateRight: 'clamp',
  });

  // Final fade to black
  const fadeToBlack = interpolate(frame, [420, 450], [0, 1], {
    extrapolateRight: 'clamp',
  });

  // Logo pulse
  const logoPulse = 1 + Math.sin(frame / 20) * 0.05;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: colors.background,
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      {/* GitHub URL */}
      <div
        style={{
          opacity: urlOpacity,
          textAlign: 'center',
          marginBottom: 80,
        }}
      >
        <div
          style={{
            fontSize: 48,
            fontFamily: fontFamilies.mono,
            color: colors.textPrimary,
            fontWeight: 'bold',
            textShadow: '0 0 40px rgba(139, 92, 246, 0.3)',
          }}
        >
          github.com/doobidoo/mcp-memory-service
        </div>
      </div>

      {/* Orbiting badges */}
      {frame >= 210 && (
        <div
          style={{
            position: 'relative',
            width: 600,
            height: 600,
          }}
        >
          {badges.map((badge, i) => {
            const badgeFrame = Math.max(0, frame - (210 + i * 10));

            const scale = spring({
              frame: badgeFrame,
              fps,
              from: 0,
              to: 1,
              config: {
                damping: 15,
                stiffness: 100,
              },
            });

            // Circular orbit
            const angle = ((frame - 210) / 60) * Math.PI * 2 + (i * Math.PI * 2) / badges.length;
            const radius = 200;
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;

            return (
              <div
                key={i}
                style={{
                  position: 'absolute',
                  left: '50%',
                  top: '50%',
                  transform: `translate(calc(-50% + ${x}px), calc(-50% + ${y}px)) scale(${scale})`,
                  padding: '16px 24px',
                  backgroundColor: colors.cardBg,
                  borderRadius: 16,
                  border: `2px solid ${colors.aiml.from}40`,
                  boxShadow: `0 0 20px ${colors.aiml.from}30`,
                  textAlign: 'center',
                  minWidth: 140,
                }}
              >
                <div style={{ fontSize: 28, marginBottom: 8 }}>{badge.icon}</div>
                <div
                  style={{
                    fontSize: 20,
                    fontWeight: 'bold',
                    color: colors.textPrimary,
                    fontFamily: fontFamilies.sans,
                    marginBottom: 4,
                  }}
                >
                  {badge.value}
                </div>
                <div
                  style={{
                    fontSize: 13,
                    color: colors.textSecondary,
                    fontFamily: fontFamilies.sans,
                  }}
                >
                  {badge.text}
                </div>
              </div>
            );
          })}

          {/* Center logo */}
          <div
            style={{
              position: 'absolute',
              left: '50%',
              top: '50%',
              transform: `translate(-50%, -50%) scale(${logoPulse})`,
              fontSize: 80,
            }}
          >
            🧠
          </div>
        </div>
      )}

      {/* Tagline */}
      {frame >= 360 && (
        <div
          style={{
            position: 'absolute',
            bottom: 120,
            left: 0,
            right: 0,
            textAlign: 'center',
            opacity: taglineOpacity,
          }}
        >
          <div
            style={{
              fontSize: 32,
              fontFamily: fontFamilies.sans,
              color: colors.textSecondary,
              lineHeight: 1.6,
            }}
          >
            Semantic Memory
            <span style={{ color: colors.aiml.from, margin: '0 16px' }}>•</span>
            Persistent Context
            <span style={{ color: colors.aiml.from, margin: '0 16px' }}>•</span>
            Built for AI
          </div>
        </div>
      )}

      {/* Fade to black */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundColor: colors.background,
          opacity: fadeToBlack,
          pointerEvents: 'none',
        }}
      />
    </AbsoluteFill>
  );
};
