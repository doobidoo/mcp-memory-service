/**
 * AIMLIntelligence Scene (55-80s / 1650-2400 frames)
 * OPTIMIZED: Cleaner layout, better visibility
 */

import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';
import { colors, gradient } from '../styles/colors';
import { fontFamilies } from '../styles/fonts';
import { VectorSpace3D } from '../components/VectorSpace3D';

export const AIMLIntelligence: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title animation
  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: 'clamp',
  });

  // Features - compact version
  const features = [
    {
      icon: '🧠',
      title: 'Vector Embeddings',
      desc: 'ONNX (384-dim)',
      delay: 240,
    },
    {
      icon: '⭐',
      title: 'Quality Scoring',
      desc: '3-tier system',
      delay: 300,
    },
    {
      icon: '🌙',
      title: 'Consolidation',
      desc: 'Dream-inspired',
      delay: 360,
    },
    {
      icon: '🔗',
      title: 'Graph Building',
      desc: 'Auto relationships',
      delay: 420,
    },
  ];

  return (
    <AbsoluteFill
      style={{
        background: gradient(colors.aiml.from, colors.aiml.to),
      }}
    >
      {/* Main container */}
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          padding: '60px 100px',
        }}
      >
        {/* Title */}
        <h1
          style={{
            fontFamily: fontFamilies.mono,
            fontSize: 72,
            fontWeight: 'bold',
            color: '#FFFFFF',
            margin: 0,
            marginBottom: 60,
            opacity: titleOpacity,
            textShadow: '0 4px 20px rgba(0, 0, 0, 0.5)',
          }}
        >
          AI/ML Intelligence
        </h1>

        {/* 2-Column Layout - VectorSpace dominant */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1.3fr 1fr',
            gap: 80,
            flex: 1,
            alignItems: 'stretch',
          }}
        >
          {/* LEFT: 3D Vector Space - BIGGER with fixed height */}
          <div
            style={{
              position: 'relative',
              backgroundColor: 'rgba(0, 0, 0, 0.4)',
              borderRadius: 20,
              border: `3px solid ${colors.aiml.from}60`,
              overflow: 'hidden',
              minHeight: 0, // Prevent grid blowout
            }}
          >
            {frame >= 60 && (
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  opacity: interpolate(frame, [60, 90], [0, 1], {
                    extrapolateRight: 'clamp',
                  }),
                }}
              >
                <VectorSpace3D frame={frame - 60} />
              </div>
            )}

            {/* Label */}
            {frame >= 120 && (
              <div
                style={{
                  position: 'absolute',
                  bottom: 30,
                  left: 0,
                  right: 0,
                  textAlign: 'center',
                  opacity: interpolate(frame, [120, 150], [0, 1], {
                    extrapolateRight: 'clamp',
                  }),
                }}
              >
                <div
                  style={{
                    fontSize: 28,
                    fontFamily: fontFamilies.sans,
                    color: '#FFFFFF',
                    fontWeight: 'bold',
                    textShadow: '0 3px 15px rgba(0, 0, 0, 1)',
                    marginBottom: 8,
                  }}
                >
                  Semantic Vector Space
                </div>
                <div
                  style={{
                    fontSize: 20,
                    color: colors.aiml.from,
                    textShadow: '0 2px 10px rgba(0, 0, 0, 1)',
                  }}
                >
                  384-dimensional embeddings, clustered by similarity
                </div>
              </div>
            )}
          </div>

          {/* RIGHT: Features - better vertical spacing */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 28,
              justifyContent: 'center',
            }}
          >
            {features.map((feature, i) => {
              const featureFrame = Math.max(0, frame - feature.delay);
              const opacity = interpolate(featureFrame, [0, 30], [0, 1], {
                extrapolateRight: 'clamp',
              });
              const x = spring({
                frame: featureFrame,
                fps,
                from: 50,
                to: 0,
                config: { damping: 20, stiffness: 80 },
              });

              return (
                <div
                  key={i}
                  style={{
                    opacity,
                    transform: `translateX(${x}px)`,
                    backgroundColor: 'rgba(0, 0, 0, 0.6)',
                    padding: '18px 24px',
                    borderRadius: 12,
                    border: `2px solid ${colors.aiml.from}40`,
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12,
                    }}
                  >
                    <div style={{ fontSize: 32 }}>{feature.icon}</div>
                    <div>
                      <div
                        style={{
                          fontSize: 20,
                          fontWeight: 'bold',
                          color: '#FFFFFF',
                          fontFamily: fontFamilies.sans,
                          marginBottom: 4,
                        }}
                      >
                        {feature.title}
                      </div>
                      <div
                        style={{
                          fontSize: 15,
                          color: '#FFFFFF',
                          fontFamily: fontFamilies.sans,
                          opacity: 0.8,
                        }}
                      >
                        {feature.desc}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}

            {/* Quality Tiers Box - more breathing room */}
            {frame >= 480 && (
              <div
                style={{
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  padding: '18px 24px',
                  borderRadius: 12,
                  border: `2px solid ${colors.aiml.from}40`,
                  opacity: interpolate(frame, [480, 510], [0, 1], {
                    extrapolateRight: 'clamp',
                  }),
                }}
              >
                <div
                  style={{
                    fontSize: 18,
                    fontWeight: 'bold',
                    color: '#FFFFFF',
                    fontFamily: fontFamilies.sans,
                    marginBottom: 10,
                  }}
                >
                  Quality Tiers
                </div>
                <div
                  style={{
                    fontSize: 14,
                    fontFamily: fontFamilies.mono,
                    color: '#A78BFA',
                    lineHeight: 1.6,
                  }}
                >
                  <div>Tier 1: ONNX (80-150ms) → $0</div>
                  <div>Tier 2: Groq (500ms) → $0.0015</div>
                  <div>Tier 3: Gemini (1-2s) → $0.01</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
