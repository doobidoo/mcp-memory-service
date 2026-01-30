/**
 * PerformanceSpotlight Scene (10-30s / 300-900 frames)
 * REDESIGNED: Clean 2-column layout with better visibility
 */

import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';
import { colors, gradient } from '../styles/colors';
import { fontFamilies } from '../styles/fonts';

export const PerformanceSpotlight: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title animation
  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: 'clamp',
  });

  // Key metrics
  const metrics = [
    { value: 5, unit: 'ms', label: 'Average Read Time', delay: 60 },
    { value: 534628, unit: 'x', label: 'Global Cache Speedup', delay: 120 },
    { value: 90, unit: '%', label: 'Token Reduction', delay: 180 },
  ];

  // Backend comparison data
  const backends = [
    { name: 'SQLite-Vec', time: 5, color: '#10B981' },
    { name: 'Hybrid', time: 5, color: '#3B82F6' },
    { name: 'Cloudflare', time: 45, color: '#8B5CF6' },
  ];

  return (
    <AbsoluteFill
      style={{
        background: gradient(colors.performance.from, colors.performance.to),
      }}
    >
      {/* Main container with proper spacing */}
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
          Performance
        </h1>

        {/* 2-Column Layout */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 80,
            flex: 1,
          }}
        >
          {/* LEFT COLUMN: Key Metrics */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 50,
              justifyContent: 'center',
            }}
          >
            {metrics.map((metric, i) => {
              const metricFrame = Math.max(0, frame - metric.delay);
              const progress = spring({
                frame: metricFrame,
                fps,
                config: { damping: 20, stiffness: 80 },
              });
              const value = interpolate(progress, [0, 1], [0, metric.value]);
              const formatted = Math.floor(value).toLocaleString();

              return (
                <div
                  key={i}
                  style={{
                    opacity: interpolate(metricFrame, [0, 20], [0, 1], {
                      extrapolateRight: 'clamp',
                    }),
                  }}
                >
                  <div
                    style={{
                      fontSize: i === 0 ? 120 : 80,
                      fontWeight: 'bold',
                      color: '#FFFFFF',
                      fontFamily: fontFamilies.mono,
                      textShadow: '0 0 40px rgba(16, 185, 129, 0.6)',
                      marginBottom: 10,
                    }}
                  >
                    {formatted}
                    <span style={{ fontSize: i === 0 ? 60 : 40 }}>{metric.unit}</span>
                  </div>
                  <div
                    style={{
                      fontSize: 24,
                      color: '#FFFFFF',
                      fontFamily: fontFamilies.sans,
                      fontWeight: '600',
                      backgroundColor: 'rgba(0, 0, 0, 0.5)',
                      padding: '12px 24px',
                      borderRadius: 12,
                      display: 'inline-block',
                      border: '2px solid rgba(255, 255, 255, 0.2)',
                    }}
                  >
                    {metric.label}
                  </div>
                </div>
              );
            })}
          </div>

          {/* RIGHT COLUMN: Backend Comparison */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              gap: 40,
            }}
          >
            {/* Config snippet at top */}
            {frame >= 240 && (
              <div
                style={{
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  padding: '30px',
                  borderRadius: 16,
                  border: '2px solid rgba(16, 185, 129, 0.3)',
                  opacity: interpolate(frame, [240, 270], [0, 1], {
                    extrapolateRight: 'clamp',
                  }),
                }}
              >
                <div
                  style={{
                    fontSize: 18,
                    fontFamily: fontFamilies.mono,
                    color: '#10B981',
                    lineHeight: 1.6,
                  }}
                >
                  MCP_MEMORY_SQLITE_PRAGMAS=
                  <br />
                  journal_mode=WAL,busy_timeout=15000
                </div>
              </div>
            )}

            {/* Backend chart */}
            {frame >= 300 && (
              <div
                style={{
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  padding: '40px',
                  borderRadius: 16,
                  border: '2px solid rgba(16, 185, 129, 0.3)',
                  opacity: interpolate(frame, [300, 330], [0, 1], {
                    extrapolateRight: 'clamp',
                  }),
                }}
              >
                <div
                  style={{
                    fontSize: 32,
                    fontFamily: fontFamilies.sans,
                    color: '#FFFFFF',
                    fontWeight: 'bold',
                    marginBottom: 30,
                    textShadow: '0 2px 10px rgba(0, 0, 0, 0.8)',
                  }}
                >
                  Backend Response Times
                </div>
                {backends.map((backend, i) => {
                  const barFrame = Math.max(0, frame - (330 + i * 15));
                  const barProgress = spring({
                    frame: barFrame,
                    fps,
                    config: { damping: 20, stiffness: 80 },
                  });
                  const barWidth = interpolate(barProgress, [0, 1], [0, (backend.time / 50) * 100]);

                  return (
                    <div
                      key={i}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 20,
                        marginBottom: 20,
                      }}
                    >
                      <div
                        style={{
                          width: 150,
                          fontSize: 22,
                          fontFamily: fontFamilies.sans,
                          color: '#FFFFFF',
                          textAlign: 'right',
                          fontWeight: 'bold',
                        }}
                      >
                        {backend.name}
                      </div>
                      <div
                        style={{
                          flex: 1,
                          height: 50,
                          backgroundColor: '#0F172A',
                          borderRadius: 10,
                          overflow: 'hidden',
                          position: 'relative',
                        }}
                      >
                        <div
                          style={{
                            width: `${barWidth}%`,
                            height: '100%',
                            backgroundColor: backend.color,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'flex-end',
                            paddingRight: 16,
                            boxShadow: `0 0 20px ${backend.color}80`,
                          }}
                        >
                          {barProgress > 0.5 && (
                            <span
                              style={{
                                fontSize: 22,
                                fontWeight: 'bold',
                                color: '#FFFFFF',
                                fontFamily: fontFamilies.mono,
                                textShadow: '0 2px 8px rgba(0, 0, 0, 1)',
                              }}
                            >
                              {backend.time}ms
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
