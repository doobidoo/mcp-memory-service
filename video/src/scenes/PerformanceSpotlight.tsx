/**
 * PerformanceSpotlight Scene (15-50s / 450-1500 frames)
 *
 * Timeline:
 * - 0-3s: Green gradient wash, title drop
 * - 3-10s: Speedometer animates (0 → 5ms)
 * - 10-20s: Metrics count up (534,628x, 5ms, 90%)
 * - 20-25s: Code snippet fades in
 * - 25-35s: Bar chart comparing backends
 */

import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';
import { colors, gradient } from '../styles/colors';
import { fontFamilies } from '../styles/fonts';
import { Speedometer } from '../components/Speedometer';
import { CountUp } from '../components/CountUp';
import { CodeBlock } from '../components/CodeBlock';

export const PerformanceSpotlight: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title animation (0-90 frames / 0-3s)
  const titleY = spring({
    frame,
    fps,
    from: -100,
    to: 0,
    config: {
      damping: 18,
      stiffness: 90,
    },
  });

  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: 'clamp',
  });

  // Racing particles for background effect
  const particleCount = 30;
  const particles = Array.from({ length: particleCount }, (_, i) => ({
    id: i,
    y: (i / particleCount) * 1080,
    speed: 5 + Math.random() * 10,
    size: 2 + Math.random() * 4,
    delay: Math.random() * 100,
  }));

  // Code snippet
  const codeSnippet = `MCP_MEMORY_SQLITE_PRAGMAS=journal_mode=WAL,\\
  busy_timeout=15000,cache_size=20000`;

  // Backend comparison data
  const backends = [
    { name: 'SQLite-Vec', time: 5, color: colors.performance.from },
    { name: 'Hybrid', time: 5, color: colors.architecture.from },
    { name: 'Cloudflare', time: 45, color: colors.aiml.from },
  ];

  return (
    <AbsoluteFill
      style={{
        background: gradient(colors.performance.from, colors.performance.to),
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      {/* Racing particles background */}
      {particles.map((particle) => {
        const particleX = ((frame + particle.delay) * particle.speed) % 2000 - 100;
        const particleOpacity = interpolate(
          particleX,
          [0, 500, 1500, 2000],
          [0, 0.5, 0.5, 0]
        );

        return (
          <div
            key={particle.id}
            style={{
              position: 'absolute',
              left: particleX,
              top: particle.y,
              width: particle.size,
              height: particle.size,
              borderRadius: '50%',
              backgroundColor: '#F8FAFC',
              opacity: particleOpacity,
            }}
          />
        );
      })}

      {/* Main content */}
      <div
        style={{
          position: 'relative',
          zIndex: 1,
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: '80px',
        }}
      >
        {/* Title */}
        <h1
          style={{
            fontFamily: fontFamilies.mono,
            fontSize: 64,
            fontWeight: 'bold',
            color: colors.textPrimary,
            margin: 0,
            transform: `translateY(${titleY}px)`,
            opacity: titleOpacity,
            textShadow: '0 0 30px rgba(16, 185, 129, 0.3)',
          }}
        >
          Performance
        </h1>

        {/* Content grid */}
        <div
          style={{
            display: 'flex',
            width: '100%',
            maxWidth: 1600,
            marginTop: 60,
            gap: 60,
          }}
        >
          {/* Left: Speedometer */}
          <div
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {frame >= 90 && (
              <Speedometer
                maxValue={100}
                currentValue={5}
                label="Average Read Time"
                color={colors.performance.from}
                unit="ms"
                startFrame={90}
              />
            )}
          </div>

          {/* Right: Metrics */}
          <div
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              gap: 40,
              justifyContent: 'center',
            }}
          >
            {/* Metric 1: Cache boost */}
            {frame >= 300 && (
              <CountUp
                to={534628}
                suffix="x"
                fontSize={72}
                color={colors.textPrimary}
                glowColor={colors.performance.from}
                delay={300}
                label="Faster with Global Caching"
                labelSize={18}
              />
            )}

            {/* Metric 2: Token reduction */}
            {frame >= 420 && (
              <CountUp
                to={90}
                suffix="%"
                fontSize={56}
                color={colors.textPrimary}
                glowColor={colors.performance.from}
                delay={420}
                label="Token Reduction (HTTP API vs MCP)"
                labelSize={18}
              />
            )}
          </div>
        </div>

        {/* Code snippet */}
        {frame >= 600 && (
          <div style={{ marginTop: 40, width: '100%', maxWidth: 900 }}>
            <CodeBlock
              code={codeSnippet}
              language="bash"
              startFrame={600}
              animationDuration={90}
              fontSize={18}
            />
          </div>
        )}

        {/* Backend comparison chart */}
        {frame >= 750 && (
          <div
            style={{
              marginTop: 50,
              width: '100%',
              maxWidth: 800,
              display: 'flex',
              flexDirection: 'column',
              gap: 20,
            }}
          >
            <div
              style={{
                fontSize: 24,
                fontFamily: fontFamilies.sans,
                color: colors.textSecondary,
                textAlign: 'center',
              }}
            >
              Backend Response Times
            </div>
            {backends.map((backend, i) => {
              const barFrame = Math.max(0, frame - (750 + i * 20));
              const barProgress = spring({
                frame: barFrame,
                fps,
                config: {
                  damping: 20,
                  stiffness: 80,
                },
              });

              const barWidth = interpolate(
                barProgress,
                [0, 1],
                [0, (backend.time / 50) * 600]
              );

              return (
                <div
                  key={backend.name}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 20,
                  }}
                >
                  <div
                    style={{
                      width: 140,
                      fontSize: 18,
                      fontFamily: fontFamilies.sans,
                      color: colors.textPrimary,
                      textAlign: 'right',
                    }}
                  >
                    {backend.name}
                  </div>
                  <div
                    style={{
                      flex: 1,
                      height: 40,
                      backgroundColor: '#1E293B',
                      borderRadius: 8,
                      position: 'relative',
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        width: barWidth,
                        height: '100%',
                        backgroundColor: backend.color,
                        borderRadius: 8,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'flex-end',
                        paddingRight: 12,
                      }}
                    >
                      {barProgress > 0.5 && (
                        <span
                          style={{
                            fontSize: 16,
                            fontWeight: 'bold',
                            color: colors.textPrimary,
                            fontFamily: fontFamilies.sans,
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
    </AbsoluteFill>
  );
};
