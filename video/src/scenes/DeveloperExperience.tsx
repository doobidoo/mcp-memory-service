/**
 * DeveloperExperience Scene (80-105s / 2400-3150 frames)
 * OPTIMIZED: Cleaner 2-column layout
 */

import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';
import { colors, gradient } from '../styles/colors';
import { fontFamilies } from '../styles/fonts';

export const DeveloperExperience: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title animation
  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: 'clamp',
  });

  // Integration examples
  const integrations = [
    {
      icon: '🖥️',
      title: 'Claude Desktop',
      desc: 'Native MCP integration',
      delay: 90,
    },
    {
      icon: '🔌',
      title: '13+ AI Apps',
      desc: 'MCP protocol support',
      delay: 150,
    },
    {
      icon: '🌐',
      title: 'HTTP API',
      desc: 'REST endpoints + SSE',
      delay: 210,
    },
  ];

  return (
    <AbsoluteFill
      style={{
        background: gradient(colors.quality.from, colors.quality.to),
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
          Developer Experience
        </h1>

        {/* 2-Column Layout */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 80,
            flex: 1,
            alignItems: 'start',
          }}
        >
          {/* LEFT: Integration Examples */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 40,
              justifyContent: 'center',
            }}
          >
            {integrations.map((integration, i) => {
              const intFrame = Math.max(0, frame - integration.delay);
              const opacity = interpolate(intFrame, [0, 30], [0, 1], {
                extrapolateRight: 'clamp',
              });
              const scale = spring({
                frame: intFrame,
                fps,
                from: 0.8,
                to: 1,
                config: { damping: 20, stiffness: 80 },
              });

              return (
                <div
                  key={i}
                  style={{
                    opacity,
                    transform: `scale(${scale})`,
                    backgroundColor: 'rgba(0, 0, 0, 0.6)',
                    padding: '30px',
                    borderRadius: 16,
                    border: `2px solid ${colors.quality.from}40`,
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 20,
                      marginBottom: 12,
                    }}
                  >
                    <div style={{ fontSize: 48 }}>{integration.icon}</div>
                    <div>
                      <div
                        style={{
                          fontSize: 28,
                          fontWeight: 'bold',
                          color: '#FFFFFF',
                          fontFamily: fontFamilies.sans,
                          marginBottom: 4,
                        }}
                      >
                        {integration.title}
                      </div>
                      <div
                        style={{
                          fontSize: 18,
                          color: '#FFFFFF',
                          fontFamily: fontFamilies.sans,
                          opacity: 0.8,
                        }}
                      >
                        {integration.desc}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}

            {/* HTTP API Code Example - moved to left */}
            {frame >= 510 && (
              <div
                style={{
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  padding: '30px',
                  borderRadius: 16,
                  border: '2px solid rgba(245, 158, 11, 0.3)',
                  opacity: interpolate(frame, [510, 540], [0, 1], {
                    extrapolateRight: 'clamp',
                  }),
                }}
              >
                <div
                  style={{
                    fontSize: 22,
                    fontWeight: 'bold',
                    color: '#FFFFFF',
                    fontFamily: fontFamilies.sans,
                    marginBottom: 16,
                  }}
                >
                  HTTP API
                </div>
                <div
                  style={{
                    fontSize: 16,
                    fontFamily: fontFamilies.mono,
                    color: '#F59E0B',
                    lineHeight: 1.8,
                  }}
                >
                  <div style={{ color: '#8B5CF6' }}>curl -X POST</div>
                  <div style={{ color: '#10B981' }}>
                    http://localhost:8000/api/memories
                  </div>
                  <div style={{ color: '#3B82F6' }}>-H "Content-Type: application/json"</div>
                  <div style={{ color: '#94A3B8' }}>-d '&#123;"content": "...", "tags": [...]&#125;'</div>
                </div>
              </div>
            )}
          </div>

          {/* RIGHT: Code Examples */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 40,
              justifyContent: 'center',
            }}
          >
            {/* Claude Desktop Config */}
            {frame >= 270 && (
              <div
                style={{
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  padding: '30px',
                  borderRadius: 16,
                  border: '2px solid rgba(245, 158, 11, 0.3)',
                  opacity: interpolate(frame, [270, 300], [0, 1], {
                    extrapolateRight: 'clamp',
                  }),
                }}
              >
                <div
                  style={{
                    fontSize: 22,
                    fontWeight: 'bold',
                    color: '#FFFFFF',
                    fontFamily: fontFamilies.sans,
                    marginBottom: 16,
                  }}
                >
                  Claude Desktop Config
                </div>
                <div
                  style={{
                    fontSize: 16,
                    fontFamily: fontFamilies.mono,
                    color: '#F59E0B',
                    lineHeight: 1.8,
                  }}
                >
                  <div style={{ color: '#94A3B8' }}>{`{`}</div>
                  <div style={{ paddingLeft: 20 }}>"mcpServers": {`{`}</div>
                  <div style={{ paddingLeft: 40 }}>"memory": {`{`}</div>
                  <div style={{ paddingLeft: 60, color: '#10B981' }}>
                    "command": "python",
                  </div>
                  <div style={{ paddingLeft: 60, color: '#10B981' }}>
                    "args": ["-m", "mcp_memory..."]
                  </div>
                  <div style={{ paddingLeft: 40 }}>{`}`}</div>
                  <div style={{ paddingLeft: 20 }}>{`}`}</div>
                  <div style={{ color: '#94A3B8' }}>{`}`}</div>
                </div>
              </div>
            )}

            {/* Python API */}
            {frame >= 390 && (
              <div
                style={{
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  padding: '30px',
                  borderRadius: 16,
                  border: '2px solid rgba(245, 158, 11, 0.3)',
                  opacity: interpolate(frame, [390, 420], [0, 1], {
                    extrapolateRight: 'clamp',
                  }),
                }}
              >
                <div
                  style={{
                    fontSize: 22,
                    fontWeight: 'bold',
                    color: '#FFFFFF',
                    fontFamily: fontFamilies.sans,
                    marginBottom: 16,
                  }}
                >
                  Python API
                </div>
                <div
                  style={{
                    fontSize: 16,
                    fontFamily: fontFamilies.mono,
                    color: '#F59E0B',
                    lineHeight: 1.8,
                  }}
                >
                  <div style={{ color: '#8B5CF6' }}>await memory.store_memory(</div>
                  <div style={{ paddingLeft: 20, color: '#10B981' }}>
                    content="User prefers dark mode",
                  </div>
                  <div style={{ paddingLeft: 20, color: '#10B981' }}>
                    tags=["preference", "ui"]
                  </div>
                  <div style={{ color: '#8B5CF6' }}>)</div>
                  <div style={{ marginTop: 12, color: '#3B82F6' }}>
                    results = await memory.search_memories(
                  </div>
                  <div style={{ paddingLeft: 20, color: '#10B981' }}>
                    query="user preferences"
                  </div>
                  <div style={{ color: '#3B82F6' }}>)</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
