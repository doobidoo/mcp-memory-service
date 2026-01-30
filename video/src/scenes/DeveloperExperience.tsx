/**
 * DeveloperExperience Scene (80-105s / 2400-3150 frames)
 * REDESIGNED: Single column with large code examples
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

  // Code examples - simplified
  const codeExamples = [
    {
      title: 'Claude Desktop Integration',
      icon: '🖥️',
      code: [
        '{ "mcpServers": {',
        '    "memory": {',
        '      "command": "python",',
        '      "args": ["-m", "mcp_memory_service.server"]',
        '    }',
        '  }',
        '}',
      ],
      delay: 90,
    },
    {
      title: 'Python API',
      icon: '🐍',
      code: [
        'await memory.store_memory(',
        '  content="User preferences",',
        '  tags=["ui", "settings"]',
        ')',
        '',
        'results = await memory.search_memories(',
        '  query="user preferences", limit=5',
        ')',
      ],
      delay: 300,
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

        {/* Single column centered layout */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 60,
            flex: 1,
            justifyContent: 'center',
            maxWidth: 1400,
            margin: '0 auto',
            width: '100%',
          }}
        >
          {/* Code Examples */}
          {codeExamples.map((example, i) => {
            const exampleFrame = Math.max(0, frame - example.delay);
            const opacity = interpolate(exampleFrame, [0, 30], [0, 1], {
              extrapolateRight: 'clamp',
            });
            const y = spring({
              frame: exampleFrame,
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
                  transform: `translateY(${y}px)`,
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  padding: '40px 50px',
                  borderRadius: 20,
                  border: `3px solid ${colors.quality.from}50`,
                  boxShadow: `0 0 40px ${colors.quality.from}30`,
                }}
              >
                {/* Header */}
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 20,
                    marginBottom: 30,
                  }}
                >
                  <div style={{ fontSize: 56 }}>{example.icon}</div>
                  <div
                    style={{
                      fontSize: 36,
                      fontWeight: 'bold',
                      color: '#FFFFFF',
                      fontFamily: fontFamilies.sans,
                    }}
                  >
                    {example.title}
                  </div>
                </div>

                {/* Code */}
                <div
                  style={{
                    fontFamily: fontFamilies.mono,
                    fontSize: 22,
                    lineHeight: 1.8,
                    color: '#F59E0B',
                  }}
                >
                  {example.code.map((line, j) => (
                    <div
                      key={j}
                      style={{
                        color:
                          line.includes('{') || line.includes('}')
                            ? '#8B5CF6'
                            : line.includes('await') || line.includes('results')
                            ? '#3B82F6'
                            : line === ''
                            ? 'transparent'
                            : '#10B981',
                      }}
                    >
                      {line || ' '}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}

          {/* Bottom Integration Info */}
          {frame >= 510 && (
            <div
              style={{
                display: 'flex',
                justifyContent: 'center',
                gap: 40,
                opacity: interpolate(frame, [510, 540], [0, 1], {
                  extrapolateRight: 'clamp',
                }),
              }}
            >
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 48, marginBottom: 12 }}>🔌</div>
                <div
                  style={{
                    fontSize: 24,
                    fontWeight: 'bold',
                    color: '#FFFFFF',
                    fontFamily: fontFamilies.sans,
                  }}
                >
                  13+ AI Apps
                </div>
                <div
                  style={{
                    fontSize: 16,
                    color: '#FFFFFF',
                    fontFamily: fontFamilies.sans,
                    opacity: 0.7,
                  }}
                >
                  MCP Protocol
                </div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 48, marginBottom: 12 }}>🌐</div>
                <div
                  style={{
                    fontSize: 24,
                    fontWeight: 'bold',
                    color: '#FFFFFF',
                    fontFamily: fontFamilies.sans,
                  }}
                >
                  HTTP API
                </div>
                <div
                  style={{
                    fontSize: 16,
                    color: '#FFFFFF',
                    fontFamily: fontFamilies.sans,
                    opacity: 0.7,
                  }}
                >
                  REST + SSE
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </AbsoluteFill>
  );
};
