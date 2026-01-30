/**
 * ArchitectureTour Scene (30-55s / 900-1650 frames)
 * OPTIMIZED: Faster timing, cleaner 2-column layout
 */

import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';
import { colors, gradient } from '../styles/colors';
import { fontFamilies } from '../styles/fonts';

export const ArchitectureTour: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title animation
  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: 'clamp',
  });

  // Architecture layers
  const layers = [
    {
      title: 'MCP Server Layer',
      items: ['12 Tools', 'Global Cache', 'Client Detection'],
      color: '#3B82F6',
      delay: 60,
    },
    {
      title: 'Storage Strategy',
      items: ['SQLite-Vec', 'Cloudflare', 'Hybrid'],
      color: '#8B5CF6',
      delay: 120,
    },
    {
      title: 'Service Layer',
      items: ['Memory Service', 'Quality', 'Consolidation'],
      color: '#10B981',
      delay: 180,
    },
  ];

  // Design patterns
  const patterns = ['Strategy', 'Singleton', 'Factory', 'Orchestrator', 'Observer'];

  return (
    <AbsoluteFill
      style={{
        background: gradient(colors.architecture.from, colors.architecture.to),
      }}
    >
      {/* Blueprint grid */}
      <svg
        width="100%"
        height="100%"
        style={{
          position: 'absolute',
          opacity: 0.08,
        }}
      >
        <defs>
          <pattern id="arch-grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#F8FAFC" strokeWidth="0.5" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#arch-grid)" />
      </svg>

      {/* Main container */}
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          padding: '60px 100px',
          position: 'relative',
          zIndex: 1,
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
          Architecture
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
          {/* LEFT: Architecture Layers */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 40,
              justifyContent: 'center',
            }}
          >
            {layers.map((layer, i) => {
              const layerFrame = Math.max(0, frame - layer.delay);
              const opacity = interpolate(layerFrame, [0, 30], [0, 1], {
                extrapolateRight: 'clamp',
              });
              const y = spring({
                frame: layerFrame,
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
                    backgroundColor: 'rgba(0, 0, 0, 0.6)',
                    padding: '30px',
                    borderRadius: 16,
                    border: `2px solid ${layer.color}60`,
                    boxShadow: `0 0 30px ${layer.color}30`,
                  }}
                >
                  <div
                    style={{
                      fontSize: 28,
                      fontWeight: 'bold',
                      color: '#FFFFFF',
                      fontFamily: fontFamilies.sans,
                      marginBottom: 16,
                      textShadow: '0 2px 10px rgba(0, 0, 0, 0.8)',
                    }}
                  >
                    {layer.title}
                  </div>
                  <div
                    style={{
                      display: 'flex',
                      flexWrap: 'wrap',
                      gap: 10,
                    }}
                  >
                    {layer.items.map((item, j) => (
                      <div
                        key={j}
                        style={{
                          fontSize: 18,
                          color: '#FFFFFF',
                          fontFamily: fontFamilies.sans,
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          padding: '8px 16px',
                          borderRadius: 8,
                          border: '1px solid rgba(255, 255, 255, 0.2)',
                        }}
                      >
                        {item}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          {/* RIGHT: Code + Patterns */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 40,
              justifyContent: 'center',
            }}
          >
            {/* Code snippet */}
            {frame >= 240 && (
              <div
                style={{
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  padding: '30px',
                  borderRadius: 16,
                  border: '2px solid rgba(59, 130, 246, 0.3)',
                  opacity: interpolate(frame, [240, 270], [0, 1], {
                    extrapolateRight: 'clamp',
                  }),
                }}
              >
                <div
                  style={{
                    fontSize: 20,
                    fontFamily: fontFamilies.sans,
                    color: '#FFFFFF',
                    fontWeight: 'bold',
                    marginBottom: 16,
                  }}
                >
                  Strategy Pattern
                </div>
                <div
                  style={{
                    fontSize: 16,
                    fontFamily: fontFamilies.mono,
                    color: '#3B82F6',
                    lineHeight: 1.6,
                  }}
                >
                  <div style={{ color: '#8B5CF6' }}>class BaseStorage(ABC):</div>
                  <div style={{ paddingLeft: 20, color: '#10B981' }}>
                    @abstractmethod
                  </div>
                  <div style={{ paddingLeft: 20 }}>async def store_memory(...)</div>
                  <div style={{ paddingLeft: 40 }}>→ SQLiteVecStorage()</div>
                  <div style={{ paddingLeft: 40 }}>→ CloudflareStorage()</div>
                  <div style={{ paddingLeft: 40 }}>→ HybridStorage()</div>
                </div>
              </div>
            )}

            {/* Design Patterns */}
            {frame >= 360 && (
              <div
                style={{
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  padding: '30px',
                  borderRadius: 16,
                  border: '2px solid rgba(59, 130, 246, 0.3)',
                  opacity: interpolate(frame, [360, 390], [0, 1], {
                    extrapolateRight: 'clamp',
                  }),
                }}
              >
                <div
                  style={{
                    fontSize: 24,
                    fontFamily: fontFamilies.sans,
                    color: '#FFFFFF',
                    fontWeight: 'bold',
                    marginBottom: 20,
                  }}
                >
                  Design Patterns
                </div>
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: 12,
                  }}
                >
                  {patterns.map((pattern, i) => {
                    const patternFrame = Math.max(0, frame - (390 + i * 10));
                    const scale = spring({
                      frame: patternFrame,
                      fps,
                      from: 0,
                      to: 1,
                      config: { damping: 15, stiffness: 100 },
                    });

                    return (
                      <div
                        key={i}
                        style={{
                          transform: `scale(${scale})`,
                          padding: '10px 20px',
                          backgroundColor: 'rgba(59, 130, 246, 0.2)',
                          borderRadius: 20,
                          fontSize: 18,
                          fontWeight: 'bold',
                          color: '#3B82F6',
                          fontFamily: fontFamilies.sans,
                          border: '2px solid rgba(59, 130, 246, 0.4)',
                        }}
                      >
                        {pattern}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Request Flow */}
            {frame >= 480 && (
              <div
                style={{
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  padding: '30px',
                  borderRadius: 16,
                  border: '2px solid rgba(59, 130, 246, 0.3)',
                  opacity: interpolate(frame, [480, 510], [0, 1], {
                    extrapolateRight: 'clamp',
                  }),
                }}
              >
                <div
                  style={{
                    fontSize: 20,
                    fontFamily: fontFamilies.sans,
                    color: '#FFFFFF',
                    fontWeight: 'bold',
                    marginBottom: 16,
                  }}
                >
                  Request Flow
                </div>
                <div
                  style={{
                    fontSize: 18,
                    fontFamily: fontFamilies.mono,
                    color: '#FFFFFF',
                    lineHeight: 1.8,
                  }}
                >
                  MCP Client
                  <span style={{ color: '#3B82F6', margin: '0 12px' }}>→</span>
                  Server Layer
                  <span style={{ color: '#3B82F6', margin: '0 12px' }}>→</span>
                  Storage
                  <br />
                  <span style={{ color: '#10B981', margin: '0 12px' }}>←</span>
                  Response
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
