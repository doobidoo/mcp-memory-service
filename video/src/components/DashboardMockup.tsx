/**
 * Dashboard UI mockup for Developer Experience scene
 * Animated memory cards and search bar
 */

import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { colors } from '../styles/colors';
import { fontFamilies } from '../styles/fonts';
import { mockMemories } from '../data/mockData';

interface DashboardMockupProps {
  startFrame: number;
}

export const DashboardMockup: React.FC<DashboardMockupProps> = ({ startFrame }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const relativeFrame = Math.max(0, frame - startFrame);

  // Slide up animation
  const slideY = spring({
    frame: relativeFrame,
    fps,
    from: 100,
    to: 0,
    config: {
      damping: 20,
      stiffness: 80,
    },
  });

  const opacity = interpolate(relativeFrame, [0, 20], [0, 1], {
    extrapolateRight: 'clamp',
  });

  // Search bar typing animation
  const searchText = 'semantic search...';
  const typedChars = Math.min(
    searchText.length,
    Math.floor(relativeFrame / 3)
  );
  const typedText = searchText.slice(0, typedChars);

  // Cursor blink
  const cursorVisible = Math.floor(relativeFrame / 15) % 2 === 0;

  return (
    <div
      style={{
        transform: `translateY(${slideY}px)`,
        opacity,
        width: '100%',
        backgroundColor: colors.cardBg,
        borderRadius: 16,
        padding: 30,
        border: `1px solid ${colors.quality.from}30`,
      }}
    >
      {/* Search bar */}
      <div
        style={{
          marginBottom: 24,
          padding: '16px 20px',
          backgroundColor: colors.background,
          borderRadius: 12,
          border: `2px solid ${colors.quality.from}40`,
          fontFamily: fontFamilies.mono,
          fontSize: 16,
          color: colors.textSecondary,
          display: 'flex',
          alignItems: 'center',
          gap: 12,
        }}
      >
        <span style={{ fontSize: 20 }}>🔍</span>
        <span>
          {typedText}
          {cursorVisible && relativeFrame < 120 && (
            <span style={{ color: colors.quality.from }}>|</span>
          )}
        </span>
      </div>

      {/* Memory cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {mockMemories.slice(0, 3).map((memory, i) => {
          const cardDelay = (i + 1) * 15;
          const cardFrame = Math.max(0, relativeFrame - cardDelay);

          const cardOpacity = interpolate(cardFrame, [0, 15], [0, 1], {
            extrapolateRight: 'clamp',
          });

          const qualityWidth = memory.quality * 100;

          return (
            <div
              key={i}
              style={{
                opacity: cardOpacity,
                padding: '16px 20px',
                backgroundColor: colors.background,
                borderRadius: 10,
                border: `1px solid ${colors.cardBg}`,
              }}
            >
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  marginBottom: 12,
                }}
              >
                <div
                  style={{
                    flex: 1,
                    fontSize: 15,
                    color: colors.textPrimary,
                    fontFamily: fontFamilies.sans,
                    lineHeight: 1.5,
                  }}
                >
                  {memory.content}
                </div>
                <div
                  style={{
                    fontSize: 13,
                    color: colors.textSecondary,
                    fontFamily: fontFamilies.sans,
                    marginLeft: 20,
                  }}
                >
                  {memory.timestamp}
                </div>
              </div>

              {/* Tags */}
              <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
                {memory.tags.map((tag) => (
                  <span
                    key={tag}
                    style={{
                      padding: '4px 12px',
                      backgroundColor: colors.cardBg,
                      borderRadius: 6,
                      fontSize: 12,
                      color: colors.quality.from,
                      fontFamily: fontFamilies.sans,
                    }}
                  >
                    {tag}
                  </span>
                ))}
              </div>

              {/* Quality bar */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div
                  style={{
                    flex: 1,
                    height: 6,
                    backgroundColor: colors.cardBg,
                    borderRadius: 3,
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${qualityWidth}%`,
                      height: '100%',
                      backgroundColor: colors.quality.from,
                      transition: 'width 0.3s ease',
                    }}
                  />
                </div>
                <span
                  style={{
                    fontSize: 12,
                    color: colors.textSecondary,
                    fontFamily: fontFamilies.sans,
                    minWidth: 45,
                  }}
                >
                  {Math.round(memory.quality * 100)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Stats footer */}
      <div
        style={{
          marginTop: 20,
          paddingTop: 20,
          borderTop: `1px solid ${colors.cardBg}`,
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 14,
          color: colors.textSecondary,
          fontFamily: fontFamilies.sans,
        }}
      >
        <span>Total Memories: 1,234</span>
        <span>Avg Quality: 91%</span>
      </div>
    </div>
  );
};
