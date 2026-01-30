/**
 * Feature list with icons and descriptions
 * Staggered slide-in animation
 */

import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { fontFamilies } from '../styles/fonts';
import { colors } from '../styles/colors';

interface Feature {
  icon: string;
  title: string;
  description: string;
}

interface FeatureListProps {
  features: Feature[];
  startFrame: number;
  color: string;
}

export const FeatureList: React.FC<FeatureListProps> = ({
  features,
  startFrame,
  color,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 24,
      }}
    >
      {features.map((feature, index) => {
        const itemDelay = index * 20;
        const itemFrame = Math.max(0, frame - (startFrame + itemDelay));

        const slideX = spring({
          frame: itemFrame,
          fps,
          from: 100,
          to: 0,
          config: {
            damping: 18,
            stiffness: 90,
          },
        });

        const opacity = interpolate(itemFrame, [0, 15], [0, 1], {
          extrapolateRight: 'clamp',
        });

        const glowIntensity = interpolate(
          Math.sin((frame + index * 30) / 20),
          [-1, 1],
          [10, 25]
        );

        return (
          <div
            key={index}
            style={{
              transform: `translateX(${slideX}px)`,
              opacity,
              display: 'flex',
              gap: 20,
              alignItems: 'flex-start',
              padding: '20px 24px',
              backgroundColor: colors.cardBg,
              borderRadius: 12,
              border: `1px solid ${color}30`,
              boxShadow: `0 0 ${glowIntensity}px ${color}20`,
            }}
          >
            {/* Icon */}
            <div
              style={{
                fontSize: 36,
                width: 50,
                height: 50,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
              }}
            >
              {feature.icon}
            </div>

            {/* Content */}
            <div style={{ flex: 1 }}>
              <div
                style={{
                  fontSize: 22,
                  fontWeight: 'bold',
                  color: colors.textPrimary,
                  fontFamily: fontFamilies.sans,
                  marginBottom: 8,
                }}
              >
                {feature.title}
              </div>
              <div
                style={{
                  fontSize: 16,
                  color: colors.textSecondary,
                  fontFamily: fontFamilies.sans,
                  lineHeight: 1.5,
                }}
              >
                {feature.description}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};
