/**
 * Layered architecture diagram with animations
 */

import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { colors } from '../styles/colors';
import { fontFamilies } from '../styles/fonts';

interface Layer {
  title: string;
  items: string[];
  color: string;
}

interface ArchitectureDiagramProps {
  layers: Layer[];
  startFrame: number;
}

export const ArchitectureDiagram: React.FC<ArchitectureDiagramProps> = ({
  layers,
  startFrame,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const relativeFrame = Math.max(0, frame - startFrame);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 30,
        width: '100%',
      }}
    >
      {layers.map((layer, layerIndex) => {
        const layerDelay = layerIndex * 25;
        const layerFrame = Math.max(0, relativeFrame - layerDelay);

        const slideY = spring({
          frame: layerFrame,
          fps,
          from: 80,
          to: 0,
          config: {
            damping: 18,
            stiffness: 90,
          },
        });

        const opacity = interpolate(layerFrame, [0, 15], [0, 1], {
          extrapolateRight: 'clamp',
        });

        return (
          <div
            key={layerIndex}
            style={{
              transform: `translateY(${slideY}px)`,
              opacity,
              padding: '28px 32px',
              backgroundColor: colors.cardBg,
              borderRadius: '16px',
              borderLeft: `6px solid ${layer.color}`,
              boxShadow: `0 0 30px ${layer.color}20`,
            }}
          >
            <h3
              style={{
                fontSize: 28,
                color: layer.color,
                marginBottom: 20,
                fontFamily: fontFamilies.mono,
                fontWeight: 'bold',
                margin: 0,
                marginBottom: 20,
              }}
            >
              {layer.title}
            </h3>

            <div
              style={{
                display: 'flex',
                gap: 12,
                flexWrap: 'wrap',
              }}
            >
              {layer.items.map((item, itemIndex) => {
                const itemDelay = itemIndex * 8;
                const itemFrame = Math.max(0, layerFrame - itemDelay);

                const scale = spring({
                  frame: itemFrame,
                  fps,
                  from: 0.5,
                  to: 1,
                  config: {
                    damping: 15,
                    stiffness: 120,
                  },
                });

                return (
                  <div
                    key={itemIndex}
                    style={{
                      transform: `scale(${scale})`,
                      padding: '10px 20px',
                      backgroundColor: colors.background,
                      borderRadius: '8px',
                      fontSize: 16,
                      color: colors.textSecondary,
                      fontFamily: fontFamilies.sans,
                      border: `1px solid ${colors.cardBg}`,
                    }}
                  >
                    {item}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
};
