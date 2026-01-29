import { AbsoluteFill } from 'remotion';
import { colors } from './styles/colors';

export const Video: React.FC = () => {
  return (
    <AbsoluteFill
      style={{
        backgroundColor: colors.background,
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <h1
        style={{
          fontFamily: 'sans-serif',
          fontSize: 72,
          color: colors.textPrimary,
        }}
      >
        MCP Memory Service
      </h1>
      <p style={{ color: colors.textSecondary, fontSize: 24 }}>
        Technical Showcase - Coming Soon
      </p>
    </AbsoluteFill>
  );
};
