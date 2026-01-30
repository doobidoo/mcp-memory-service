/**
 * AIMLIntelligence Scene (90-135s / 2700-4050 frames)
 *
 * Timeline:
 * - 0-3s: Purple gradient wash, title with wave animation
 * - 3-15s: 3D vector space builds and animates
 * - 15-30s: Feature list slides in (4 features, staggered)
 * - 30-45s: Code snippet reveals (quality tiers)
 */

import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';
import { colors, gradient } from '../styles/colors';
import { fontFamilies } from '../styles/fonts';
import { VectorSpace3D } from '../components/VectorSpace3D';
import { FeatureList } from '../components/FeatureList';
import { CodeBlock } from '../components/CodeBlock';

export const AIMLIntelligence: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title with wave animation
  const titleY = spring({
    frame,
    fps,
    from: 100,
    to: 0,
    config: {
      damping: 20,
      stiffness: 70,
    },
  });

  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: 'clamp',
  });

  // Organic wave for title
  const titleWave = Math.sin(frame / 25) * 5;

  // Particle field background
  const particleCount = 40;
  const particles = Array.from({ length: particleCount }, (_, i) => ({
    id: i,
    x: (i / particleCount) * 1920,
    y: Math.random() * 1080,
    speed: 0.5 + Math.random() * 1,
    size: 1 + Math.random() * 2,
  }));

  // AI/ML features
  const features = [
    {
      icon: '🧠',
      title: 'Vector Embeddings',
      description: 'ONNX local model (384 dimensions) - sentence-transformers/all-MiniLM-L6-v2',
    },
    {
      icon: '⭐',
      title: 'Quality Scoring',
      description: '3-tier system: Local ONNX (80-150ms), Groq (500ms), Gemini (1-2s)',
    },
    {
      icon: '🌙',
      title: 'Memory Consolidation',
      description: 'Dream-inspired maintenance with decay, associations, and archival',
    },
    {
      icon: '🔗',
      title: 'Relationship Inference',
      description: 'Automatic graph building with multi-factor analysis and confidence scoring',
    },
  ];

  // Code snippet
  const codeSnippet = `# Quality Scoring Tiers
Tier 1: ONNX (80-150ms)    → $0 (Local, Private)
Tier 2: Groq (500-800ms)   → $0.0015
Tier 3: Gemini (1-2s)      → $0.01

# Automatic fallback with quality boost
quality_score = await scorer.evaluate(memory)`;

  return (
    <AbsoluteFill
      style={{
        background: gradient(colors.aiml.from, colors.aiml.to),
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      {/* Floating particle field */}
      {particles.map((particle) => {
        const y = (particle.y + frame * particle.speed) % 1080;
        const opacity = interpolate(
          y,
          [0, 200, 880, 1080],
          [0, 0.3, 0.3, 0]
        );

        return (
          <div
            key={particle.id}
            style={{
              position: 'absolute',
              left: particle.x,
              top: y,
              width: particle.size,
              height: particle.size,
              borderRadius: '50%',
              backgroundColor: colors.aiml.from,
              opacity,
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
          padding: '80px',
        }}
      >
        {/* Title with wave */}
        <h1
          style={{
            fontFamily: fontFamilies.mono,
            fontSize: 64,
            fontWeight: 'bold',
            color: colors.textPrimary,
            margin: 0,
            marginBottom: 50,
            transform: `translateY(${titleY + titleWave}px)`,
            opacity: titleOpacity,
            textShadow: '0 0 30px rgba(139, 92, 246, 0.4)',
          }}
        >
          AI/ML Intelligence
        </h1>

        {/* Content grid */}
        <div
          style={{
            display: 'flex',
            gap: 60,
            flex: 1,
          }}
        >
          {/* Left: 3D Vector Space */}
          <div
            style={{
              flex: 0.9,
              position: 'relative',
            }}
          >
            {frame >= 90 && (
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  opacity: interpolate(frame, [90, 150], [0, 1], {
                    extrapolateRight: 'clamp',
                  }),
                }}
              >
                <VectorSpace3D frame={frame - 90} />
              </div>
            )}

            {/* Label for 3D */}
            {frame >= 200 && (
              <div
                style={{
                  position: 'absolute',
                  bottom: 20,
                  left: '50%',
                  transform: 'translateX(-50%)',
                  fontSize: 18,
                  fontFamily: fontFamilies.sans,
                  color: colors.textSecondary,
                  textAlign: 'center',
                  opacity: interpolate(frame, [200, 230], [0, 1], {
                    extrapolateRight: 'clamp',
                  }),
                }}
              >
                Semantic Vector Space
                <br />
                <span style={{ fontSize: 14, opacity: 0.6 }}>
                  Clustered by similarity
                </span>
              </div>
            )}
          </div>

          {/* Right: Features and Code */}
          <div
            style={{
              flex: 1.1,
              display: 'flex',
              flexDirection: 'column',
              gap: 40,
            }}
          >
            {/* Feature list */}
            {frame >= 450 && (
              <FeatureList
                features={features}
                startFrame={450}
                color={colors.aiml.from}
              />
            )}

            {/* Code snippet */}
            {frame >= 900 && (
              <div style={{ marginTop: 'auto' }}>
                <CodeBlock
                  code={codeSnippet}
                  language="python"
                  startFrame={900}
                  animationDuration={90}
                  fontSize={15}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
