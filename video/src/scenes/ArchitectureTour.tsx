/**
 * ArchitectureTour Scene (50-90s / 1500-2700 frames)
 *
 * Timeline:
 * - 0-3s: Blue gradient wash, title slide
 * - 3-13s: Layer-by-layer diagram reveal (3 layers)
 * - 13-23s: Code snippets carousel (2 snippets, 5s each)
 * - 23-30s: Design pattern badges appear
 * - 30-40s: Data flow arrows animation
 */

import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig, Sequence } from 'remotion';
import { colors, gradient } from '../styles/colors';
import { fontFamilies } from '../styles/fonts';
import { ArchitectureDiagram } from '../components/ArchitectureDiagram';
import { CodeBlock } from '../components/CodeBlock';

export const ArchitectureTour: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title animation
  const titleX = spring({
    frame,
    fps,
    from: -200,
    to: 0,
    config: {
      damping: 18,
      stiffness: 90,
    },
  });

  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: 'clamp',
  });

  // Blueprint grid background
  const gridOpacity = interpolate(frame, [0, 60], [0, 0.1], {
    extrapolateRight: 'clamp',
  });

  // Architecture layers
  const architectureLayers = [
    {
      title: 'MCP Server Layer',
      items: ['35 Tools', 'Global Caching', 'Client Detection', 'Modular Handlers'],
      color: colors.architecture.from,
    },
    {
      title: 'Storage Strategy',
      items: ['SQLite-Vec', 'Cloudflare', 'Hybrid', 'BaseStorage Interface'],
      color: colors.aiml.from,
    },
    {
      title: 'Service Layer',
      items: ['Memory Service', 'Quality System', 'Consolidation', 'Embeddings'],
      color: colors.performance.from,
    },
  ];

  // Code snippets
  const codeSnippets = [
    {
      language: 'python',
      code: `# BaseStorage interface
class BaseStorage(ABC):
    @abstractmethod
    async def store_memory(
        self, content: str, tags: List[str]
    ) -> str:
        pass`,
    },
    {
      language: 'python',
      code: `# Factory pattern
def create_storage(backend: str) -> BaseStorage:
    if backend == "hybrid":
        return HybridStorage()
    elif backend == "cloudflare":
        return CloudflareStorage()
    return SQLiteVecStorage()`,
    },
  ];

  // Design patterns
  const designPatterns = ['Strategy', 'Singleton', 'Orchestrator', 'Factory', 'Observer'];

  return (
    <AbsoluteFill
      style={{
        background: gradient(colors.architecture.from, colors.architecture.to),
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      {/* Blueprint grid background */}
      <svg
        width="100%"
        height="100%"
        style={{
          position: 'absolute',
          opacity: gridOpacity,
        }}
      >
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path
              d="M 40 0 L 0 0 0 40"
              fill="none"
              stroke="#F8FAFC"
              strokeWidth="0.5"
            />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
      </svg>

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
        {/* Title */}
        <h1
          style={{
            fontFamily: fontFamilies.mono,
            fontSize: 64,
            fontWeight: 'bold',
            color: colors.textPrimary,
            margin: 0,
            marginBottom: 50,
            transform: `translateX(${titleX}px)`,
            opacity: titleOpacity,
            textShadow: '0 0 30px rgba(59, 130, 246, 0.3)',
          }}
        >
          Architecture
        </h1>

        {/* Content grid */}
        <div
          style={{
            display: 'flex',
            gap: 60,
            flex: 1,
          }}
        >
          {/* Left: Architecture diagram */}
          <div style={{ flex: 1.2 }}>
            {frame >= 90 && (
              <ArchitectureDiagram layers={architectureLayers} startFrame={90} />
            )}
          </div>

          {/* Right: Code snippets and patterns */}
          <div
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              gap: 40,
            }}
          >
            {/* Code snippets carousel */}
            {frame >= 390 && frame < 540 && (
              <div>
                <CodeBlock
                  code={codeSnippets[0].code}
                  language={codeSnippets[0].language}
                  startFrame={390}
                  animationDuration={60}
                  fontSize={16}
                  showLineNumbers
                />
              </div>
            )}

            {frame >= 540 && (
              <div>
                <CodeBlock
                  code={codeSnippets[1].code}
                  language={codeSnippets[1].language}
                  startFrame={540}
                  animationDuration={60}
                  fontSize={16}
                  showLineNumbers
                />
              </div>
            )}

            {/* Design patterns */}
            {frame >= 690 && (
              <div>
                <div
                  style={{
                    fontSize: 24,
                    fontFamily: fontFamilies.sans,
                    color: colors.textSecondary,
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
                  {designPatterns.map((pattern, i) => {
                    const badgeFrame = Math.max(0, frame - (690 + i * 10));
                    const scale = spring({
                      frame: badgeFrame,
                      fps,
                      from: 0,
                      to: 1,
                      config: {
                        damping: 12,
                        stiffness: 100,
                      },
                    });

                    return (
                      <div
                        key={pattern}
                        style={{
                          transform: `scale(${scale})`,
                          padding: '12px 24px',
                          backgroundColor: colors.cardBg,
                          borderRadius: '24px',
                          fontSize: 18,
                          fontWeight: 'bold',
                          color: colors.architecture.from,
                          fontFamily: fontFamilies.sans,
                          border: `2px solid ${colors.architecture.from}40`,
                          boxShadow: `0 0 20px ${colors.architecture.from}20`,
                        }}
                      >
                        {pattern}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Data flow arrows */}
            {frame >= 900 && (
              <div
                style={{
                  marginTop: 'auto',
                  padding: 30,
                  backgroundColor: colors.cardBg,
                  borderRadius: 16,
                  border: `1px solid ${colors.architecture.from}30`,
                }}
              >
                <div
                  style={{
                    fontSize: 18,
                    fontFamily: fontFamilies.mono,
                    color: colors.textPrimary,
                    lineHeight: 1.8,
                  }}
                >
                  <div style={{ opacity: 0.6 }}>Request Flow:</div>
                  <div style={{ marginTop: 12 }}>
                    MCP Client
                    <span style={{ color: colors.architecture.from, margin: '0 12px' }}>→</span>
                    Server Layer
                    <span style={{ color: colors.architecture.from, margin: '0 12px' }}>→</span>
                    Storage
                    <span style={{ color: colors.architecture.from, margin: '0 12px' }}>→</span>
                    Response
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
