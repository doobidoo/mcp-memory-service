/**
 * CodeBlock with syntax highlighting
 * Character-by-character reveal animation
 */

import { interpolate, useCurrentFrame } from 'remotion';
import { Highlight, themes } from 'prism-react-renderer';

interface CodeBlockProps {
  code: string;
  language: string;
  startFrame: number;
  animationDuration: number;
  showLineNumbers?: boolean;
  fontSize?: number;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({
  code,
  language,
  startFrame,
  animationDuration,
  showLineNumbers = false,
  fontSize = 16,
}) => {
  const frame = useCurrentFrame();
  const relativeFrame = Math.max(0, frame - startFrame);

  // Character reveal progress
  const progress = Math.min(1, relativeFrame / animationDuration);
  const totalChars = code.length;
  const visibleChars = Math.floor(progress * totalChars);
  const visibleCode = code.slice(0, visibleChars);

  // Opacity fade-in
  const opacity = interpolate(relativeFrame, [0, 15], [0, 1], {
    extrapolateRight: 'clamp',
  });

  return (
    <div style={{ opacity, fontFamily: 'JetBrains Mono' }}>
      <Highlight theme={themes.nightOwl} code={visibleCode} language={language}>
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <pre
            className={className}
            style={{
              ...style,
              padding: '24px',
              borderRadius: '12px',
              fontSize,
              lineHeight: '1.6',
              backgroundColor: '#011627',
              border: '1px solid #1E293B',
            }}
          >
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                {showLineNumbers && (
                  <span
                    style={{
                      color: '#5c6370',
                      marginRight: '24px',
                      userSelect: 'none',
                      minWidth: '24px',
                      display: 'inline-block',
                    }}
                  >
                    {i + 1}
                  </span>
                )}
                {line.map((token, key) => (
                  <span key={key} {...getTokenProps({ token })} />
                ))}
              </div>
            ))}
          </pre>
        )}
      </Highlight>
    </div>
  );
};
