/**
 * Professional scene transition with 3D card flip effect
 */

import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';

interface SceneTransitionProps {
  children: React.ReactNode;
  /** Duration of transition in frames */
  transitionDuration?: number;
  /** Which direction to flip */
  direction?: 'left' | 'right' | 'up' | 'down';
  /** Type of transition */
  type?: 'flip' | 'slide' | 'fade';
}

export const SceneTransition: React.FC<SceneTransitionProps> = ({
  children,
  transitionDuration = 30,
  direction = 'left',
  type = 'flip',
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  // Entry transition (first frames)
  const entryProgress = spring({
    frame,
    fps,
    config: {
      damping: 20,
      stiffness: 80,
    },
    durationInFrames: transitionDuration,
  });

  // Exit transition (last frames)
  const exitFrame = Math.max(0, frame - (durationInFrames - transitionDuration));
  const exitProgress = spring({
    frame: exitFrame,
    fps,
    config: {
      damping: 20,
      stiffness: 80,
    },
    durationInFrames: transitionDuration,
  });

  const isExiting = frame >= durationInFrames - transitionDuration;
  const progress = isExiting ? 1 - exitProgress : entryProgress;

  // Different transition types
  let transform = '';
  let opacity = 1;

  switch (type) {
    case 'flip': {
      const rotation = interpolate(progress, [0, 1], [-90, 0]);
      const axis = direction === 'left' || direction === 'right' ? 'Y' : 'X';
      transform = `perspective(2000px) rotate${axis}(${rotation}deg)`;
      opacity = interpolate(progress, [0, 0.5, 1], [0, 0, 1]);
      break;
    }
    case 'slide': {
      const slideMap = {
        left: [-1920, 0],
        right: [1920, 0],
        up: [0, -1080],
        down: [0, 1080],
      };
      const [from, to] = slideMap[direction];
      const x = direction === 'left' || direction === 'right'
        ? interpolate(progress, [0, 1], [from, to])
        : 0;
      const y = direction === 'up' || direction === 'down'
        ? interpolate(progress, [0, 1], [from, to])
        : 0;
      transform = `translate(${x}px, ${y}px)`;
      break;
    }
    case 'fade': {
      opacity = progress;
      break;
    }
  }

  return (
    <AbsoluteFill
      style={{
        transform,
        opacity,
        transformOrigin: 'center center',
      }}
    >
      {children}
    </AbsoluteFill>
  );
};
