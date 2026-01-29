/**
 * Animation configuration presets for MCP Memory Service video
 */

import type { SpringConfig } from 'remotion';

/**
 * Spring configurations for different animation styles
 */
export const springConfigs: Record<string, SpringConfig> = {
  // Fast, energetic animations (Performance section)
  performance: {
    damping: 20,
    stiffness: 300,
    mass: 1,
  },

  // Smooth, controlled animations (Architecture section)
  architecture: {
    damping: 15,
    stiffness: 100,
    mass: 1,
  },

  // Organic, flowing animations (AI/ML section)
  aiml: {
    damping: 12,
    stiffness: 80,
    mass: 1,
  },

  // Bouncy entrance animations
  entrance: {
    damping: 12,
    stiffness: 100,
    overshootClamping: false,
  },

  // Gentle fade animations
  fade: {
    damping: 20,
    stiffness: 80,
    mass: 1,
  },
} as const;

/**
 * Common easing functions
 */
export const easings = {
  easeOut: [0.25, 0.1, 0.25, 1] as const,
  easeIn: [0.42, 0, 1, 1] as const,
  easeInOut: [0.42, 0, 0.58, 1] as const,
  linear: [0, 0, 1, 1] as const,
} as const;
