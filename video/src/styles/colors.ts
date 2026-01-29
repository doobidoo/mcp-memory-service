/**
 * Color palette for MCP Memory Service Technical Showcase
 * Based on design document: 2026-01-29-remotion-video-design.md
 */

export const colors = {
  // Section themes - each feature card has its own gradient
  performance: {
    from: '#10B981', // Emerald-500
    to: '#059669',   // Emerald-600
  },
  architecture: {
    from: '#3B82F6', // Blue-500
    to: '#1D4ED8',   // Blue-700
  },
  aiml: {
    from: '#8B5CF6', // Violet-500
    to: '#6D28D9',   // Violet-700
  },
  quality: {
    from: '#F59E0B', // Amber-500
    to: '#D97706',   // Amber-600
  },

  // Base colors
  background: '#0F172A',    // Slate-950
  cardBg: '#1E293B',        // Slate-800
  textPrimary: '#F8FAFC',   // Slate-50
  textSecondary: '#94A3B8', // Slate-400
  accent: '#F8FAFC',        // Slate-50

  // Utility colors
  success: '#10B981',
  warning: '#F59E0B',
  error: '#EF4444',
  info: '#3B82F6',
} as const;

/**
 * Generate gradient CSS string
 */
export const gradient = (from: string, to: string, angle = 135) =>
  `linear-gradient(${angle}deg, ${from}, ${to})`;
