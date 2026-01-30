/**
 * Font loading utilities for MCP Memory Service video
 * Loads Google Fonts for consistent typography
 */

import { continueRender, delayRender } from 'remotion';
import { useEffect, useState } from 'react';

/**
 * Google Fonts to load
 */
const FONTS = [
  {
    family: 'JetBrains Mono',
    weights: ['400', '700'],
  },
  {
    family: 'Inter',
    weights: ['400', '700'],
  },
];

/**
 * Load Google Fonts
 * Call this hook in Root.tsx to ensure fonts are loaded before rendering
 */
export const useLoadFonts = () => {
  const [handle] = useState(() => delayRender());

  useEffect(() => {
    const loadFonts = async () => {
      try {
        // Construct Google Fonts URL
        const fontUrls = FONTS.map((font) => {
          const weights = font.weights.join(';');
          const family = font.family.replace(/ /g, '+');
          return `${family}:wght@${weights}`;
        }).join('&family=');

        const googleFontsUrl = `https://fonts.googleapis.com/css2?family=${fontUrls}&display=swap`;

        // Load fonts via CSS
        const link = document.createElement('link');
        link.href = googleFontsUrl;
        link.rel = 'stylesheet';
        document.head.appendChild(link);

        // Wait for fonts to load
        await document.fonts.ready;

        continueRender(handle);
      } catch (error) {
        console.error('Failed to load fonts:', error);
        continueRender(handle); // Continue anyway with fallback fonts
      }
    };

    loadFonts();
  }, [handle]);
};

/**
 * Font family constants
 */
export const fontFamilies = {
  mono: '"JetBrains Mono", monospace',
  sans: '"Inter", sans-serif',
} as const;
