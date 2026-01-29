# MCP Memory Service - Technical Showcase Video

Remotion-based technical showcase video highlighting the architecture, performance, and AI/ML features.

## Quick Start

```bash
npm install
npm run extract-data  # Extract real metrics from project
npm run dev           # Open Remotion Studio
```

## Structure

- `src/scenes/` - Individual feature cards (6 scenes)
- `src/components/` - Reusable visual components
- `src/data/` - Real and mock data sources
- `public/` - Fonts, logos, static assets

## Rendering

- **Full video:** `npm run build` → 3-4 minutes at 1080p
- **Short version:** `npm run build:short` → 60s vertical for social
- **Preview GIF:** `npm run build:gif` → smaller preview file

## Customization

Edit `src/data/mockData.ts` for visual tweaks without changing real data.
Edit individual scene files in `src/scenes/` for timing or content changes.

## Performance

- Rendering time: ~5-10 minutes for full video (depends on hardware)
- Uses concurrent rendering for speed
- 3D scenes are the most expensive (60-70% of render time)

## Design Document

See `../../docs/plans/2026-01-29-remotion-video-design.md` for complete design specification.
