import { Sequence } from 'remotion';
import { HeroIntro } from './scenes/HeroIntro';

/**
 * Main video composition for MCP Memory Service Technical Showcase
 *
 * Timeline:
 * - 0-450 frames (0-15s): HeroIntro
 * - 450-1500 frames (15-50s): PerformanceSpotlight (TODO)
 * - 1500-2700 frames (50-90s): ArchitectureTour (TODO)
 * - 2700-4050 frames (90-135s): AIMLIntelligence (TODO)
 * - 4050-4950 frames (135-165s): DeveloperExperience (TODO)
 * - 4950-5400 frames (165-180s): Outro (TODO)
 */
export const Video: React.FC = () => {
  return (
    <>
      {/* Scene 1: Hero Intro (0-15s) */}
      <Sequence from={0} durationInFrames={450}>
        <HeroIntro />
      </Sequence>

      {/* TODO: Add remaining scenes */}
    </>
  );
};
