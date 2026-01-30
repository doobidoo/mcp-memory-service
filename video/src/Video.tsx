import { Sequence } from 'remotion';
import { HeroIntro } from './scenes/HeroIntro';
import { PerformanceSpotlight } from './scenes/PerformanceSpotlight';
import { ArchitectureTour } from './scenes/ArchitectureTour';
import { AIMLIntelligence } from './scenes/AIMLIntelligence';
import { SceneTransition } from './components/SceneTransition';

/**
 * Main video composition for MCP Memory Service Technical Showcase
 *
 * Timeline:
 * - 0-450 frames (0-15s): HeroIntro
 * - 450-1500 frames (15-50s): PerformanceSpotlight
 * - 1500-2700 frames (50-90s): ArchitectureTour
 * - 2700-4050 frames (90-135s): AIMLIntelligence
 * - 4050-4950 frames (135-165s): DeveloperExperience (TODO)
 * - 4950-5400 frames (165-180s): Outro (TODO)
 *
 * Transitions:
 * - 30 frame (1s) transitions between scenes
 * - 3D flip effect for professional look
 */
export const Video: React.FC = () => {
  return (
    <>
      {/* Scene 1: Hero Intro (0-15s) */}
      <Sequence from={0} durationInFrames={450}>
        <SceneTransition type="fade">
          <HeroIntro />
        </SceneTransition>
      </Sequence>

      {/* Scene 2: Performance Spotlight (15-50s) */}
      <Sequence from={450} durationInFrames={1050}>
        <SceneTransition type="flip" direction="left">
          <PerformanceSpotlight />
        </SceneTransition>
      </Sequence>

      {/* Scene 3: Architecture Tour (50-90s) */}
      <Sequence from={1500} durationInFrames={1200}>
        <SceneTransition type="flip" direction="left">
          <ArchitectureTour />
        </SceneTransition>
      </Sequence>

      {/* Scene 4: AI/ML Intelligence (90-135s) */}
      <Sequence from={2700} durationInFrames={1350}>
        <SceneTransition type="flip" direction="left">
          <AIMLIntelligence />
        </SceneTransition>
      </Sequence>

      {/* TODO: Add remaining scenes */}
    </>
  );
};
