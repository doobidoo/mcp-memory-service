import { Sequence, Audio, staticFile } from 'remotion';
import { HeroIntro } from './scenes/HeroIntro';
import { PerformanceSpotlight } from './scenes/PerformanceSpotlight';
import { ArchitectureTour } from './scenes/ArchitectureTour';
import { AIMLIntelligence } from './scenes/AIMLIntelligence';
import { DeveloperExperience } from './scenes/DeveloperExperience';
import { Outro } from './scenes/Outro';
import { SceneTransition } from './components/SceneTransition';

/**
 * Main video composition for MCP Memory Service Technical Showcase
 *
 * Complete 2-minute technical showcase with 6 scenes (faster pace):
 * - HeroIntro: Brain icon + 3D particles (0-10s)
 * - Performance: Speedometer, metrics, benchmarks (10-30s)
 * - Architecture: Layered diagram, code, patterns (30-55s)
 * - AI/ML: Vector space, features, quality tiers (55-80s)
 * - Developer Experience: Code examples, dashboard (80-105s)
 * - Outro: GitHub link, badges, tagline (105-120s)
 *
 * Professional 3D flip transitions throughout
 */
export const Video: React.FC = () => {
  // Transition duration: 30 frames = 1 second overlap
  const transitionDuration = 30;

  return (
    <>
      {/* Background Music */}
      <Audio
        src={staticFile('showreel-music-promo-advertising-opener-vlog-background-intro-theme-261601.mp3')}
        volume={0.5}
      />
      {/* Scene 1: Hero Intro (0-10s) */}
      <Sequence from={0} durationInFrames={300 + transitionDuration}>
        <SceneTransition type="fade">
          <HeroIntro />
        </SceneTransition>
      </Sequence>

      {/* Scene 2: Performance Spotlight (10-30s) */}
      <Sequence from={300 - transitionDuration} durationInFrames={600 + transitionDuration * 2}>
        <SceneTransition type="flip" direction="left">
          <PerformanceSpotlight />
        </SceneTransition>
      </Sequence>

      {/* Scene 3: Architecture Tour (30-55s) */}
      <Sequence from={900 - transitionDuration} durationInFrames={750 + transitionDuration * 2}>
        <SceneTransition type="flip" direction="left">
          <ArchitectureTour />
        </SceneTransition>
      </Sequence>

      {/* Scene 4: AI/ML Intelligence (55-80s) */}
      <Sequence from={1650 - transitionDuration} durationInFrames={750 + transitionDuration * 2}>
        <SceneTransition type="flip" direction="left">
          <AIMLIntelligence />
        </SceneTransition>
      </Sequence>

      {/* Scene 5: Developer Experience (80-105s) */}
      <Sequence from={2400 - transitionDuration} durationInFrames={750 + transitionDuration * 2}>
        <SceneTransition type="flip" direction="left">
          <DeveloperExperience />
        </SceneTransition>
      </Sequence>

      {/* Scene 6: Outro (105-120s) */}
      <Sequence from={3150 - transitionDuration} durationInFrames={450 + transitionDuration}>
        <SceneTransition type="fade">
          <Outro />
        </SceneTransition>
      </Sequence>
    </>
  );
};
