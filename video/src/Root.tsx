import { Composition } from 'remotion';
import { Video } from './Video';
import { useLoadFonts } from './styles/fonts';

export const RemotionRoot: React.FC = () => {
  // Load fonts before rendering
  useLoadFonts();

  return (
    <>
      <Composition
        id="MCPMemoryShowcase"
        component={Video}
        durationInFrames={5400} // 180 seconds at 30fps
        fps={30}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
    </>
  );
};
