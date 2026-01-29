import { Composition } from 'remotion';
import { Video } from './Video';

export const RemotionRoot: React.FC = () => {
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
