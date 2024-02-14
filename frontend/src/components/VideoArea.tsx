import { Box, styled } from '@mui/material'
import YouTube, { YouTubeEvent, YouTubePlayer, YouTubeProps } from 'react-youtube'


interface VideoAreaProps {
    vid: string;
    setYtPlayer: React.Dispatch<React.SetStateAction<YouTubePlayer | undefined>>;
}

const VideoAreaContainer = styled(Box)({
    width: "100%",
    margin: "0 auto",
});

export function VideoArea (props: VideoAreaProps) {
    const { vid, setYtPlayer } = props;

    const onReadyHanler: YouTubeProps['onReady'] = (event: YouTubeEvent) => {
        setYtPlayer(event.target);
    }

    return (
        <VideoAreaContainer id="videoarea-box-01">
            <YouTube
                videoId={vid}
                onReady={onReadyHanler}
                opts={{
                    width: "100%",
                    height: "100%",
                }}
                style={{
                    width: "80%",
                    aspectRatio: "16/9",
                    margin: "0 auto"
                }}
            />
        </VideoAreaContainer>
    )
}
