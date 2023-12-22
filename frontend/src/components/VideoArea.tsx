import { Box } from '@mui/material'
import YouTube, { YouTubeEvent, YouTubePlayer, YouTubeProps } from 'react-youtube'

import { SummaryResponseBody, SummaryRequestBody } from './types'


interface VideoAreaProps {
    vid: string;
    setYtPlayer: React.Dispatch<React.SetStateAction<YouTubePlayer | undefined>>;
    setSummary: React.Dispatch<React.SetStateAction<SummaryResponseBody | null>>;
    setLoading: React.Dispatch<React.SetStateAction<boolean>>;
}

export function VideoArea (props: VideoAreaProps) {
    const { vid, setYtPlayer, setSummary, setLoading } = props;

    const onReadyHanler: YouTubeProps['onReady'] = (event: YouTubeEvent) => {
        setYtPlayer(event.target);
        setSummary(null);
        setLoading(true);

        const requestBody: SummaryRequestBody = {
            vid: vid
        }
        fetch(
            '/summary',
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            }
        )
        .then((res => res.json()))
        .then((res => {
            setSummary(res);
            setLoading(false);
        }))
        .catch((err) => {
            console.log(err);
            alert('要約作成中にエラーが発生しました。');
            setLoading(false);
        })
    }

    return (
        <Box>
            <YouTube
                videoId={vid}
                onReady={onReadyHanler}
            />
        </Box>
    )
}
