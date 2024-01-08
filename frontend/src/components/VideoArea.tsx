import { Box } from '@mui/material'
import YouTube, { YouTubeEvent, YouTubePlayer, YouTubeProps } from 'react-youtube'

import { SummaryResponseBody, SummaryRequestBody } from './types'


interface VideoAreaProps {
    vid: string;
    setYtPlayer: React.Dispatch<React.SetStateAction<YouTubePlayer | undefined>>;
    setSummary: React.Dispatch<React.SetStateAction<SummaryResponseBody | null>>;
    setLoading: React.Dispatch<React.SetStateAction<boolean>>;
}

const boxSx = {
    width: "100%",
    margin: "0 auto",
}

export function VideoArea (props: VideoAreaProps) {
    const { vid, setYtPlayer, setSummary, setLoading } = props;

    const onReadyHanler: YouTubeProps['onReady'] = (event: YouTubeEvent) => {
        setYtPlayer(event.target);
        setSummary(null);
        setLoading(true);

        if (vid === "") {
            setLoading(false);
            return;
        }

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
        .then((res => {
            if (!res.ok) {
                throw new Error(res.statusText);
            }
            return res.json();
        }))
        .then((res => {
            setSummary(res);
            setLoading(false);
        }))
        .catch((err) => {
            const errmessage: string = `要約作成中にエラーが発生しました。${err}`;
            console.error(errmessage);
            alert(errmessage);
            setLoading(false);
        })
    }

    return (
        <Box sx={boxSx} id="videoarea-box-01">
            <YouTube
                videoId={vid}
                onReady={onReadyHanler}
            />
        </Box>
    )
}
