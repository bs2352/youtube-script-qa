import { useEffect } from 'react';
import { Box } from '@mui/material'
import YouTube, { YouTubeEvent, YouTubePlayer, YouTubeProps } from 'react-youtube'

import { SummaryResponseBody, SummaryRequestBody, VideoInfoType } from './types'


interface VideoAreaProps {
    vid: string;
    setYtPlayer: React.Dispatch<React.SetStateAction<YouTubePlayer | undefined>>;
    setSummary: React.Dispatch<React.SetStateAction<SummaryResponseBody | null>>;
    setSummaryLoading: React.Dispatch<React.SetStateAction<boolean>>;
    setVideoInfo: React.Dispatch<React.SetStateAction<VideoInfoType | null>>;
    setVideoInfoLoading: React.Dispatch<React.SetStateAction<boolean>>;
}

const boxSx = {
    width: "100%",
    margin: "0 auto",
}

export function VideoArea (props: VideoAreaProps) {
    const { vid, setYtPlayer, setSummary, setSummaryLoading, setVideoInfo, setVideoInfoLoading } = props;

    useEffect(() => {
        if (vid === "") {
            return;
        }
        setSummary(null);
        setSummaryLoading(true);
        setVideoInfo(null);
        setVideoInfoLoading(true);
        const requestBody: SummaryRequestBody = {
            vid: vid
        }
        // 要約を取得
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
            setSummaryLoading(false);
        }))
        .catch((err) => {
            const errmessage: string = `要約作成中にエラーが発生しました。${err}`;
            console.error(errmessage);
            alert(errmessage);
            setSummaryLoading(false);
        })
        // 動画情報を取得
        fetch(
            '/info',
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
            setVideoInfo(res);
            setVideoInfoLoading(false);
        }))
        .catch((err) => {
            const errmessage: string = `動画情報の取得中にエラーが発生しました。${err}`;
            console.error(errmessage);
            alert(errmessage);
            setVideoInfoLoading(false);
        })
    }, [vid]);

    const onReadyHanler: YouTubeProps['onReady'] = (event: YouTubeEvent) => {
        setYtPlayer(event.target);
    }

    return (
        <Box sx={boxSx} id="videoarea-box-01">
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
        </Box>
    )
}
