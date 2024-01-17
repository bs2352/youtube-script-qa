import { useState } from 'react'
import { YouTubePlayer } from 'react-youtube'
import { Box } from '@mui/material'

import { SummaryResponseBody, VideoInfoType } from './components/types'
// import { Header } from './components/Header'
import { InputVid } from './components/InputVid'
import { VideoArea } from './components/VideoArea'
import { Result } from './components/Result'
import './App.css'


function App() {
    const [ vid, setVid ] = useState<string>('cEynsEWpXdA');
    // @ts-ignore
    const [ ytplayer, setYtPlayer ] = useState<YouTubePlayer>();
    const [ summary, setSummary ] = useState<SummaryResponseBody|null>(null);
    const [ summaryLoading, setSummaryLoading ] = useState<boolean>(false);
    const [ videoInfo, setVideInfo ] = useState<VideoInfoType|null>(null);
    const [ videoInfoLoading, setVideoInfoLoading ] = useState<boolean>(false);

    return (
        <Box sx={{width: "70%", margin: "0 auto"}} id="app-box-01">
            {/* <Header /> */}
            <InputVid
                vid={vid}
                setVid={setVid}
                setSummary={setSummary}
                summaryLoading={summaryLoading}
                setSummaryLoading={setSummaryLoading}
            />
            <VideoArea
                vid={vid}
                setYtPlayer={setYtPlayer}
                setSummary={setSummary}
                setSummaryLoading={setSummaryLoading}
                setVideoInfo={setVideInfo}
                setVideoInfoLoading={setVideoInfoLoading}
            />
            { ytplayer &&
                <Result
                    summary={summary}
                    setSummary={setSummary}
                    summaryLoading={summaryLoading}
                    videoInfo={videoInfo}
                    videoInfoLoading={videoInfoLoading}
                    vid={vid}
                    ytplayer={ytplayer}
                />
            }
        </Box>
    )
}

export default App
