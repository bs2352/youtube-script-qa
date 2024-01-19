import { useState } from 'react'
import { YouTubePlayer } from 'react-youtube'
import { Box } from '@mui/material'

// import { Header } from './components/Header'
import { InputVid } from './components/InputVid'
import { VideoArea } from './components/VideoArea'
import { Result } from './components/Result'
import './App.css'


function App() {
    const [ vid, setVid ] = useState<string>('cEynsEWpXdA');
    // @ts-ignore
    const [ ytplayer, setYtPlayer ] = useState<YouTubePlayer>();
    const [ videoInfoLoading, setVideoInfoLoading ] = useState<boolean>(false);
    const [ summaryLoading, setSummaryLoading ] = useState<boolean>(false);
    const [ agendaLoading, setAgendaLoading ] = useState<boolean>(false);
    const [ topicLoading, setTopicLoading ] = useState<boolean>(false);
    const [ qaLoading, setQaLoading ] = useState<boolean>(false);
    const [ updateSummary, setUpdateSummary ] = useState<boolean>(false);

    return (
        <Box sx={{width: "70%", margin: "0 auto"}} id="app-box-01">
            {/* <Header /> */}
            <InputVid
                vid={vid}
                setVid={setVid}
                loading={ videoInfoLoading || summaryLoading || agendaLoading || topicLoading || qaLoading }
                setUpdateSummary={setUpdateSummary}
            />
            <VideoArea
                vid={vid}
                setYtPlayer={setYtPlayer}
            />
            { ytplayer &&
                <Result
                    vid={vid}
                    ytplayer={ytplayer}
                    videoInfoLoading={videoInfoLoading}
                    setVideoInfoLoading={setVideoInfoLoading}
                    summaryLoading={summaryLoading}
                    setSummaryLoading={setSummaryLoading}
                    agendaLoading={agendaLoading}
                    setAgendaLoading={setAgendaLoading}
                    topicLoading={topicLoading}
                    setTopicLoading={setTopicLoading}
                    qaLoading={qaLoading}
                    setQaLoading={setQaLoading}
                    updateSummary={updateSummary}
                    setUpdateSummary={setUpdateSummary}
                />
            }
        </Box>
    )
}

export default App
