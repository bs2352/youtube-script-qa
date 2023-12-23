import { useState } from 'react'
import { YouTubePlayer } from 'react-youtube'
import { Box } from '@mui/material'

import { SummaryResponseBody } from './components/types'
import { Header } from './components/Header'
import { InputVid } from './components/InputVid'
import { VideoArea } from './components/VideoArea'
import { Result } from './components/Result'
import { Loading }  from './components/Loading'
import './App.css'


function App() {
    const [vid, setVid] = useState<string>('cEynsEWpXdA')
    // @ts-ignore
    const [ytplayer, setYtPlayer] = useState<YouTubePlayer>()
    const [summary, setSummary] = useState<SummaryResponseBody|null>(null)
    const [loading, setLoading] = useState<boolean>(false)

    return (
        <Box sx={{width: "70%", margin: "0 auto"}} id="app-box-01">
            <Header />
            <InputVid vid={vid} setVid={setVid} />
            <VideoArea
                vid={vid}
                setYtPlayer={setYtPlayer}
                setSummary={setSummary}
                setLoading={setLoading}
            />
            { !loading && summary && <Result summary={summary} vid={vid} /> }
            { loading && <Loading /> }
        </Box>
    )
}

export default App
