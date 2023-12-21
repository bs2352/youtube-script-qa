import { useState } from 'react'
import YouTube, { YouTubeEvent, YouTubePlayer, YouTubeProps } from 'react-youtube'
import { Box } from '@mui/material'

import { SummaryRequestBody, SummaryResponseBody } from './components/types'
import { Header } from './components/Header'
import { InputVid } from './components/InputVid'
import { Result } from './components/Result'
import './App.css'


function App() {
    const [vid, setVid] = useState<string>('cEynsEWpXdA')
    // @ts-ignore
    const [ytplayer, setYtPlayer] = useState<YouTubePlayer>()
    const [summary, setSummary] = useState<SummaryResponseBody|null>(null)
    const [loading, setLoading] = useState<boolean>(false)

    const onReadyHanler: YouTubeProps['onReady'] = (event: YouTubeEvent) => {
        setYtPlayer(event.target)
        setSummary(null)
        setLoading(true)

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
            <Header />
            <InputVid vid={vid} setVid={setVid} />
            <YouTube
                videoId={vid}
                onReady={onReadyHanler}
            />
            {
                summary && !loading && <Result summary={summary} />
            }
            {
                !summary && loading && <div className='div-loading' />
            }
        </Box>
    )
}

export default App
