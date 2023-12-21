import { useState, useRef } from 'react'
import YouTube, { YouTubeEvent, YouTubePlayer, YouTubeProps } from 'react-youtube'
import { Box } from '@mui/material'

import { SummaryRequestBody, SummaryResponseBody } from './components/types'
import { Header } from './components/Header'
import { Input } from './components/Input'
import { Result } from './components/Result'
import './App.css'


function App() {
    const [vid, setVid] = useState<string>('cEynsEWpXdA')
    // @ts-ignore
    const [ytplayer, setYtPlayer] = useState<YouTubePlayer>()
    const [summary, setSummary] = useState<SummaryResponseBody|null>(null)
    const [loading, setLoading] = useState<boolean>(false)

    const refInputTextVid = useRef<HTMLInputElement>(null)

    const onKeyDownHandlerVid = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter') {
            setVid(refInputTextVid.current!.value)
        }
    }

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

    const vidInputBoxSx = {
        padding: "2em",
    }

    return (
        <Box>
            <Header />
            <Input />
            <Box sx={vidInputBoxSx}>
                Youtube Video ID
                <input
                    type="text"
                    defaultValue={vid}
                    ref={refInputTextVid}
                    onKeyDown={onKeyDownHandlerVid}
                    className='input-vid'
                />
            </Box>
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
