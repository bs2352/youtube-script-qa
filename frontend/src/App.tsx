import { useState, useRef } from 'react'
import YouTube, { YouTubeEvent, YouTubePlayer, YouTubeProps } from 'react-youtube'

import { SummaryRequestBody, SummaryResponseBody } from './components/types'
import { Result } from './components/Result'
import './App.css'


function App() {
    const [vid, setVid] = useState<string>('cEynsEWpXdA')
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
        .catch((err) => console.log(err))
    }

    return (
        <>
            <h1>Youtube Supporter</h1>
            <div className='div-vid'>
                Youtube Video ID
                <input
                    type="text"
                    defaultValue={vid}
                    ref={refInputTextVid}
                    onKeyDown={onKeyDownHandlerVid}
                    className='input-vid'
                />
            </div>
            <YouTube
                videoId={vid}
                onReady={onReadyHanler}
            />
            {
                summary && <Result summary={summary} />
            }
            {
                !summary && loading && <div className='div-loading' />
            }
        </>
    )
}

export default App
