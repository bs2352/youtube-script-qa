import { useState, useEffect } from 'react'
import { Box, TextField, Link } from '@mui/material'
import { YouTubePlayer } from 'react-youtube'

import { TranscriptType } from './types'
import { Loading } from './Loading'
import { s2hms } from '../utils'


interface TranscriptProps {
    vid: string;
    ytplayer: YouTubePlayer;
    transcripts: TranscriptType[] | null;
    setTranscripts: React.Dispatch<React.SetStateAction<TranscriptType[] | null>>;
}

const boxSx = {
    width: "100%",
    margin: "0 auto",
}

const boxQaSx = {
    width: "90%",
    height: "400px",
    overflow: "scroll",
    margin: "0 auto",
    marginBottom: "1em",
}

const textFieldTranscriptSx = {
    width: "95%",
    marginTop: "1em",
}

export function Transcript (props: TranscriptProps) {
    const { vid, ytplayer, transcripts, setTranscripts } = props;

    const [ loading, setLoading ] = useState<boolean>(false);

    useEffect(() => {
        if (transcripts !== null) {
            return; // 字幕がある場合は何もしない。
        }
        setLoading(true);
        setTranscripts(null);
        fetch(
            '/transcript',
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({vid: vid})
            }
        )
        .then((res => {
            if (!res.ok) {
                throw new Error(res.statusText);
            }
            return res.json();
        }))
        .then((res => {
            setTranscripts(res.transcripts);
            setLoading(false);
        }))
        .catch((err) => {
            const errmessage: string = `字幕取得中にエラーが発生しました。${err}`;
            console.error(errmessage);
            alert(errmessage);
            setLoading(false);
        })
    });

    const onClickHandlerTranscript = (time: number) => {
        ytplayer.seekTo(Math.round(time), true);
    }

    const LabelLink = (props: {transcript: TranscriptType}) => {
        const { transcript } = props;
        return (
            <Link
                href="#"
                onClick={()=>onClickHandlerTranscript(transcript.start)}
                underline="always"
                variant='h6'
            >
                {`${s2hms(Math.round(transcript.start))}`}
            </Link>
        )
    }

    return (
        <Box sx={boxSx} id="transcript-box-01" >
            <Box sx={boxQaSx} id="transcript-box-02" >
                { transcripts && transcripts.map((transcript, idx) => {
                    return (
                        <TextField
                            key={`transcript-${idx}`}
                            sx={textFieldTranscriptSx}
                            variant='outlined'
                            label={<LabelLink transcript={transcript} />}
                            defaultValue={transcript.text.replace(/\n/g, ' ')}
                            multiline
                            inputProps={{readOnly: true}}
                        />
                    )
                })}
                { loading && <Loading /> }
                { !loading && !transcripts && <p>字幕がありません。</p> }
            </Box>
        </Box>
    )
}
