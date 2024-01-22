import { Box, TextField, Link } from '@mui/material'
import { YouTubePlayer } from 'react-youtube'

import { TranscriptType } from './types'
import { Loading } from './Loading'
import { s2hms } from './utils'


interface TranscriptProps {
    ytplayer: YouTubePlayer;
    transcripts: TranscriptType[] | null;
    transcriptLoading: boolean;
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
    const { ytplayer, transcripts, transcriptLoading } = props;

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
                { transcriptLoading && <Loading /> }
                { !transcriptLoading && !transcripts && <p>字幕がありません。</p> }
            </Box>
        </Box>
    )
}
