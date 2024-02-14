import { Box, TextField, Link, styled } from '@mui/material'
import { YouTubePlayer } from 'react-youtube'

import { TranscriptType } from '../../common/types'
import { s2hms } from '../../common/utils'
import { Loading } from '../Loading'


interface TranscriptProps {
    ytplayer: YouTubePlayer;
    transcripts: TranscriptType[] | null;
    transcriptLoading: boolean;
}

const TranscriptContainer = styled(Box)({
    width: "100%",
    margin: "0 auto",
});

const TranscriptInternalContainer = styled(Box)({
    width: "90%",
    height: "400px",
    overflow: "scroll",
    margin: "0 auto",
    marginBottom: "1em",
});

const TranscriptTextField = styled(TextField)({
    width: "95%",
    marginTop: "1em",
});


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
        <TranscriptContainer id="transcript-box-01" >
            <TranscriptInternalContainer id="transcript-box-02" >
                { transcripts && transcripts.map((transcript, idx) => {
                    return (
                        <TranscriptTextField
                            key={`transcript-${idx}`}
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
            </TranscriptInternalContainer>
        </TranscriptContainer>
    )
}
