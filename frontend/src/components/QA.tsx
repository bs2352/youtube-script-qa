import { useRef, useState } from 'react'

import { Box, TextField, IconButton, Link } from '@mui/material'
import { Send } from '@mui/icons-material'
import { YouTubePlayer } from 'react-youtube'

import { QaRequestBody, QaAnswerSource, QaResponseBody } from './types'
import { Loading } from './Loading'
import { hms2s } from '../utils'


interface QAProps {
    vid: string;
    ytplayer: YouTubePlayer;
}

const boxSx = {
    width: "100%",
    margin: "0 auto",
}

const boxQaSx = {
    width: "80%",
    margin: "0 auto",
    marginBottom: "1em",
}

const textFieldQuestionSx = {
    width: "80%",
    verticalAlign: "bottom",
}

const iconButtonSendSx = {
    verticalAlign: "bottom",
    marginLeft: "5px",
}

const textFieldAnswerSx = {
    width: "87%",
    margin: "0 auto",
    marginTop: "1em",
    // pointerEvents: "none",
}


export function QA (props: QAProps) {
    const { vid, ytplayer } = props;

    const [ disabledSendButton, setDisabledSendButton] = useState<boolean>(true);
    const [ loading, setLoading ] = useState<boolean>(false);
    const [ answer, setAnswer ] = useState<QaResponseBody|null>(null);

    const questionRef = useRef<HTMLInputElement>(null);

    const onChangeHandlerQuestion = () => {
        if (questionRef.current === undefined) {
            return;
        }
        const questionInput = questionRef.current as HTMLInputElement;
        if (questionInput.value.length > 0) {
            setDisabledSendButton(false);
        } else {
            setDisabledSendButton(true);
        }
    }

    const onClickHandlerSendQuestion = () => {
        if (questionRef.current === undefined) {
            return;
        }
        setLoading(true);
        setDisabledSendButton(true);
        setAnswer(null);
        const questionInput = questionRef.current as HTMLInputElement;
        const requestBody: QaRequestBody = {
            vid: vid,
            question: questionInput.value,
            ref_source: 3,
        }
        fetch(
            '/qa',
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
            setAnswer(res);
            setLoading(false);
            setDisabledSendButton(false);
        }))
        .catch((err) => {
            const errmessage: string = `回答作成中にエラーが発生しました。${err}`;
            console.error(errmessage);
            alert(errmessage);
            setLoading(false);
            setDisabledSendButton(false);
        })
    }

    const Answer = () => {
        return (
            <TextField
                variant='outlined'
                label='回答'
                defaultValue={answer?.answer}
                sx={textFieldAnswerSx}
                multiline
                id="qa-answer-01"
                inputProps={{readOnly: true}}
            />
        )
    }

    const onClickHandlerSource = (time: string) => {
        // alert(time);
        ytplayer.seekTo(hms2s(time), true);
    }

    const LabelLink = (props: {source: QaAnswerSource}) => {
        const { source } = props;
        return (
            <Link
                href="#"
                onClick={()=>onClickHandlerSource(source.time)}
                underline="hover"
            >
                {`${source.time} （スコア：${Math.round(source.score*1000)/1000}）`}
            </Link>
        )
    }

    const Sources = () => {
        if (!answer || !answer.sources) {
            return <></>;
        }
        // とりあえず時間順で並べる（スコア順でよい？）
        const sorted_sources: QaAnswerSource[] = answer.sources.sort((a, b) => {
            if (a.time > b.time) {
                return 1;
            } else if (a.time < b.time) {
                return -1;
            } else {
                return 0;
            }
        });
        const sourceList = sorted_sources.map((source: QaAnswerSource, idx: number) => {
            return (
                <TextField
                    key={idx}
                    sx={textFieldAnswerSx}
                    label={<LabelLink source={source} />}
                    variant="outlined"
                    defaultValue={source.source}
                    multiline
                    rows={5}
                    inputProps={{readOnly: true}}
                />
            )
        })

        return (
            <>{sourceList}</>
        )
    }


    return (
        <Box sx={boxSx} >
            <Box sx={boxQaSx} id="qa-box-02" >
                <TextField
                    label="質問"
                    variant="outlined"
                    placeholder='質問を入力してください。'
                    inputRef={questionRef}
                    multiline
                    rows={3}
                    sx={textFieldQuestionSx}
                    size="small"
                    onChange={onChangeHandlerQuestion}
                    InputLabelProps={{shrink: true}}
                />
                <IconButton
                    sx={iconButtonSendSx}
                    onClick={onClickHandlerSendQuestion}
                    disabled={disabledSendButton}
                    size='small'
                >
                    <Send fontSize='large' />
                </IconButton>
            </Box>
            <Box sx={boxQaSx} id="qa-box-03" >
                {loading && <Loading />}
                {answer && <Answer />}
                {answer && <Sources />}
            </Box>
        </Box>
    )
}