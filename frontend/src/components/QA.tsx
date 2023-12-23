import { useRef, useState } from 'react'

import { Box, TextField, IconButton } from '@mui/material'
import { PlayArrow } from '@mui/icons-material'

import { QaRequestBody, QaAnswerSource, QaResponseBody } from './types'
import { Loading } from './Loading'


interface QAProps {
    vid: string;
}

const boxSx = {
    width: "100%",
    margin: "0 auto",
}

const boxQaSx = {
    width: "80%",
    margin: "0 auto",
    marginBottom: "1em"
}

const textFieldQuestionSx = {
    width: "80%",
    verticalAlign: "bottom",
}

const iconButtonSendSx = {
    verticalAlign: "bottom",
}

const textFieldAnswerSx = {
    width: "85%",
    margin: "0 auto",
    marginTop: "1em",
    padding: "0.5em",
    paddingTop: "0",
    // readOnly: "true",
    pointerEvents: "none"
}


export function QA (props: QAProps) {
    const { vid } = props;

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
                value={answer?.answer}
                sx={textFieldAnswerSx}
                multiline
                id="qa-answer-01"
                // disabled
            />
        )
    }

    return (
        <Box sx={boxSx} >
            <Box sx={boxQaSx}>
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
                />
                <IconButton
                    sx={iconButtonSendSx}
                    onClick={onClickHandlerSendQuestion}
                    disabled={disabledSendButton}
                >
                    <PlayArrow fontSize='large' />
                </IconButton>
            </Box>
            <Box sx={boxQaSx} >
                {loading && <Loading />}
                {answer && <Answer />}
            </Box>
        </Box>
    )
}