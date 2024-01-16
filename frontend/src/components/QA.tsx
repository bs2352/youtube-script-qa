import { useRef, useState } from 'react'

import { Box, TextField, IconButton, Link, Typography, ToggleButtonGroup, ToggleButton, ButtonGroup } from '@mui/material'
import { Send, Clear } from '@mui/icons-material'
import { YouTubePlayer } from 'react-youtube'

import { QaRequestBody, QaAnswerSource, QaResponseBody } from './types'
import { Loading } from './Loading'
import { hms2s } from './utils'


interface QAProps {
    vid: string;
    ytplayer: YouTubePlayer;
    question: string | null;
    setQuestion: React.Dispatch<React.SetStateAction<string|null>>;
    answer: QaResponseBody | null;
    setAnswer: React.Dispatch<React.SetStateAction<QaResponseBody | null>>;
    alignment: string;
    setAlignment: React.Dispatch<React.SetStateAction<string>>;
}

const boxSx = {
    width: "100%",
    margin: "0 auto",
}

const boxQuestionSx = {
    width: "90%",
    margin: "0 auto",
    marginBottom: "1em",
}

const boxAnswerSx = {
    width: "85%",
    margin: "0 auto",
    marginBottom: "1em",
    // height: "300px",
    // overflowY: "scroll",
}

const textFieldQuestionSx = {
    width: "87%",
    verticalAlign: "bottom",
}

const iconButtonSendSx = {
    verticalAlign: "bottom",
    marginLeft: "5px",
}

const textFieldAnswerSx = {
    width: "98%",
    margin: "0 auto",
    marginTop: "1em",
    // pointerEvents: "none",
}

const boxToggleButton = {
    marginBottom: "20px",
    marginLeft: '4%',
    textAlign: 'left',
}


export function QA (props: QAProps) {
    const { vid, ytplayer, question, setQuestion, answer, setAnswer, alignment, setAlignment  } = props;

    const [ disabledSendButton, setDisabledSendButton] = useState<boolean>(true);
    const [ loading, setLoading ] = useState<boolean>(false);

    const questionRef = useRef<HTMLInputElement>(null);


    const onChangeHandlerMode = (
        _: React.MouseEvent<HTMLElement, MouseEvent>,
        newAlignment: string|null
    ) => {
        setAlignment(newAlignment !== null ? newAlignment : 'qa');
    }

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

    const onClickHandlerClearQuestion = () => {
        setQuestion(null);
        setAnswer(null);
        if (questionRef.current === undefined) {
            return;
        }
        questionRef.current!.value = "";
    }

    const onClickHandlerSendQuestion = () => {
        if (questionRef.current === undefined) {
            return;
        }
        setLoading(true);
        setDisabledSendButton(true);
        setQuestion(null);
        setAnswer(null);
        const questionInput = questionRef.current as HTMLInputElement;
        const url: string = alignment !== 'retrieve' ? '/qa' : '/retrieve';
        const requestBody: QaRequestBody = alignment !== 'retrieve' ? {
            vid: vid,
            question: questionInput.value,
            ref_sources: 3,
        } : {
            vid: vid,
            query: questionInput.value,
            ref_sources: 5,
        }
        fetch(
            url,
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
            setQuestion(questionInput.value);
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

    const onKeyDownHandlerQuestion = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter' && event.shiftKey) {
            event.preventDefault();
            onClickHandlerSendQuestion();
        }
    }

    const Answer = () => {
        if (!answer?.answer) {
            return (<></>)
        }
        const Label = () => {
            return (
                <Typography variant='h6'>
                    回答
                </Typography>
            )
        }
        return (
            <TextField
                variant='outlined'
                label={<Label />}
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
        const LabelLink = (props: {idx: number, source: QaAnswerSource}) => {
            const { idx, source } = props;
            return (
                <Link
                    href="#"
                    onClick={()=>onClickHandlerSource(source.time)}
                    underline="always"
                    variant='h6'
                >
                    {`[${idx+1}/${sorted_sources.length}]　${source.time}（スコア：${Math.round(source.score*1000)/1000}）`}
                </Link>
            )
        }
        const sourceList = sorted_sources.map((source: QaAnswerSource, idx: number) => {
            return (
                <TextField
                    key={idx}
                    sx={textFieldAnswerSx}
                    label={<LabelLink idx={idx} source={source} />}
                    variant="outlined"
                    defaultValue={source.source}
                    multiline
                    rows={4}
                    inputProps={{readOnly: true}}
                />
            )
        })

        return (
            <>{sourceList}</>
        )
    }

    const QuestionLebel = () => {
        return (
            <Typography variant='h6'>
                { alignment !== 'retrieve' ? '質問' : '検索クエリ' }
            </Typography>
        )
    }

    const makePlaceholder = () => {
        const type: string = alignment !== 'retrieve' ? "質問" : "検索クエリ";
        return `${type}を入力してください。(Shift + Enterで送信します）`;
    }

    return (
        <Box sx={boxSx} >
            <Box sx={boxQuestionSx} id="qa-box-02" >
                <Box sx={boxToggleButton} >
                    <ToggleButtonGroup
                        value={alignment}
                        exclusive
                        size='small'
                        onChange={onChangeHandlerMode}
                    >
                        <ToggleButton value="qa">QA</ToggleButton>
                        <ToggleButton value="retrieve">検索</ToggleButton>
                    </ToggleButtonGroup>
                </Box>
                <Box>
                    <TextField
                        label={<QuestionLebel/>}
                        variant="outlined"
                        placeholder={makePlaceholder()}
                        inputRef={questionRef}
                        multiline
                        rows={3}
                        sx={textFieldQuestionSx}
                        onChange={onChangeHandlerQuestion}
                        onKeyDown={onKeyDownHandlerQuestion}
                        InputLabelProps={{shrink: true}}
                        defaultValue={question}
                    />
                    <ButtonGroup orientation='vertical'>
                        <IconButton
                            sx={iconButtonSendSx}
                            onClick={onClickHandlerClearQuestion}
                            size='small'
                        >
                            <Clear fontSize='medium' />
                        </IconButton>
                        <IconButton
                            sx={iconButtonSendSx}
                            onClick={onClickHandlerSendQuestion}
                            disabled={disabledSendButton}
                            size='small'
                        >
                            <Send fontSize='medium' />
                        </IconButton>
                    </ButtonGroup>
                </Box>
            </Box>
            <Box sx={boxAnswerSx} id="qa-box-03" >
                {loading && <Loading />}
                {answer?.answer && <Answer />}
                {answer && <Sources />}
            </Box>
        </Box>
    )
}