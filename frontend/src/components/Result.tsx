import { Box, Tabs, Tab, styled } from '@mui/material'
import { useEffect, useState } from 'react'
import { YouTubePlayer } from 'react-youtube'

import {
    SummaryRequestBody, SummaryResponseBody, TranscriptType, QaResponseBody, VideoInfoType,
    AgendaType,
} from "../common/types"
import { VideoInfo } from './Results/VideoInfo'
import { Summary } from './Results/Summary'
import { QA } from './Results/QA'
import { Transcript } from './Results/Transcript'


interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

interface ResultProps {
    vid: string;
    ytplayer: YouTubePlayer;
    summaryLoading: boolean;
    setSummaryLoading: React.Dispatch<React.SetStateAction<boolean>>;
    agendaLoading: boolean;
    setAgendaLoading: React.Dispatch<React.SetStateAction<boolean>>;
    qaLoading: boolean;
    setQaLoading: React.Dispatch<React.SetStateAction<boolean>>;
    refreshSummary: boolean;
    setRefreshSummary: React.Dispatch<React.SetStateAction<boolean>>;
}

const ResultContainer = styled(Box)({
    width: "100%",
    margin: "0 auto",
});

const TabsContainer = styled(Box)({
    width: '100%',
    bgcolor: 'background.paper',
    marginBottom: 0,
    marginTop: 1
});

function TabPanel (props: TabPanelProps) {
    const { children, value, index, ...other } = props;
    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            {...other}
        >
            {value === index && (
                <Box sx={{ p: 3 }}>
                    {children}
                </Box>
            )}
        </div>
    );
}


export function Result (props: ResultProps) {
    const {
        vid, ytplayer,
        summaryLoading, setSummaryLoading,
        agendaLoading, setAgendaLoading,
        qaLoading, setQaLoading,
        refreshSummary, setRefreshSummary,
    } = props;

    const [ value, setValue ] = useState<number>(0); // 概要タブ
    const [ videoInfoLoading, setVideoInfoLoading ] = useState<boolean>(false);
    const [ transcriptLoading, setTranscriptLoading ] = useState<boolean>(false);
    const [ videoInfo, setVideoInfo ] = useState<VideoInfoType|null>(null);
    const [ summary, setSummary ] = useState<SummaryResponseBody|null>(null);
    const [ qaQuestion, setQaQuestion ] = useState<string|null>(null);
    const [ qaAnswer, setQaAnswer ] = useState<QaResponseBody|null>(null);
    const [ transcripts, setTranscripts ] = useState<TranscriptType[]|null>(null);
    const [ qaAlignment, setQaAlignment ] = useState<string>('qa');
    const [ summaryAlignment, setSummaryAlignment ] = useState<string>('summary');

    const clearResult = () => {
        setValue(refreshSummary === true ? 1 : 0); // 要約の再作成時は要約タブへ遷移する
        setSummaryAlignment('summary');
        if (refreshSummary === true) {
            return;
        }
        if (qaQuestion !== null) {
            setQaQuestion(null);
        }
        if (qaAnswer !== null) {
            setQaAnswer(null);
        }
        setQaAlignment('qa');
    }

    const fetch_info = () => {
        setVideoInfoLoading(true);
        const requestBody: SummaryRequestBody = {
            vid: vid
        }
        fetch(
            '/info',
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
            setVideoInfo(res);
        }))
        .catch((err) => {
            const errmessage: string = `動画情報の取得中にエラーが発生しました。${err}`;
            console.error(errmessage);
            alert(errmessage);
        })
        .finally(() => {
            setVideoInfoLoading(false);
        })
    }

    const fetch_summary = () => {
        setSummaryLoading(true);
        const requestBody: SummaryRequestBody = {
            vid: vid,
            refresh: refreshSummary,
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
        .then((res => {
            if (!res.ok) {
                throw new Error(res.statusText);
            }
            return res.json();
        }))
        .then((res => {
            setSummary(res);
        }))
        .catch((err) => {
            const errmessage: string = `要約作成中にエラーが発生しました。${err}`;
            console.error(errmessage);
            alert(errmessage);
        })
        .finally(() => {
            setSummaryLoading(false);
        })
    }

    const fetch_transcript = () => {
        setTranscriptLoading(true);
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
        }))
        .catch((err) => {
            const errmessage: string = `字幕取得中にエラーが発生しました。${err}`;
            console.error(errmessage);
            alert(errmessage);
        })
        .finally(() => {
            setTranscriptLoading(false);
        })
    }

    const fetch_agenda = () => {
        setAgendaLoading(true);
        const requestBody: SummaryRequestBody = {
            vid: vid,
        }
        fetch(
            '/agenda',
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
            setSummary(res);
        }))
        .catch((err) => {
            const errmessage: string = `目次のタイムテーブル作成中にエラーが発生しました。${err}`;
            console.error(errmessage);
            alert(errmessage);
        })
        .finally(() => {
            setAgendaLoading(false);
        })
    }

    // vidが変化したら概要、要約、字幕を取得する
    useEffect(() => {
        clearResult();
        if (summaryLoading === false) {
            fetch_summary();
        }
        if (videoInfoLoading === false) {
            fetch_info();
        }
        if (transcriptLoading === false) {
            fetch_transcript();
        }
    }, [vid])

    // 要約の再作成ボタンが押されたら要約を取得する
    useEffect(() => {
        if (refreshSummary === false) {
            return;
        }
        setRefreshSummary(false);
        clearResult();
        fetch_summary();
    }, [refreshSummary])

    // 要約が生成されたらアジェンダ、トピックの順でタイムテーブルを取得する
    useEffect(() => {
        if (summary === null) {
            return;
        }
        const summaryAgenda: AgendaType[] = summary.summary.agenda;
        if (!(summaryAgenda.length > 0 && summaryAgenda[0].time.length > 0)) {
            if (agendaLoading === false) {
                fetch_agenda();
                return;
            }
        }
    }, [summary])

    const onTabChangeHandler = (_: React.SyntheticEvent, value: number) => {
        setValue(value);
    }

    const tabItemList: string[] = [
        '概要','要約', 'QA/検索', '字幕'
    ]

    return (
        <ResultContainer id="result-box-01" >
            <TabsContainer id="result-box-02" >
                <Tabs
                    value={value}
                    onChange={onTabChangeHandler}
                    centered
                >
                    {tabItemList.map((item, idx) =>
                        <Tab key={idx} label={item} />
                    )}
                </Tabs>
            </TabsContainer>
            <TabPanel value={value} index={0}>
                <VideoInfo
                    videoInfo={videoInfo}
                    videoInfoLoading={videoInfoLoading}
                />
            </TabPanel>
            <TabPanel value={value} index={1}>
                <Summary
                    ytplayer={ytplayer}
                    summary={summary}
                    alignment={summaryAlignment}
                    setAlignment={setSummaryAlignment}
                    summaryLoading={summaryLoading}
                    agendaLoading={agendaLoading}
                />
            </TabPanel>
            <TabPanel value={value} index={2}>
                <QA
                    vid={vid}
                    ytplayer={ytplayer}
                    question={qaQuestion}
                    setQuestion={setQaQuestion}
                    answer={qaAnswer}
                    setAnswer={setQaAnswer}
                    alignment={qaAlignment}
                    setAlignment={setQaAlignment}
                    qaLoading={qaLoading}
                    setQaLoading={setQaLoading}
                />
            </TabPanel>
            <TabPanel value={value} index={3}>
                <Transcript
                    ytplayer={ytplayer}
                    transcripts={transcripts}
                    transcriptLoading={transcriptLoading}
                />
            </TabPanel>
        </ResultContainer>
    ) 
}