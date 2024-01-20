import { Box, Tabs, Tab } from '@mui/material'
import { useEffect, useState } from 'react'
import { YouTubePlayer } from 'react-youtube'

import { SummaryResponseBody, TranscriptType, QaResponseBody, VideoInfoType } from "./types"
import { VideoInfo } from './VideoInfo'
import { Summary } from './Summary'
import { QA } from './QA'
import { Transcript } from './Transcript'


interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

interface ResultProps {
    vid: string;
    ytplayer: YouTubePlayer;
    videoInfoLoading: boolean;
    setVideoInfoLoading: React.Dispatch<React.SetStateAction<boolean>>;
    summaryLoading: boolean;
    setSummaryLoading: React.Dispatch<React.SetStateAction<boolean>>;
    agendaLoading: boolean;
    setAgendaLoading: React.Dispatch<React.SetStateAction<boolean>>;
    topicLoading: boolean;
    setTopicLoading: React.Dispatch<React.SetStateAction<boolean>>;
    qaLoading: boolean;
    setQaLoading: React.Dispatch<React.SetStateAction<boolean>>;
    refreshSummary: boolean;
    setRefreshSummary: React.Dispatch<React.SetStateAction<boolean>>;
}

const boxSx = {
    width: "100%",
    margin: "0 auto",
}

const boxTabsSx = {
    width: '100%',
    bgcolor: 'background.paper',
    marginBottom: 0,
    marginTop: 1
}

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
        videoInfoLoading, setVideoInfoLoading,
        summaryLoading, setSummaryLoading,
        agendaLoading, setAgendaLoading,
        topicLoading, setTopicLoading,
        qaLoading, setQaLoading,
        refreshSummary, setRefreshSummary,
    } = props;

    const [ value, setValue ] = useState<number>(1); // 要約タブをデフォにする
    const [ videoInfo, setVideoInfo ] = useState<VideoInfoType|null>(null);
    const [ summary, setSummary ] = useState<SummaryResponseBody|null>(null);
    const [ transcripts, setTranscripts ] = useState<TranscriptType[]|null>(null);
    const [ qaQuestion, setQaQuestion ] = useState<string|null>(null);
    const [ qaAnswer, setQaAnswer ] = useState<QaResponseBody|null>(null);
    const [ qaAlignment, setQaAlignment ] = useState<string>('qa');
    const [ summaryAlignment, setSummaryAlignment ] = useState<string>('summary');

    const clearResult = (refresh: boolean = false) => {
        setValue(1);
        setSummary(null);
        setSummaryAlignment('summary');
        if (refresh === true) {
            return;
        }
        setVideoInfo(null);
        setQaQuestion(null);
        setQaAnswer(null);
        setTranscripts(null);
        setQaAlignment('qa');
    }

    useEffect(() => {
        clearResult();
    }, [vid])

    useEffect(() => {
        if (refreshSummary) {
            clearResult(true);
        }
    }, [refreshSummary])

    const onTabChangeHandler = (_: React.SyntheticEvent, value: number) => {
        setValue(value);
    }

    const tabItemList: string[] = [
        '概要','要約', 'QA/検索', '字幕'
    ]

    return (
        <Box sx={boxSx} id="result-box-01" >
            <Box sx={boxTabsSx} id="result-box-02" >
                <Tabs
                    value={value}
                    onChange={onTabChangeHandler}
                    centered
                >
                    {tabItemList.map((item, idx) =>
                        <Tab key={idx} label={item} />
                    )}
                </Tabs>
            </Box>
            <TabPanel value={value} index={0}>
                <VideoInfo
                    vid={vid}
                    videoInfo={videoInfo}
                    setVideoInfo={setVideoInfo}
                    videoInfoLoading={videoInfoLoading}
                    setVideoInfoLoading={setVideoInfoLoading}
                />
            </TabPanel>
            <TabPanel value={value} index={1}>
                <Summary
                    vid={vid}
                    ytplayer={ytplayer}
                    summary={summary}
                    setSummary={setSummary}
                    alignment={summaryAlignment}
                    setAlignment={setSummaryAlignment}
                    summaryLoading={summaryLoading}
                    setSummaryLoading={setSummaryLoading}
                    agendaLoading={agendaLoading}
                    setAgendaLoading={setAgendaLoading}
                    topicLoading={topicLoading}
                    setTopicLoading={setTopicLoading}
                    refreshSummary={refreshSummary}
                    setRefreshSummary={setRefreshSummary}
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
                    vid={vid}
                    ytplayer={ytplayer}
                    transcripts={transcripts}
                    setTranscripts={setTranscripts}
                />
            </TabPanel>
        </Box>
    ) 
}