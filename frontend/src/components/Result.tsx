import { Box, Tabs, Tab } from '@mui/material'
import { useState } from 'react'
import { YouTubePlayer } from 'react-youtube'

import { SummaryResponseBody, TranscriptType } from "./types"
import { VideoInfo } from './VideoInfo'
import { DetailSummary } from './DetailSummary'
import { Topic } from './Topic'
import { QA } from './QA'
import { Transcript } from './Transcript'


interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

interface ResultProps {
    summary: SummaryResponseBody;
    vid: string;
    ytplayer: YouTubePlayer
}

const boxSx = {
    width: "100%",
    margin: "0 auto",
}

const boxTabsSx = {
    width: '100%',
    bgcolor: 'background.paper',
    marginBottom: 0.5,
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
    const { summary, vid, ytplayer } = props;

    const [ value, setValue ] = useState<number>(0)
    const [ transcripts, setTranscripts] = useState<TranscriptType[]|null>(null);

    const tabItemList: string[] = ['概要', '詳細', 'トピック', 'QA', '字幕']

    const onTabChangeHandler = (_: React.SyntheticEvent, value: number) => {
        setValue(value);
    }

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
                <VideoInfo summary={summary.summary} />
            </TabPanel>
            <TabPanel value={value} index={1}>
                <DetailSummary summary={summary.summary} />
            </TabPanel>
            <TabPanel value={value} index={2}>
                <Topic summary={summary.summary} />
            </TabPanel>
            <TabPanel value={value} index={3}>
                <QA vid={vid} ytplayer={ytplayer} />
            </TabPanel>
            <TabPanel value={value} index={4}>
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