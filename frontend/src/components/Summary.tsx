import {
    Box, ToggleButton, ToggleButtonGroup,
    Table, TableBody, TableRow, TableCell,
    List, ListItem, Divider,
    Link
} from '@mui/material'
import { useEffect, useState } from 'react'
import { YouTubePlayer } from 'react-youtube'

import { Loading } from './Loading'
import { SummaryType, SummaryResponseBody, SummaryRequestBody, TopicType } from "./types"
import { s2hms, hms2s } from './utils'


interface SummaryProps {
    summary: SummaryResponseBody;
    setSummary: React.Dispatch<React.SetStateAction<SummaryResponseBody | null>>;
    alignment: string;
    setAlignment: React.Dispatch<React.SetStateAction<string>>;
    ytplayer: YouTubePlayer;
}

interface AgendaProps {
    summaryRes: SummaryResponseBody;
    setSummaryRes: React.Dispatch<React.SetStateAction<SummaryResponseBody | null>>;
    ytplayer: YouTubePlayer;
}

interface TopicProps {
    summaryRes: SummaryResponseBody;
    setSummaryRes: React.Dispatch<React.SetStateAction<SummaryResponseBody | null>>;
    ytplayer: YouTubePlayer;
}


const boxSx = {
    width: "100%",
    margin: "0 auto",
}

const boxSummarySx = {
    width: "85%",
    margin: "0 auto",
    marginBottom: "1em",
}

const boxToggleButtonSx = {
    marginBottom: "20px",
    // marginLeft: '4%',
    textAlign: 'left',
}

const boxContentSx = {
    marginBottom: "20px",
    // marginLeft: '4%',
    textAlign: 'left',
}

const tableSx = {
    // width: "90%",
    margin: "0 auto"
}

const tableRowSx = {
    textAlign: "left",
}

const tableCellSx = {
    border: "1px solid",
    borderColor: "darkgrey",
    padding: "1.0em",
    paddingRight: "1.0em",
}

const tableCellTitleSx = {
    whiteSpace: "nowrap",
    fontWeight: "bold",
    backgroundColor: "lightgrey",
    ...tableCellSx
}

const detailBoxSx = {
    margin: "0 auto",
    border: "1px solid",
    borderColor: "darkgrey",
    padding: "1.0em",
    paddingTop: "0.5em",
    paddingBottom: "0.5em",
    // height: "400px",
    // overflowY: "scroll",
}

const listSx = {
    marginTop: "0",
}

const listItemSx = {
    margin: "100",
    // padding: "100",
}

const dividerSx = {
    marginTop: "0em",
    marginBottom: "0em",
}

const listItemTitleSx = {
    fontWeight: "bold",
    textDecoration: "underline",
    textDecorationThickness: "10%",
}

const listItemAbstractSx = {
    paddingLeft: "3em",
}

const agendaDividerSx = {
    marginTop: "1em",
}


function Concise (props: {summary: SummaryType}) {
    const { summary } = props;

    return (
        <Box id="concide-box">
            <Table sx={tableSx} id="concise-table-01">
                <TableBody>
                    <TableRow sx={tableRowSx}>
                        <TableCell sx={tableCellTitleSx}>要約</TableCell>
                        <TableCell sx={tableCellSx}>{summary.concise}</TableCell>
                    </TableRow>
                    <TableRow  sx={tableRowSx}>
                        <TableCell sx={tableCellTitleSx}>キーワード</TableCell>
                        <TableCell sx={tableCellSx}>{summary.keyword.join(', ')}</TableCell>
                    </TableRow>
                </TableBody>
            </Table>
        </Box>
    );  

}


function Detail (props: {summary: SummaryType, ytplayer: YouTubePlayer}) {
    const { summary, ytplayer } = props;

    const onClickHandlerDetailSummary = (start: number) => {
        ytplayer.seekTo(start, true);
    }

    return (
        <Box sx={detailBoxSx} id="detail-box" >
            <List sx={listSx} disablePadding >
                {summary.detail.map((detail, idx) =>
                    {
                        return (
                            <Box key={`detail-${idx}`}>
                                <ListItem key={`item-detail-summary-${idx}`} sx={listItemSx}>
                                    <span>
                                        <span style={{fontWeight: "bold"}}>{`[${idx+1}/${summary.detail.length}] `}</span>
                                        {`(`}
                                        <Link
                                            href="#"
                                            onClick={()=>onClickHandlerDetailSummary(Math.round(detail.start))}
                                            underline="always"
                                        >
                                            {`${s2hms(detail.start)}`}
                                        </Link>
                                        {`) `}
                                        {detail.text}
                                    </span>
                                </ListItem>
                                { (idx < summary.detail.length - 1) && <Divider sx={dividerSx} /> }
                            </Box>
                        )
                    })
                }
            </List>
        </Box>
    );  

}


function Agenda (props: AgendaProps) {
    const { summaryRes, setSummaryRes, ytplayer } = props;
    const [ loading, setLoading] = useState<boolean>(false);
    const summary: SummaryType = summaryRes.summary

    useEffect(() => {
        if (summary.agenda.length > 0 && summary.agenda[0].time.length > 0) {
            return; // 既にタイムテーブルを持っている場合は何もしない。
        }
        setLoading(true);
        const requestBody: SummaryRequestBody = {
            vid: summaryRes.vid
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
            setSummaryRes(res);
            setLoading(false);
        }))
        .catch((err) => {
            const errmessage: string = `目次のタイムテーブル作成中にエラーが発生しました。${err}`;
            console.error(errmessage);
            alert(errmessage);
            setLoading(false);
        })
    }, []);

    const onClickHandlerTimeLink = (start_str: string) => {
        if (start_str.slice(-1) === "*") {
            start_str = start_str.slice(0, -1);
        }
        const start: number = hms2s(start_str);
        ytplayer.seekTo(start, true);
    }

    const TimeLink = (props: {time: string[]}) => {
        const { time } = props;
        return (
            <span style={{marginLeft: 10}}>
                {`(`}
                {time.map((t, tidx) => {
                    return (
                        <span key={`time-${tidx}`}>
                            <Link
                                href="#"
                                underline="hover"
                                onClick={()=>onClickHandlerTimeLink(t)}
                            >
                                {t}
                            </Link>
                            {tidx < time.length - 1 && ", "}
                        </span>
                    )
                })}
                {`)`}
            </span>
        )
    }

    return (
        <Box >
            <Box sx={detailBoxSx} id="agenda-box">
                {loading && <Loading size={30} margin={'5px'} />}
                <List id="agenda-list-01" sx={listSx} disablePadding >
                    {summary.agenda.map((agenda, aidx) => {
                        return (
                            <Box key={`agenda-${aidx}`}>
                                <ListItem sx={listItemTitleSx}>
                                    {agenda.title}
                                    {
                                        agenda.time.length > 0 && agenda.time[0].length > 0 &&
                                        <TimeLink time={agenda.time[0]} />
                                    }
                                </ListItem>
                                <List disablePadding >
                                    {agenda.subtitle.map((subtitle, sidx) =>
                                        <ListItem
                                            key={`agenda-subtitle-${sidx}`}
                                            sx={listItemAbstractSx}
                                            disablePadding
                                        >
                                            {subtitle}
                                            {
                                                agenda.time.length > sidx + 1 && agenda.time[sidx+1].length > 0 &&
                                                <TimeLink time={agenda.time[sidx+1]} />
                                            }
                                        </ListItem>
                                    )}
                                </List>
                                { aidx < summary.agenda.length -1 && <Divider sx={agendaDividerSx} /> }
                            </Box>
                        )
                    })}
                </List>
            </Box>
        </Box>
    );  

}


function Topic (props: TopicProps) {
    const { summaryRes, setSummaryRes, ytplayer } = props;
    const [ loading, setLoading] = useState<boolean>(false);
    const summaryTopic: TopicType[] = summaryRes.summary.topic;

    useEffect(() => {
        if (summaryTopic.length > 0 && summaryTopic[0].time.length > 0) {
            return; // 既にタイムテーブルを持っている場合は何もしない。
        }
        setLoading(true);
        const requestBody: SummaryRequestBody = {
            vid: summaryRes.vid
        }
        fetch(
            '/topic',
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
            setSummaryRes(res);
            setLoading(false);
        }))
        .catch((err) => {
            const errmessage: string = `トピックのタイムテーブル作成中にエラーが発生しました。${err}`;
            console.error(errmessage);
            alert(errmessage);
            setLoading(false);
        })
    }, []);

    const onClickHandlerTimeLink = (start_str: string) => {
        if (start_str.slice(-1) === "*") {
            start_str = start_str.slice(0, -1);
        }
        const start: number = hms2s(start_str);
        ytplayer.seekTo(start, true);
    }

    const TimeLink = (props: {time: string[]}) => {
        const { time } = props;
        return (
            <span style={{marginLeft: 10}}>
                {`(`}
                {time.map((t, tidx) => {
                    return (
                        <span key={`time-${tidx}`}>
                            <Link
                                href="#"
                                underline="hover"
                                onClick={()=>onClickHandlerTimeLink(t)}
                            >
                                {t}
                            </Link>
                            {tidx < time.length - 1 && ", "}
                        </span>
                    )
                })}
                {`)`}
            </span>
        )
    }

    return (
        <Box >
            <Box sx={detailBoxSx} id="topic-box">
                {loading && <Loading size={30} margin={'5px'} />}
                <ul id="topic-ul" style={{...listSx, marginBottom: 0}} >
                    {summaryTopic.map((topic, tidx) =>{
                        return (
                            <li key={`topic-${tidx}`} style={listItemSx} >
                                {topic.topic}
                                {
                                    topic.time.length > 0 &&
                                    <TimeLink time={topic.time} />
                                }
                            </li>
                        )
                    })}
                </ul>
            </Box>
        </Box>
    )
}


export function Summary (props: SummaryProps) {
    const { summary, setSummary, alignment, setAlignment, ytplayer } = props;

    const onChangeHandlerMode = (
        _: React.MouseEvent<HTMLElement, MouseEvent>,
        newAlignment: string|null
    ) => {
        setAlignment(newAlignment === null? 'summary' : newAlignment);
    }

    return (
        <Box sx={boxSx}>
            <Box sx={boxSummarySx} >
                <Box sx={boxToggleButtonSx} >
                        <ToggleButtonGroup
                            value={alignment}
                            exclusive
                            size='small'
                            onChange={onChangeHandlerMode}
                        >
                            <ToggleButton value="summary">要約</ToggleButton>
                            <ToggleButton value="detail">あらすじ</ToggleButton>
                            <ToggleButton value="agenda">目次</ToggleButton>
                            <ToggleButton value="topic">トピック</ToggleButton>
                        </ToggleButtonGroup>
                </Box>
                <Box sx={boxContentSx} >
                    {alignment === null || alignment === 'summary' &&
                        <Concise summary={summary.summary} />
                    }
                    {alignment === 'detail' && <Detail summary={summary.summary} ytplayer={ytplayer} />}
                    {alignment === 'agenda' &&
                        <Agenda summaryRes={summary} setSummaryRes={setSummary} ytplayer={ytplayer} />
                    }
                    {alignment === 'topic' &&
                        <Topic summaryRes={summary} setSummaryRes={setSummary} ytplayer={ytplayer} />
                    }
                </Box>
            </Box>
        </Box>
    );  
}