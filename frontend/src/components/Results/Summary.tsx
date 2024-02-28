import {
    Box, ToggleButton, ToggleButtonGroup,
    Table, TableBody, TableRow, TableCell,
    List, ListItem, Divider,
    Link,
    styled
} from '@mui/material'
import { YouTubePlayer } from 'react-youtube'

import { Loading } from '../Loading'
import { SummaryType, SummaryResponseBody, AgendaType, TopicType } from "../../common/types"
import { s2hms, hms2s } from '../../common/utils'


interface SummaryProps {
    ytplayer: YouTubePlayer;
    summary: SummaryResponseBody | null;
    alignment: string;
    setAlignment: React.Dispatch<React.SetStateAction<string>>;
    summaryLoading: boolean;
    agendaLoading: boolean;
}

interface AgendaProps {
    summaryAgenda: AgendaType[];
    ytplayer: YouTubePlayer;
    agendaLoading: boolean;
}

interface TopicProps {
    summaryTopic: TopicType[];
    ytplayer: YouTubePlayer;
}


const SummaryContainer = styled(Box)({
    width: "100%",
    margin: "0 auto",
});

const SummaryInternalContainer = styled(Box)({
    width: "85%",
    margin: "0 auto",
    marginBottom: "1em",
});

const ToggleButtonContainer = styled(Box)({
    marginBottom: "10px",
    // marginLeft: '4%',
    textAlign: 'left',
});

const ContentContainer = styled(Box)({
    marginBottom: "20px",
    // marginLeft: '4%',
    textAlign: 'left',
});

const ConciseTable = styled(Table)({
    // width: "90%",
    margin: "0 auto"
});

const ConciseValueTableCell = styled(TableCell)({
    border: "1px solid",
    borderColor: "darkgrey",
    padding: "1.0em",
    paddingRight: "1.0em",
});

const ConciseTitleTableCell = styled(TableCell)({
    whiteSpace: "nowrap",
    fontWeight: "bold",
    backgroundColor: "lightgrey",
    border: "1px solid",
    borderColor: "darkgrey",
    padding: "1.0em",
    paddingRight: "1.0em",
});

const SummaryContentContainer = styled(Box)({
    margin: "0 auto",
    border: "1px solid",
    borderColor: "darkgrey",
    padding: "1.0em",
    paddingTop: "0.5em",
    paddingBottom: "0.5em",
    // height: "400px",
    // overflowY: "scroll",
});

const SummaryList = styled(List)({
    marginTop: "0",
    // paddingTop: "5px"
});

const DetailSummaryListItem = styled(ListItem)({
    margin: "100",
    // padding: "100",
});

const DetailDivider = styled(Divider)({
    marginTop: "0em",
    marginBottom: "0em",
});

const AgendaTitleListItem = styled(ListItem)({
    fontWeight: "bold",
    textDecoration: "underline",
    textDecorationThickness: "10%",
});

const AgendaAbstractListItem = styled(ListItem)({
    paddingLeft: "3em",
});

const AgendaDivider = styled(Divider)({
    marginTop: "1em",
});

const TopicUl = styled('ul')({
    marginTop: "0",
    // paddingTop: "5px",
    marginBottom: 0,
    paddingTop: "5px",
});

const TopicLi = styled('li')({
    margin: "100",
    // padding: "100",
});


function Concise (props: {summary: SummaryType}) {
    const { summary } = props;

    const ConciseElements = [
        { title: "要約", value: summary.concise },
        { title: "キーワード", value: summary.keyword.join(', ') },
    ]

    return (
        <Box id="concide-box">
            <ConciseTable id="concise-table-01">
                <TableBody>
                    {
                        ConciseElements.map((item, idx) => {
                            return (
                                <TableRow key={`concise-tr-${idx}`}>
                                    <ConciseTitleTableCell>{item.title}</ConciseTitleTableCell>
                                    <ConciseValueTableCell>{item.value}</ConciseValueTableCell>
                                </TableRow>
                            )
                        })
                    }
                </TableBody>
            </ConciseTable>
        </Box>
    );  

}


function Detail (props: {summary: SummaryType, ytplayer: YouTubePlayer}) {
    const { summary, ytplayer } = props;

    const onClickHandlerDetailSummary = (start: number) => {
        ytplayer.seekTo(start, true);
    }

    return (
        <SummaryContentContainer id="detail-box" >
            <SummaryList disablePadding >
                {summary.detail.map((detail, idx) =>
                    {
                        return (
                            <Box key={`detail-${idx}`}>
                                <DetailSummaryListItem key={`item-detail-summary-${idx}`}>
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
                                </DetailSummaryListItem>
                                { (idx < summary.detail.length - 1) && <DetailDivider /> }
                            </Box>
                        )
                    })
                }
            </SummaryList>
        </SummaryContentContainer>
    );

}


function Agenda (props: AgendaProps) {
    const { summaryAgenda, ytplayer, agendaLoading } = props;

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
        <SummaryContentContainer id="agenda-box">
            {agendaLoading && <Loading size={30} margin={'5px'} />}
            <SummaryList id="agenda-list-01" disablePadding >
                {summaryAgenda.map((agenda, aidx) => {
                    return (
                        <Box key={`agenda-${aidx}`}>
                            <AgendaTitleListItem>
                                {agenda.title}
                                {
                                    agenda.time.length > 0 && agenda.time[0].length > 0 &&
                                    <TimeLink time={agenda.time[0]} />
                                }
                            </AgendaTitleListItem>
                            <List disablePadding >
                                {agenda.subtitle.map((subtitle, sidx) =>
                                    <AgendaAbstractListItem
                                        key={`agenda-subtitle-${sidx}`}
                                        disablePadding
                                    >
                                        {subtitle}
                                        {
                                            agenda.time.length > sidx + 1 && agenda.time[sidx+1].length > 0 &&
                                            <TimeLink time={agenda.time[sidx+1]} />
                                        }
                                    </AgendaAbstractListItem>
                                )}
                            </List>
                            { aidx < summaryAgenda.length -1 && <AgendaDivider /> }
                        </Box>
                    )
                })}
            </SummaryList>
        </SummaryContentContainer>
    );  

}


function Topic (props: TopicProps) {
    const { summaryTopic, ytplayer } = props;

    const onClickHandlerTimeLink = (start_str: string) => {
        if (start_str.slice(-1) === "*") {
            start_str = start_str.slice(0, -1);
        }
        const start: number = hms2s(start_str);
        ytplayer.seekTo(start, true);
    }

    const TimeLink = (props: {time: string}) => {
        const { time } = props;
        return (
            <span style={{marginLeft: 10}}>
                {'('}
                <Link
                    href="#"
                    underline="hover"
                    onClick={()=>onClickHandlerTimeLink(time)}
                >
                    {time}
                </Link>
                {') '}
            </span>
        )
    }

    return (
        <SummaryContentContainer id="topic-box">
            <TopicUl id="topic-ul">
                {summaryTopic.map((topic, tidx) =>{
                    return (
                        <TopicLi key={`topic-${tidx}`} >
                            <TimeLink time={topic.time} />
                            {topic.topic}
                        </TopicLi>
                    )
                })}
            </TopicUl>
        </SummaryContentContainer>
    )
}


export function Summary (props: SummaryProps) {
    const {
        ytplayer,
        summary,
        alignment, setAlignment,
        summaryLoading,
        agendaLoading,
    } = props;

    const onChangeHandlerToggleMode = (
        _: React.MouseEvent<HTMLElement, MouseEvent>,
        newAlignment: string|null
    ) => {
        setAlignment(newAlignment === null? 'summary' : newAlignment);
    }

    if (summaryLoading) {
        return <Loading />
    }

    if (!summary) {
        return <></>
    }

    return (
        <SummaryContainer>
            <SummaryInternalContainer>
                <ToggleButtonContainer>
                        <ToggleButtonGroup
                            value={alignment}
                            exclusive
                            size='small'
                            onChange={onChangeHandlerToggleMode}
                        >
                            <ToggleButton value="summary">要約</ToggleButton>
                            <ToggleButton value="detail">あらすじ</ToggleButton>
                            <ToggleButton value="agenda">目次</ToggleButton>
                            <ToggleButton value="topic">トピック</ToggleButton>
                        </ToggleButtonGroup>
                </ToggleButtonContainer>
                <ContentContainer>
                    {alignment === null || alignment === 'summary' &&
                        <Concise summary={summary.summary} />
                    }
                    {alignment === 'detail' &&
                        <Detail summary={summary.summary} ytplayer={ytplayer} />
                    }
                    {alignment === 'agenda' &&
                        <Agenda
                            summaryAgenda={summary.summary.agenda} ytplayer={ytplayer} agendaLoading={agendaLoading}
                        />
                    }
                    {alignment === 'topic' &&
                        <Topic
                            summaryTopic={summary.summary.topic} ytplayer={ytplayer}
                        />
                    }
                </ContentContainer>
            </SummaryInternalContainer>
        </SummaryContainer>
    );  
}