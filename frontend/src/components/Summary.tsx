import {
    Box, ToggleButton, ToggleButtonGroup,
    Table, TableBody, TableRow, TableCell,
    List, ListItem, Divider,
    Link
} from '@mui/material'
import { YouTubePlayer } from 'react-youtube'

import { SummaryType } from "./types"
import { s2hms } from './utils'


interface SummaryProps {
    summary: SummaryType;
    alignment: string;
    setAlignment: React.Dispatch<React.SetStateAction<string>>;
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


function Agenda (props: {summary: SummaryType}) {
    const { summary } = props;

    return (
        <Box >
            <Box sx={detailBoxSx} id="agenda-box">
                <List id="agenda-list-01" sx={listSx} disablePadding >
                    {summary.agenda.map((agenda, aidx) =>
                    {
                        return (
                            <Box key={`agenda-${aidx}`}>
                                <ListItem sx={listItemTitleSx}>{agenda.title}</ListItem>
                                <List disablePadding >
                                    {agenda.subtitle.map((subtitle, sidx) =>
                                        <ListItem
                                            key={`agenda-subtitle-${sidx}`}
                                            sx={listItemAbstractSx}
                                            disablePadding
                                        >{subtitle}</ListItem>
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


export function Summary (props: SummaryProps) {
    const { summary, alignment, setAlignment, ytplayer } = props;

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
                        </ToggleButtonGroup>
                </Box>
                <Box sx={boxContentSx} >
                    {alignment === null || alignment === 'summary' &&
                        <Concise summary={summary} />
                    }
                    {alignment === 'detail' && <Detail summary={summary} ytplayer={ytplayer} />}
                    {alignment === 'agenda' && <Agenda summary={summary} />}
                </Box>
            </Box>
        </Box>
    );  
}