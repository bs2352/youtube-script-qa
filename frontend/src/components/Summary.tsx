import { useState } from 'react';
import {
    Box, ToggleButton, ToggleButtonGroup,
    Table, TableBody, TableRow, TableCell,
    List, ListItem, Divider
} from '@mui/material'

import { SummaryType } from "./types"


interface SummaryProps {
    summary: SummaryType;
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
    height: "400px",
    overflowY: "scroll",
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

const topicDividerSx = {
    marginTop: "1em",
}


function Concise (props: SummaryProps) {
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


function Detail (props: SummaryProps) {
    const { summary } = props;

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
                                        {detail}
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


function Topic (props: SummaryProps) {
    const { summary } = props;

    return (
        <Box >
            <Box sx={detailBoxSx} id="topic-box">
                <List id="topic-list-01" sx={listSx} disablePadding >
                    {summary.topic.map((topic, idx) =>
                    {
                        return (
                            <Box key={`topic-${idx}`}>
                                <ListItem sx={listItemTitleSx}>{topic.title}</ListItem>
                                <List disablePadding >
                                    {topic.abstract.map((abstract, idx) =>
                                        <ListItem
                                            key={`topic-abstract-${idx}`}
                                            sx={listItemAbstractSx}
                                            disablePadding
                                        >{abstract}</ListItem>
                                    )}
                                </List>
                                { idx < summary.topic.length -1 && <Divider sx={topicDividerSx} /> }
                            </Box>
                        )
                    })}
                </List>
            </Box>
        </Box>
    );  

}


export function Summary (props: SummaryProps) {
    const { summary } = props;

    const [ alignment, setAlignment] = useState<string|null>('summary')

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
                            <ToggleButton value="topic">トピック</ToggleButton>
                        </ToggleButtonGroup>
                </Box>
                <Box sx={boxContentSx} >
                    {alignment === null || alignment === 'summary' &&
                        <Concise summary={summary} />
                    }
                    {alignment === 'detail' && <Detail summary={summary} />}
                    {alignment === 'topic' && <Topic summary={summary} />}
                </Box>
            </Box>
        </Box>
    );  
}