import Box from '@mui/material/Box'
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableRow from '@mui/material/TableRow';
import TableCell from '@mui/material/TableCell';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';

import { SummaryType } from "./types";

interface TopicProps {
    summary: SummaryType;
}

const boxSx = {
    width: "75%",
    margin: "0 auto",
    padding: "2em",
    paddingTop: "1em",
}

const tableSx = {
    // border: "1px solid",
    // borderColor: "red",
    // borderCollapse: "collapse",
}

const tableRowSx = {
    textAlign: "left",
}

const tableCellSx = {
    border: "1px solid",
    borderColor: "darkgrey",
    padding: "0.8em",
    paddingLeft: "3em",
    paddingRight: "3em",
    paddingTop: "0em",
    whiteSpace: "nowrap",
}

const listItemTitleSx = {
    fontWeight: "bold",
    fontSize: "118%",
    textDecoration: "underline",
    textDecorationThickness: "10%",
    paddingBottom: "0.2em",
}

const listItemAbstractSx = {
    paddingLeft: "3em",
}

export function Topic (props: TopicProps) {
    const { summary } = props;
    return (
        <Box sx={boxSx}>
            <Table sx={tableSx}>
                <TableBody>
                    {summary.topic.map((topic, idx) =>
                        <TableRow sx={tableRowSx} key={idx}>
                            <TableCell sx={tableCellSx}>
                                <List disablePadding>
                                    <ListItem sx={listItemTitleSx}>{topic.title}</ListItem>
                                    <List disablePadding>
                                        {topic.abstract.map((abstract, idx) =>
                                            <ListItem key={idx} sx={listItemAbstractSx} disablePadding>{abstract}</ListItem>
                                        )}
                                    </List>
                                </List>
                            </TableCell>
                        </TableRow>
                    )}
                </TableBody>
            </Table>
        </Box>
    )
}