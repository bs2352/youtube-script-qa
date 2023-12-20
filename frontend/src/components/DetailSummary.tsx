import Box from '@mui/material/Box'
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableRow from '@mui/material/TableRow';
import TableCell from '@mui/material/TableCell';

import { SummaryType } from "./types"


interface DetailSummaryProps {
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
    padding: "1.0em",
    paddingRight: "1.0em",
}

export function DetailSummary (props: DetailSummaryProps) {
    const { summary } = props;
    return (
        <Box sx={boxSx}>
            <Table sx={tableSx}>
                <TableBody>
                    {summary.detail.map((detail, idx) =>
                        <TableRow sx={tableRowSx} key={idx}>
                            <TableCell sx={tableCellSx}>
                                {detail}
                            </TableCell>
                        </TableRow>
                    )}
                </TableBody>
            </Table>
        </Box>
    )
}