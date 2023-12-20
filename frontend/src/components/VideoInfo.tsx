import Box from '@mui/material/Box'
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableRow from '@mui/material/TableRow';
import TableCell from '@mui/material/TableCell';

import { SummaryType } from "./types"


function s2hms (seconds: number) {
    const h = Math.floor(seconds / 3600).toString().padStart(2, '0')
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0')
    const s = Math.floor(seconds % 60).toString().padStart(2, '0')
    return `${h}:${m}:${s}`
}


interface VideInfoProps {
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

const tableCellTitleSx = {
    whiteSpace: "nowrap",
    fontWeight: "bold",
    // backgroundColor: "bisque",
    backgroundColor: "lightgrey",
    ...tableCellSx
}


export function VideoInfo (props: VideInfoProps) {
    const { summary } = props;
    return (
        <Box sx={boxSx}>
            <Table sx={tableSx}>
                <TableBody>
                    <TableRow sx={tableRowSx}>
                        <TableCell sx={tableCellTitleSx}>タイトル</TableCell>
                        <TableCell sx={tableCellSx}>{summary.title}</TableCell>
                    </TableRow>
                    <TableRow  sx={tableRowSx}>
                        <TableCell sx={tableCellTitleSx}>チャンネル名</TableCell>
                        <TableCell sx={tableCellSx}>{summary.author}</TableCell>
                    </TableRow>
                    <TableRow  sx={tableRowSx}>
                        <TableCell sx={tableCellTitleSx}>時間</TableCell>
                        <TableCell sx={tableCellSx}>{s2hms(summary.lengthSeconds)}</TableCell>
                    </TableRow>
                    <TableRow  sx={tableRowSx}>
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
	)
}