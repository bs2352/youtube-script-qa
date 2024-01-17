import { Box, Table, TableBody, TableRow, TableCell, Link } from '@mui/material';

import { Loading } from './Loading';
import { VideoInfoType } from "./types"


function s2hms (seconds: number) {
    const h = Math.floor(seconds / 3600).toString().padStart(2, '0')
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0')
    const s = Math.floor(seconds % 60).toString().padStart(2, '0')
    return `${h}:${m}:${s}`
}


interface VideInfoProps {
    videoInfo: VideoInfoType | null;
    videoInfoLoading: boolean
}

const boxSx = {
    width: "100%",
    margin: "0 auto",
}

const tableSx = {
    width: "85%",
    margin: "0 auto"
}

const tableRowSx = {
    textAlign: "left",
}

const tableCellSx = {
    border: "1px solid",
    borderColor: "darkgrey",
    padding: "1.0em",
}

const tableCellTitleSx = {
    // whiteSpace: "nowrap",
    fontWeight: "bold",
    backgroundColor: "lightgrey",
    ...tableCellSx
}


export function VideoInfo (props: VideInfoProps) {
    const { videoInfo, videoInfoLoading } = props;
    if (videoInfoLoading) {
        return <Loading />
    }
    if (!videoInfo) {
        return <></>
    }
    return (
        <Box sx={boxSx} id="videoinfo-box-01">
            <Table sx={tableSx} id="videoinfo-table-01">
                <TableBody>
                    <TableRow sx={tableRowSx}>
                        <TableCell sx={tableCellTitleSx}>タイトル</TableCell>
                        <TableCell sx={tableCellSx}>{videoInfo.title}</TableCell>
                    </TableRow>
                    <TableRow  sx={tableRowSx}>
                        <TableCell sx={tableCellTitleSx}>チャンネル名</TableCell>
                        <TableCell sx={tableCellSx}>{videoInfo.author}</TableCell>
                    </TableRow>
                    <TableRow  sx={tableRowSx}>
                        <TableCell sx={tableCellTitleSx}>時間</TableCell>
                        <TableCell sx={tableCellSx}>{s2hms(videoInfo.lengthSeconds)}</TableCell>
                    </TableRow>
                    <TableRow  sx={tableRowSx}>
                        <TableCell sx={tableCellTitleSx}>URL</TableCell>
                        <TableCell sx={tableCellSx}>
                            <Link href={videoInfo.url} target={`_blank`} >{videoInfo.url}</Link>
                        </TableCell>
                    </TableRow>
                    <TableRow  sx={tableRowSx}>
                        <TableCell sx={tableCellTitleSx}>Video ID</TableCell>
                        <TableCell sx={tableCellSx}>{videoInfo.vid}</TableCell>
                    </TableRow>
               </TableBody>
            </Table>
		</Box>
	)
}