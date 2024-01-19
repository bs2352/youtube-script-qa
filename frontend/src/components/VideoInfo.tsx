import { useEffect, useState } from 'react';
import { Box, Table, TableBody, TableRow, TableCell, Link } from '@mui/material';

import { Loading } from './Loading';
import { VideoInfoType, SummaryRequestBody } from "./types"


function s2hms (seconds: number) {
    const h = Math.floor(seconds / 3600).toString().padStart(2, '0')
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0')
    const s = Math.floor(seconds % 60).toString().padStart(2, '0')
    return `${h}:${m}:${s}`
}


interface VideInfoProps {
    vid: string;
    videoInfo: VideoInfoType | null;
    setVideoInfo: React.Dispatch<React.SetStateAction<VideoInfoType | null>>;
    videoInfoLoading: boolean;
    setVideoInfoLoading: React.Dispatch<React.SetStateAction<boolean>>;
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
    const { vid, videoInfo, setVideoInfo, videoInfoLoading, setVideoInfoLoading } = props;
    const [ curVid, setCurVid ] = useState<string>(vid);

    useEffect(() => {
        if ((vid === "" || vid === curVid) && videoInfo) {
            return;
        }
        setCurVid(vid);
        setVideoInfo(null);
        setVideoInfoLoading(true);
        const requestBody: SummaryRequestBody = {
            vid: vid
        }
        fetch(
            '/info',
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
            setVideoInfo(res);
            setVideoInfoLoading(false);
        }))
        .catch((err) => {
            const errmessage: string = `動画情報の取得中にエラーが発生しました。${err}`;
            console.error(errmessage);
            alert(errmessage);
            setVideoInfoLoading(false);
        })
    }, [vid]);

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