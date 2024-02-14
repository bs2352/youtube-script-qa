import { Box, Table, TableBody, TableRow, TableCell, Link, styled } from '@mui/material';

import { Loading } from '../Loading';
import { VideoInfoType } from "../../common/types"


function s2hms (seconds: number) {
    const h = Math.floor(seconds / 3600).toString().padStart(2, '0')
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0')
    const s = Math.floor(seconds % 60).toString().padStart(2, '0')
    return `${h}:${m}:${s}`
}


interface VideInfoProps {
    videoInfo: VideoInfoType | null;
    videoInfoLoading: boolean;
}

const VideoInfoContainer = styled(Box)({
    width: "100%",
    margin: "0 auto",
});

const VideoInfoTable = styled(Table)({
    width: "85%",
    margin: "0 auto"
});

const VideoInfoTableRow = styled(TableRow)({
    textAlign: "left",
});

const TitleTableCell = styled(TableCell)({
    // whiteSpace: "nowrap",
    fontWeight: "bold",
    backgroundColor: "lightgrey",
    border: "1px solid",
    borderColor: "darkgrey",
    padding: "1.0em",
});

const ValueTableCell = styled(TableCell)({
    border: "1px solid",
    borderColor: "darkgrey",
    padding: "1.0em",
});


export function VideoInfo (props: VideInfoProps) {
    const { videoInfo, videoInfoLoading } = props;

    if (videoInfoLoading) {
        return <Loading />
    }
    if (!videoInfo) {
        return <></>
    }

    const TableElements = [
        { title: "タイトル", value: videoInfo.title },
        { title: "チャンネル名", value: videoInfo.author },
        { title: "時間", value: s2hms(videoInfo.lengthSeconds) },
        { title: "URL", value: <Link href={videoInfo.url} target={`_blank`} >{videoInfo.url}</Link> },
        { title: "Video ID", value: videoInfo.vid },
    ]

    return (
        <VideoInfoContainer id="videoinfo-box-01">
            <VideoInfoTable id="videoinfo-table-01">
                <TableBody>
                    {
                        TableElements.map((item, idx) => {
                            return (
                                <VideoInfoTableRow key={`key-tr-${idx}`}>
                                    <TitleTableCell>{item.title}</TitleTableCell>
                                    <ValueTableCell>{item.value}</ValueTableCell>
                                </VideoInfoTableRow>
                            )
                        })
                    }
                </TableBody>
           </VideoInfoTable>
		</VideoInfoContainer>
	)
}