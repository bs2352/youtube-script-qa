import { useState, useEffect, useRef } from 'react';
import { Box, TextField, MenuItem, IconButton, Stack, Tooltip } from '@mui/material'
import { Clear, Refresh } from '@mui/icons-material'

import { SampleVideoInfo, SummaryRequestBody, SummaryResponseBody } from './types';


interface InputVidProps {
    vid: string;
    setVid: React.Dispatch<React.SetStateAction<string>>;
    setSummary: React.Dispatch<React.SetStateAction<SummaryResponseBody | null>>;
    summaryLoading: boolean;
    setSummaryLoading: React.Dispatch<React.SetStateAction<boolean>>;
}

const boxSx = {
    width: "80%",
    margin: "0 auto 10px auto",
}

const titleDivStyle = {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "70px",
    height: "50px",
    margin: "10px 20px 10px 0px",
    borderRadius: "5px",
    backgroundColor: "#FF5252", // material ui colors Red[A200]
    color: "white",
    fontWeight: "bold",
    fontSize: "1.5em",
}

const textFieldVidSx = {
    margin: "20px",
    marginLeft: "15px",
    marginRight: "0px",
    maxWidth: "150px"
}

const textFieldSampleSx = {
    margin: "20px",
    marginLeft: "15px",
    marginRight: "15px",
    maxWidth: "300px",
}

const iconButtonClearSx = {
    verticalAlign: "bottom",
    margin: "20px",
    marginLeft: "10px",
    marginRight: "0px",
}

const iconButtonRefreshSx = {
    verticalAlign: "bottom",
    margin: "20px",
    marginLeft: "0px",
    marginRight: "10px",
}

export function InputVid (props: InputVidProps) {
    const { vid, setVid, setSummary, summaryLoading, setSummaryLoading } = props;
    const [ sampleVideoList, setSampleVideoList ] = useState<SampleVideoInfo[]|null>(null);
    const vidRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        fetch ("/sample")
        .then((res => res.json()))
        .then((res => setSampleVideoList(res.info)))
        .catch((err) => console.log(err))
    }, []);

    const onKeyDownHandlerVid = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter') {
            const inputElement = event.target as HTMLInputElement;
            setVid(inputElement.value);
        }
    }

    const onChangeHandlerSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const inputElement = event.target as HTMLInputElement;
        setVid(inputElement.value);
        if (vidRef && vidRef.current)
            vidRef.current.value = inputElement.value
    }

    const onClickHandlerClearVid = () => {
        // setVid("");
        if (vidRef && vidRef.current) {
            vidRef.current.value = "";
        }
    }

    const onClickHandlerRefreshSummary = () => {
        if (vid === "") {
            return;
        }
        setSummary(null);
        setSummaryLoading(true);
        const requestBody: SummaryRequestBody = {
            vid: vid,
            refresh: true,
        }
        fetch(
            '/summary',
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
            setSummary(res);
            setSummaryLoading(false);
        }))
        .catch((err) => {
            const errmessage: string = `要約作成中にエラーが発生しました。${err}`;
            console.error(errmessage);
            alert(errmessage);
            setSummaryLoading(false);
        })
    }

    const TitleBox = () => {
        return (
            <div>
                <div style={titleDivStyle} >
                    <div>YTS</div>
                </div>
            </div>
        )
    }

    const VidInputBox = () => {
        const ClearButton = (props: {children: JSX.Element}) => {
            return (
                <>
                    {
                        summaryLoading ? props.children
                        :
                        <Tooltip title="入力のクリア" placement="top-end" arrow={true} >
                            {props.children}
                        </Tooltip>
                    }
                </>
            )
        }
        const RefreshButton = (props: {children: JSX.Element}) => {
            return (
                <>
                    {
                        summaryLoading ? props.children
                        :
                        <Tooltip title="要約を再生成します" placement="top-end" arrow={true} >
                            {props.children}
                        </Tooltip>
                    }
                </>
            )
        }
        return (
            <div style={{display: "flex", flexWrap: "nowrap"}}>
                <TextField
                    label="Video ID"
                    defaultValue={vid}
                    onKeyDown={onKeyDownHandlerVid}
                    size="small"
                    sx={textFieldVidSx}
                    inputRef={vidRef}
                    placeholder="xxxxx"
                    disabled={summaryLoading}
                />
                <ClearButton>
                    <IconButton
                        sx={iconButtonClearSx}
                        onClick={onClickHandlerClearVid}
                        size='small'
                        disabled={summaryLoading}
                    >
                        <Clear fontSize='medium' />
                    </IconButton>
                </ClearButton>
                <RefreshButton>
                    <IconButton
                        sx={iconButtonRefreshSx}
                        onClick={onClickHandlerRefreshSummary}
                        size='small'
                        disabled={summaryLoading}
                    >
                        <Refresh fontSize='medium' />
                    </IconButton>
                </RefreshButton>
            </div>
        )
    }

    const SampleVidSelectBox = () => {
        if (sampleVideoList === null) {
            return <></>
        }
        return (
            <div>
                <TextField
                    select
                    label="Sample Video"
                    defaultValue={vid}
                    onChange={onChangeHandlerSelect}
                    size="small"
                    sx={textFieldSampleSx}
                    disabled={summaryLoading}
                >
                    {sampleVideoList.map((video, index) => {
                        return (
                            <MenuItem key={`sample-vid-${index}`} value={video.vid} >
                                ({video.title})
                            </MenuItem>
                        )
                    })}
                </TextField>
            </div>
        )
    }

    const InputBox = () => {
        return (
            <Stack direction="row" sx={{flexWrap: "wrap"}} >
                <VidInputBox />
                { sampleVideoList && <SampleVidSelectBox /> }
            </Stack>
        )
    }

    return (
        <Box sx={boxSx} id="inputvid-box-01">
            <Stack direction="row" >
                <TitleBox />
                <InputBox />
            </Stack>
        </Box>
    )
}