import { useState, useEffect, useRef } from 'react';
import { Box, TextField, MenuItem, IconButton } from '@mui/material'
import { Clear, Refresh } from '@mui/icons-material'

import { SampleVideoInfo, SummaryRequestBody, SummaryResponseBody } from './types';


interface InputVidProps {
    vid: string;
    setVid: React.Dispatch<React.SetStateAction<string>>;
    setSummary: React.Dispatch<React.SetStateAction<SummaryResponseBody | null>>;
    loading: boolean;
    setLoading: React.Dispatch<React.SetStateAction<boolean>>;
}

const boxSx = {
    width: "100%",
    margin: "0 auto",
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
    marginLeft: "5px",
    marginRight: "0px",
}

const iconButtonRefreshSx = {
    verticalAlign: "bottom",
    margin: "20px",
    marginLeft: "0px",
    marginRight: "10px",
}

export function InputVid (props: InputVidProps) {
    const { vid, setVid, setSummary, loading, setLoading } = props;
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
        setSummary(null);
        setLoading(true);

        if (vid === "") {
            setLoading(false);
            return;
        }

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
            setLoading(false);
        }))
        .catch((err) => {
            const errmessage: string = `要約作成中にエラーが発生しました。${err}`;
            console.error(errmessage);
            alert(errmessage);
            setLoading(false);
        })
    }

    return (
        <Box sx={boxSx} id="inputvid-box-01">
            <TextField
                label="Video ID"
                defaultValue={vid}
                onKeyDown={onKeyDownHandlerVid}
                size="small"
                sx={textFieldVidSx}
                // InputLabelProps={{shrink: true}}
                inputRef={vidRef}
                placeholder="xxxxx"
            />
            <IconButton
                sx={iconButtonClearSx}
                onClick={onClickHandlerClearVid}
                size='small'
            >
                <Clear fontSize='medium' />
            </IconButton>
            <IconButton
                sx={iconButtonRefreshSx}
                onClick={onClickHandlerRefreshSummary}
                size='small'
                disabled={loading}
            >
                <Refresh fontSize='medium' />
            </IconButton>
            { sampleVideoList &&
                <TextField
                    select
                    label="Sample Video"
                    defaultValue={vid}
                    onChange={onChangeHandlerSelect}
                    size="small"
                    sx={textFieldSampleSx}
                    // InputLabelProps={{shrink: true}}
                >
                    {sampleVideoList.map((video, index) => {
                        return (
                            <MenuItem key={`sample-vid-${index}`} value={video.vid} >
                                ({video.title})
                            </MenuItem>
                        )
                    })}
                </TextField>
            }
        </Box>
    )
}