import { useState, useEffect, useRef } from 'react';
import { Box, TextField, MenuItem, IconButton } from '@mui/material'
import { Clear } from '@mui/icons-material'

import { SampleVideoInfo } from './types';


interface InputVidProps {
    vid: string;
    setVid: React.Dispatch<React.SetStateAction<string>>;
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
    marginLeft: "0px",
    marginRight: "10px",
}

export function InputVid (props: InputVidProps) {
    const { vid, setVid } = props;
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