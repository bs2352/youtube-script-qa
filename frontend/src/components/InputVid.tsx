import { useState, useEffect } from 'react';
import { Box, TextField, MenuItem } from '@mui/material'

import { SampleVideoInfo } from './types';


interface InputVidProps {
    vid: string;
    setVid: React.Dispatch<React.SetStateAction<string>>;
}

const boxSx = {
    width: "100%",
    margin: "0 auto",
}

const textFieldSx = {
    margin: "30px",
    marginLeft: "15px",
    marginRight: "15px",
    maxWidth: "150px"
}

export function InputVid (props: InputVidProps) {
    const { vid, setVid } = props;
    const [ sampleVideoList, setSampleVideoList ] = useState<SampleVideoInfo[]|null>(null);

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
    }

    return (
        <Box sx={boxSx} id="inputvid-box-01">
            <TextField
                label="Video ID"
                defaultValue={vid}
                onKeyDown={onKeyDownHandlerVid}
                size="small"
                sx={textFieldSx}
                // InputLabelProps={{shrink: true}}
            />
            { sampleVideoList &&
                <TextField
                    select
                    label="Sample Video"
                    defaultValue={vid}
                    onChange={onChangeHandlerSelect}
                    size="small"
                    sx={textFieldSx}
                    // InputLabelProps={{shrink: true}}
                >
                    {sampleVideoList.map((video, index) => {
                        return (
                            <MenuItem
                                key={`sample-vid-${index}`}
                                value={video.vid}
                            >
                                ({video.title})
                            </MenuItem>
                        )
                    })}
                </TextField>
}
        </Box>
    )
}