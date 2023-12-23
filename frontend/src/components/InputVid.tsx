import { useState, useEffect } from 'react';
import { Box, TextField, MenuItem } from '@mui/material'

interface InputVidProps {
    vid: string;
    setVid: React.Dispatch<React.SetStateAction<string>>;
}

const vidInputBoxSx = {
    padding: "2em",
}

const textFieldSx = {
    marginLeft: "15px",
    marginRight: "15px",
}

export function InputVid (props: InputVidProps) {
    const { vid, setVid } = props;
    const [ sampleVidList, setSampleVidList ] = useState<string[]|null>(null);

    useEffect(() => {
        fetch ("/sample")
        .then((res => res.json()))
        .then((res => setSampleVidList(res.vid)))
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
        <Box sx={vidInputBoxSx}>
            <TextField
                label="Video ID"
                defaultValue={vid}
                value={vid}
                onKeyDown={onKeyDownHandlerVid}
                size="small"
                sx={textFieldSx}
            />
            { sampleVidList &&
                <TextField
                    select
                    label="Sample Video ID"
                    defaultValue={sampleVidList[0]}
                    onChange={onChangeHandlerSelect}
                    size="small"
                    sx={textFieldSx}
                >
                    {sampleVidList.map((vid, index) => {
                        return <MenuItem value={vid} key={index}>{vid}</MenuItem>
                    })}
                </TextField>
}
        </Box>
    )
}