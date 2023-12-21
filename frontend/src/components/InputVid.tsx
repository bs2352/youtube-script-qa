import { useRef } from 'react';

import { Box } from '@mui/material'

interface InputVidProps {
    vid: string;
    setVid: React.Dispatch<React.SetStateAction<string>>;
}

const vidInputBoxSx = {
    padding: "2em",
}

const inputVidSx = {
    padding: 5,
    paddingLeft: 7,
    margin: '1em'
}

export function InputVid (props: InputVidProps) {
    const { vid, setVid } = props;

    const refInputTextVid = useRef<HTMLInputElement>(null)

    const onKeyDownHandlerVid = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter') {
            setVid(refInputTextVid.current!.value)
        }
    }

    return (
        <Box sx={vidInputBoxSx}>
            Youtube Video ID
            <input
                type="text"
                defaultValue={vid}
                ref={refInputTextVid}
                onKeyDown={onKeyDownHandlerVid}
                style={inputVidSx}
            />
        </Box>
    )
}