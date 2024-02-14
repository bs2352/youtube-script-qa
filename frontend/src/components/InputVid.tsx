import { useState, useEffect, useRef } from 'react';
import { Box, TextField, MenuItem, IconButton, Stack, Tooltip, styled } from '@mui/material'
import { Clear, Refresh } from '@mui/icons-material'

import { SampleVideoInfo } from '../common/types';


interface InputVidProps {
    vid: string;
    setVid: React.Dispatch<React.SetStateAction<string>>;
    loading: boolean;
    setRefreshSummary: React.Dispatch<React.SetStateAction<boolean>>;
}

const InputVidContainer = styled(Box)({
    width: "80%",
    margin: "0 auto 10px auto",
});

const TitleDivIcon = styled('div')({
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
});

const VidTextField = styled(TextField)({
    margin: "20px",
    marginLeft: "15px",
    marginRight: "0px",
    maxWidth: "150px"
});

const SampleVideoTextField = styled(TextField)({
    margin: "20px",
    marginLeft: "15px",
    marginRight: "15px",
    maxWidth: "300px",
});

const ClearIconButton = styled(IconButton)({
    verticalAlign: "bottom",
    margin: "20px",
    marginLeft: "10px",
    marginRight: "0px",
});

const RefreshIconButton = styled(IconButton)({
    verticalAlign: "bottom",
    margin: "20px",
    marginLeft: "0px",
    marginRight: "10px",
});

export function InputVid (props: InputVidProps) {
    const { vid, setVid, loading, setRefreshSummary } = props;
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
        if (vidRef && vidRef.current) {
            vidRef.current.value = "";
        }
    }

    const onClickHandlerRefreshSummary = () => {
        if (vid === "") {
            return;
        }
        setRefreshSummary(true);
    }

    const TitleBox = () => {
        return (
            <div>
                <TitleDivIcon>
                    <div>YTS</div>
                </TitleDivIcon>
            </div>
        )
    }

    const VidInputBox = () => {
        const ClearButton = (props: {children: JSX.Element}) => {
            return (
                <>
                    {
                        loading ? props.children
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
                        loading ? props.children
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
                <VidTextField
                    label="Video ID"
                    defaultValue={vid}
                    onKeyDown={onKeyDownHandlerVid}
                    size="small"
                    inputRef={vidRef}
                    placeholder="xxxxx"
                    disabled={loading}
                />
                <ClearButton>
                    <ClearIconButton
                        onClick={onClickHandlerClearVid}
                        size='small'
                        disabled={loading}
                    >
                        <Clear fontSize='medium' />
                    </ClearIconButton>
                </ClearButton>
                <RefreshButton>
                    <RefreshIconButton
                        onClick={onClickHandlerRefreshSummary}
                        size='small'
                        disabled={loading}
                    >
                        <Refresh fontSize='medium' />
                    </RefreshIconButton>
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
                <SampleVideoTextField
                    select
                    label="Sample Video"
                    defaultValue={vid}
                    onChange={onChangeHandlerSelect}
                    size="small"
                    disabled={loading}
                >
                    {sampleVideoList.map((video, index) => {
                        return (
                            <MenuItem key={`sample-vid-${index}`} value={video.vid} >
                                ({video.title})
                            </MenuItem>
                        )
                    })}
                </SampleVideoTextField>
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
        <InputVidContainer id="inputvid-box-01">
            <Stack direction="row" >
                <TitleBox />
                <InputBox />
            </Stack>
        </InputVidContainer>
    )
}