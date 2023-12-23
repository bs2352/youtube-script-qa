import { Box, List, ListItem, Divider } from '@mui/material';

import { SummaryType } from "./types"


interface DetailSummaryProps {
    summary: SummaryType;
}

const boxSx = {
    width: "100%",
    margin: "0 auto",
}

const boxListSx = {
    width: "80%",
    margin: "0 auto",
    border: "1px solid",
    borderColor: "darkgrey",
    padding: "1.0em",
    paddingTop: "0.5em",
    paddingBottom: "0.5em",
}

const listSx = {
    marginTop: "0",
}

const listItemSx = {
    margin: "100",
    // padding: "100",
}

const dividerSx = {
    marginTop: "0.5em",
    marginBottom: "0.5em",
}

export function DetailSummary (props: DetailSummaryProps) {
    const { summary } = props;
    return (
        <Box sx={boxSx}>
            <Box sx={boxListSx} >
                <List sx={listSx} disablePadding >
                    {summary.detail.map((detail, idx) =>
                    {
                        return (
                            <>
                                <ListItem key={idx} sx={listItemSx}>{detail}</ListItem>
                                { (idx < summary.detail.length - 1) && <Divider sx={dividerSx} /> }
                            </>
                        )
                    })}
                </List>
            </Box>
        </Box>
    )
}