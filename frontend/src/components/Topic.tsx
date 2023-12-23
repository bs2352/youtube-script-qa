import { Box, List, ListItem, Divider } from '@mui/material';

import { SummaryType } from "./types";

interface TopicProps {
    summary: SummaryType;
}

const boxSx = {
    width: "auto",
    margin: "0 auto",
    padding: "2em",
    paddingTop: "0.0em",
    paddingBottom: "0.0em",
    border: "1px solid",
    borderColor: "darkgrey",
}

const listSx = {
    // margin: "1em",
    // marginTop: "0"
}

const listItemTitleSx = {
    fontWeight: "bold",
    fontSize: "110%",
    textDecoration: "underline",
    textDecorationThickness: "10%",
    // paddingBottom: "0.2em",
}

const listItemAbstractSx = {
    paddingLeft: "3em",
}

const dividerSx = {
    marginTop: "0.5em",
    // marginBottom: "0.5em",
}

export function Topic (props: TopicProps) {
    const { summary } = props;
    return (
        <Box sx={boxSx}>
            <List sx={listSx}>
                {summary.topic.map((topic, idx) =>
                {
                    return (
                        <>
                            <ListItem key={idx} sx={listItemTitleSx}>{topic.title}</ListItem>
                            <List>
                                {topic.abstract.map((abstract, idx) =>
                                    <ListItem key={idx} sx={listItemAbstractSx} disablePadding >{abstract}</ListItem>
                                )}
                            </List>
                            { idx < summary.topic.length -1 && <Divider sx={dividerSx} /> }
                        </>
                    )
                })}
            </List>
        </Box>
    )
}