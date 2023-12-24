import { Box, List, ListItem, Divider } from '@mui/material';

import { SummaryType } from "./types";

interface TopicProps {
    summary: SummaryType;
}

const boxSx = {
    width: "100%",
    margin: "0 auto",
}

const listBoxSx = {
    width: "80%",
    margin: "0 auto",
    border: "1px solid",
    borderColor: "darkgrey",
}

const listSx = {
    margin: "1em",
    marginTop: "0",
}

const listItemTitleSx = {
    fontWeight: "bold",
    textDecoration: "underline",
    textDecorationThickness: "10%",
}

const listItemAbstractSx = {
    paddingLeft: "3em",
}

const dividerSx = {
    marginTop: "1em",
}

export function Topic (props: TopicProps) {
    const { summary } = props;
    return (
        <Box sx={boxSx} id="topic-box-01" >
            <Box sx={listBoxSx}>
                <List id="topic-list-01" sx={listSx} disablePadding >
                    {summary.topic.map((topic, idx) =>
                    {
                        return (
                            <Box>
                                <ListItem key={idx} sx={listItemTitleSx}>{topic.title}</ListItem>
                                <List disablePadding>
                                    {topic.abstract.map((abstract, idx) =>
                                        <ListItem key={idx} sx={listItemAbstractSx} disablePadding >{abstract}</ListItem>
                                    )}
                                </List>
                                { idx < summary.topic.length -1 && <Divider sx={dividerSx} /> }
                            </Box>
                        )
                    })}
                </List>
            </Box>
        </Box>
    )
}