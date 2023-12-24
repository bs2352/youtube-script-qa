import { Box } from '@mui/material'

const boxSx = {
    width: "100%",
    margin: "0 auto",
}

export function Header () {
    return (
        <Box sx={boxSx} id="header-box-01">
            <h1>Youtube Supporter</h1>
        </Box>
    )
}