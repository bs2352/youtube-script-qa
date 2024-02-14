import { Box, styled } from '@mui/material'

const HeaderContainer = styled(Box)({
    width: "100%",
    margin: "0 auto",
});

export function Header () {
    return (
        <HeaderContainer id="header-box-01">
            <h1>Youtube Supporter</h1>
        </HeaderContainer>
    )
}