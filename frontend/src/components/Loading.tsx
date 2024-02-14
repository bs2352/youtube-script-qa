// import { styled } from '@mui/material/styles'
import { CircularProgress, Box, styled } from '@mui/material';
import { grey } from '@mui/material/colors'

// const SpinCircle = styled(
//     () => {
//         return (
//             <div
//                 style={{
//                     display: 'inline-block', width: '30px', height: '30px',
//                     margin: '30px', verticalAlign: 'middle',
//                     border: '5px solid', borderRadius: '50%', borderColor: 'red blue green orange',
//                     animation: 'spin-circle 1s infinite linear',
//                 }}
//             />
//         );
//     }
// )({
//     '@keyframes spin-circle': {
//         from: {
//             transform: "rotate(0deg);",
//         },
//         to: {
//             transform: 'rotate(360deg);',
//         }
//     }
// });

interface LoadingProps {
    size?: number | string;
    margin?: number | string;
}

const LoadingContainer = styled(Box)({
    color: grey[500],
});

export function Loading(props: LoadingProps) {
    const { size, margin } = props;

    const StyledircularProgress = styled(CircularProgress)({
        margin: (margin === undefined) ? "30px" : margin,
    })

    return (
        // <SpinCircle />
        <LoadingContainer>
            <StyledircularProgress color="inherit" size={size} />
        </LoadingContainer>
    )
}