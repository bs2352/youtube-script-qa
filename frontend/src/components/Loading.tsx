// import { styled } from '@mui/material/styles'
import { CircularProgress, Box } from '@mui/material';

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

const circularProgressSx = {
    margin: "30px",
};

export function Loading() {
    return (
        // <SpinCircle />
        <Box sx={{ color: 'grey.500' }} >
            <CircularProgress sx={circularProgressSx} color="inherit" />
        </Box>
    )
}