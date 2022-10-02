import {Paper,Typography, Button} from '@mui/material';



export const PopularCondition = ({open, onClose}) => {
    if (!open) return null;
    return (
        <Paper sx={{position: 'absolute', top: '10%', left: '0%',  bgcolor: 'background.paper', width: '100%', height:'100%', zIndex:'2'}}>
            <div style={{textAlign:'right', paddingRight:'10px', paddingTop:'10px'}}>
            <Button size="small" variant="contained" onClick={onClose} color='error'>X</Button>
            </div>
            
            <Typography variant="h5" component="div">Popular Condition</Typography>
            <Typography variant="h7" color="text.secondary" style={{textAlign:"left"}}>
                Below is a list of popular condition for this application:
                <ol>
                    <li>Icons To use the font Icon component or the prebuilt SVG Material Icons (such as those found in the icon demos ), you must first install the Material Icons font. You can do so with npm or yarn, …</li>
                    <li>Icons To use the font Icon component or the prebuilt SVG Material Icons (such as those found in the icon demos ), you must first install the Material Icons font. You can do so with npm or yarn, …</li>
                    <li>Icons To use the font Icon component or the prebuilt SVG Material Icons (such as those found in the icon demos ), you must first install the Material Icons font. You can do so with npm or yarn, …</li>
                </ol>
            </Typography>
            
        </Paper>
    )
}