
import { Typography, Card, CardContent, Box, CardActions, CardMedia, Button} from "@mui/material";
export const RouteModal = ({open, onClose, routeList, imageURL}) => {
    const numList = routeList.map((num) => <Typography variant="body2" color="text.secondary" style={{textAlign:"left"}}>{num}</Typography>);
    if (!open) return null;
    return (

        <Box sx={{position: 'absolute', top: '30%', left: '50%'-300,  bgcolor: 'background.paper', width: '600px'}}>
            <Card variant="outlined">
                <CardMedia
                    component="img"
                    height="140"
                    image={imageURL}
                    alt="Route Map"
                    />
                <CardContent>
                    <Typography variant="h5" component="div">Route Map</Typography>
                    <Typography variant="body2" color="text.secondary" style={{textAlign:"left"}}>
                        Below is a list of route for this application:
                        {numList}
                    </Typography>
                </CardContent>
                
                <CardActions>
                    <Button size="small" variant="contained" onClick={onClose}>Close</Button>
                </CardActions>
            </Card>
        </Box>

        )
}