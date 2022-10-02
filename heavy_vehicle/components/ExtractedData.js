import { Card, Box, CardContent, Typography, CardActions, Button, CardMedia} from "@mui/material"
import { useState } from "react"
import { ConditionTable } from "./ConditionTable";
import { RouteModal } from "./RouteModal";
import axios from "axios";

export const ExtractedData = ({extractedData}) => {
    const [openRoute, setOpenRoute] = useState(false);
    const [openTable, setOpenTable] = useState(false);
    const [conditionData, setConditionData] = useState([{}]);

    const showRoute = () => {
        setOpenRoute(!openRoute);
           
    }

    const handleDownload = () => {
        console.log("Download");
    }

    const handleConditions = () => {
        setOpenTable(!openTable);
        axios.post('http://localhost:8000/getConditions', {width: 600, height: 400, length:300, routeList: ["Route"]})
        .then(response => {
            setConditionData(response.data.data);
        })
        .catch(error => {
            console.log(error);
        })
    }

    const handleExport = (e) => {
        setOpenTable(!openTable);
        axios.post('http://localhost:8000/export', e)
        .then(response => {
            console.log(response.data.message);
        })
        .catch(error => {
            console.log(error);
        })
        
    }

    if (extractedData === undefined || extractedData === null || Object.keys(extractedData).length === 0) {
        return <div></div>
    }

    return (
        <>
        {!openRoute & !openTable ?
        <Box width='600px'>
            <Card variant="outlined">
                <CardMedia
                    component="img"
                    height="140"
                    image={extractedData.vehicleImageURL}
                    alt="vehicle type"
                    />
                <CardContent>
                    <Typography gutterBottom variant="h5" component='div'>{extractedData.caseNumber}</Typography>
                    <Typography variant="h6" color="text.secondary" style={{textAlign:"left"}}>
                        {extractedData.width};{extractedData.height};{extractedData.length}
                        </Typography>
                    <Typography variant="body2" color="text.secondary" style={{textAlign:"left"}}>
                        Condition access lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
                    </Typography>
                </CardContent>
                <CardActions>
                    <Button size="small" variant="contained" onClick={showRoute}>Route Map</Button>
                    <Button size="small" variant="contained" onClick={handleDownload}>Download Attachments</Button>
                    <Button size="small" variant="contained" onClick={handleConditions}>Get General Condition</Button>
                </CardActions>
            </Card>
            
        </Box>
        : null}
    
        <RouteModal open={openRoute} onClose={() => setOpenRoute(false)} routeList={extractedData.routeList} imageURL={extractedData.imageURL}/>
        <ConditionTable open={openTable} onClose={() => setOpenTable(false)} conditionData={conditionData} handleExport={handleExport}/>
        </>
    )
}

