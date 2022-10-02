
import { Card, CardContent, CardActions, Typography, Table, TableHead, TableCell, TableRow, TableBody, Box, Button } from '@mui/material'
import { useState,useEffect } from 'react'

export const ConditionTable = ({open, onClose, conditionData, handleExport}) => {
    useEffect(() => {
        const initialList = conditionData;
        setConditionList(initialList);
    }, [conditionData])

    const [conditionList, setConditionList] = useState([{}]);
    const handleDelete = (index) => {
        const newData = conditionList.filter((_, i) => i !== index);
        setConditionList(newData);

    }

    if (!open) return null;
    return (
        <Box sx={{position: 'absolute', top: '30%', left: '50%'-300, width: '900px'}}>
            <Card>
                <CardContent>
                <Typography variant="h5" component="div">General Condition</Typography>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell sx={{width:'10%'}}>District</TableCell>
                            <TableCell sx={{width:'30%'}}>Description</TableCell>
                            <TableCell sx={{width:'20%'}}>Criteria</TableCell>
                            <TableCell sx={{width:'30%'}}>Condition</TableCell>
                            <TableCell sx={{width:'10%'}}>Action</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {conditionList.map((row,index) => (
                            <TableRow key={index}>
                                <TableCell sx={{width:'10%'}}>{row.district}</TableCell>
                                <TableCell sx={{width:'30%'}}>{row.description}</TableCell>
                                <TableCell sx={{width:'20%'}}>{row.criteria}</TableCell>
                                <TableCell sx={{width:'30%'}}>{row.condition}</TableCell>
                                <TableCell sx={{width:'10%'}}>
                                <Button variant='contained' size='small' color='warning' onClick={() => handleDelete(index)}>Delete</Button>
                                </TableCell>
                            </TableRow>

                        ))}
                    </TableBody>
                </Table>
                </CardContent>
                <CardActions>
                    <Button size="small" variant="contained" onClick={onClose}>Close</Button>
                    <Button size="small" variant='contained' onClick={() =>handleExport(conditionList)}>Export to PDF</Button>
                </CardActions>
            </Card>
        </Box>
      
        
      
    )


}
