import { Card, CardContent, CardActions, Typography, Table, TableHead, TableCell, TableRow, TableBody, Box, Button } from '@mui/material'
import { useState,useEffect } from 'react'

export const RegisterTable = ({open, onClose}) => {
    if (!open) return null;
    return (
        <Box sx={{position: 'absolute', top: '10%', left: '0%',  bgcolor: 'background.paper', width: '100%', height:'100%', zIndex:'2'}}>
            <Card>
                <CardActions sx={{
                    display: 'flex',
                    justifyContent: 'flex-end',
                }}>
                    <Button size="small" variant="contained" onClick={onClose} color='error'>Close</Button>
                </CardActions>
                <CardContent>
                <Typography variant="h5" component="div">General Condition</Typography>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell sx={{width:'10%'}}>Case Number</TableCell>
                            <TableCell sx={{width:'10%'}}>Date Due</TableCell>
                            <TableCell sx={{width:'10%'}}>Completed By</TableCell>
                            <TableCell sx={{width:'10%'}}>Responsible</TableCell>
                            <TableCell sx={{width:'10%'}}>Status</TableCell>
                            <TableCell sx={{width:'20%'}}>Comment</TableCell>
                            <TableCell sx={{width:'20%'}}>Action</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                            <TableRow >
                                <TableCell sx={{width:'10%'}}>234324323</TableCell>
                                <TableCell sx={{width:'10%'}}>01/01/2022</TableCell>
                                <TableCell sx={{width:'10%'}}>02/09/2022</TableCell>
                                <TableCell sx={{width:'10%'}}>John Doe</TableCell>
                                <TableCell sx={{width:'10%'}}>Open</TableCell>
                                <TableCell sx={{width:'20%'}}>Lorem Impsum</TableCell>
                                <TableCell sx={{width:'20%'}}>
                                <Button variant='contained' size='small' color='info'>Edit</Button>
                                </TableCell>
                            </TableRow>
                    </TableBody>
                </Table>
                </CardContent>
            </Card>
        </Box>
      
        
      
    )


}