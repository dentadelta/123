import { Paper, Typography, Button, TextField} from '@mui/material';
import { useState } from 'react';
import axios from 'axios';

export const EmailPage = ({open,onClose}) => {
    const [to, setTo] = useState('');
    const [cc, setCc] = useState('');
    const [bcc, setBcc] = useState('');
    const [subject, setSubject] = useState('');
    const [body, setBody] = useState('');

    const handleTo = (event) => {
        setTo(event.target.value);
    }

    const handleCC = (event) => {
        setCc(event.target.value);
    }

    const handleBCC = (event) => {
        setBcc(event.target.value);
    }

    const handleSubject = (event) => {
        setSubject(event.target.value);
    }

    const handleBody = (event) => {
        setBody(event.target.value);
    }

    const handleSend = () => {
        if (to === '' || subject === '' || body === '') {
            alert('Please fill in all required fields (to, subject, body)');

        } else {
        axios.post('http://localhost:8000/sendEmail', {to: to, cc: cc, bcc: bcc, subject: subject, body: body})
        .then(response => {
            console.log(response.data.message);
        })
        .catch(error => {
            console.log(error);
        })
        }
    }

    if (!open) return null;
    return (
        <Paper sx={{position: 'absolute', top: '10%', left: '0%',  bgcolor: 'background.paper', width: '100%', height:'100%', zIndex:'2'}}>
            <div style={{textAlign:'right', paddingRight:'10px', paddingTop:'10px'}}>
            <Button size="small" variant="contained" onClick={onClose} color='error'>X</Button>
            </div>
            <Typography variant='h5'>Email Page</Typography>
            <Typography variant='h7' color='text.secondary' style={{textAlign:'left'}}>
                This is the email page
            </Typography>
            <div style={{textAlign:'left', paddingLeft:'10px', paddingTop:'10px', padding:'10px'}}>
            <TextField fullWidth label="To:" id="fullWidth" size='small' paddingTop='10px' onChange={handleTo} />
            <TextField fullWidth label="CC:" id="fullWidth" size='small' paddingTop='10px' margin="dense" onChange={handleCC}/>
            <TextField fullWidth label="BCC:" id="fullWidth" size='small' paddingTop='10px' margin="dense" onChange={handleBCC}/>
            <TextField fullWidth label="Subject:" id="fullWidth" size='small' paddingTop='10px' margin="dense" onChange={handleSubject}/>
            <textarea rows="20" cols="60" name="text" style={{
                textAlign:'left', 
                textAlignVertical:true,
                paddingTop:'10px',
                minWidth:'100%',
                minHeight:'100%',
                fontStyle:'normal',
                fontSize:'16px',
                fontFamily:'Arial',
            }}
            placeholder='Email Body:' onChange={handleBody}></textarea>
            </div>
            <div className='emailButton' style={{textAlign:'right', paddingRight:'10px', paddingTop:'10px',paddingLeft:'10px',padding:'10px'}}>
            <Button size="small" variant="contained" onClick={onClose} color='error'>Cancel</Button>
            <Button size="small" variant="contained" onClick={handleSend} color='success'>Send</Button>
            </div>
        </Paper>
    )
}