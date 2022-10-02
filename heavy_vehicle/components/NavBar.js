import * as React from 'react';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import CommuteIcon from '@mui/icons-material/Commute';
import { PopularCondition } from './PopularCondition';
import { EmailPage } from './EmailPage';
import { RegisterTable } from './RegisterTable';

export const NavBar = () => {
    const [showPopularCondition, setShowPopularCondition] = React.useState(false);
    const [showEmailPage, setShowEmailPage] = React.useState(false);
    const [showRegisterTable, setShowRegisterTable] = React.useState(false);

    const handlePopularCondition = () => {
        setShowPopularCondition(!showPopularCondition);
        setShowEmailPage(false);
        setShowRegisterTable(false);
    }

    const handleEmail = () => {
        setShowPopularCondition(false);
        setShowEmailPage(!showEmailPage);
        setShowRegisterTable(false);
    }

    const handleRegister = () => {
        setShowPopularCondition(false);
        setShowEmailPage(false);
        setShowRegisterTable(!showRegisterTable);

    }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="absolute">
        <Toolbar>
        <CommuteIcon sx={{ display: { xs: 'none', md: 'flex' }, mr: 1 }} />
        <Typography
            variant="h6"
            noWrap
            component="a"
            href="https://www.nhvr.gov.au/road-access/mass-dimension-and-loading"
            sx={{
              mr: 2,
              display: { xs: 'none', md: 'flex' },
              fontFamily: 'monospace',
              fontWeight: 700,
              letterSpacing: '.3rem',
              color: 'inherit',
              textDecoration: 'none',
            }}
          >
            NHVR
          </Typography>  
        
          <Button color="inherit" onClick={handleEmail}>Email</Button>
          <Button color="inherit" onClick={handleRegister}>Register</Button>
          <Button color="inherit" onClick={handlePopularCondition}>Popular Conditions</Button>
        </Toolbar>
       
      </AppBar>
      <PopularCondition open= {showPopularCondition} onClose={()=>setShowPopularCondition(false)}/>
      <EmailPage open={showEmailPage} onClose={()=>setShowEmailPage(false)}/>
      <RegisterTable open={showRegisterTable} onClose={()=>setShowRegisterTable(false)}/>
      
    </Box>
  );
}