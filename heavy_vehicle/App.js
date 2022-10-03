
import './App.css';
import { SearchButton } from './components/SearchButton';
import { useState } from 'react';
import { ExtractedData } from './components/ExtractedData';
import { Typography} from "@mui/material";
import { NavBar } from './components/NavBar';
import { PayPalScriptProvider } from '@paypal/react-paypal-js';
import React from 'react'
import {FastAPI} from './components/FastAPI';

function App() {
  const [search, setSearch] = useState('');
  const [extractedData, setExtractedData] = useState({});
  const [showExtractedData, setShowExtractedData] = useState(true);
  const [buttonClicked, setButtonClicked] = useState(false);
  const {response,loading} = FastAPI('/case', {caseNumber: search}, buttonClicked);
  
  const extractCaseButton = () => {
    setShowExtractedData(false);
    if (response) {
      setButtonClicked(!buttonClicked);
      setExtractedData(response.data);
      setShowExtractedData(true);}
    if (loading) {
      setShowExtractedData(false);
    }
  };

  const handleSearch = event => {
    setSearch(event.target.value);
    setExtractedData({});
  
  };

  const handleFocus = () => {
    setShowExtractedData(false);
    setExtractedData({});
    
  }

  const handleUnfocus = () => {
    setShowExtractedData(true);
  };
  
  const clientID = process.env.REACT_APP_PAYPAL_CLIENT_ID;
  
  return (
    <PayPalScriptProvider options={{ "client-id": clientID }}>
    <div className="App">
      <NavBar /><br /><br /><br />
      <header className="App-header">
        <br/>
      <Typography variant='h4' align='center'>Heavy Vehicle Case Extractor</Typography>
        <br/>
        <SearchButton extractCase={extractCaseButton} handleSearch={handleSearch} handleFocus={handleFocus} handleUnfocus={handleUnfocus}/>
        <br/>
        {showExtractedData ?
        <ExtractedData extractedData={extractedData}/>
        : null}
      
      </header>
      
    </div>
    </PayPalScriptProvider>
  );
}

export default App;
