
import './App.css';
import { SearchButton } from './components/SearchButton';
import { useState,useEffect } from 'react';
import { ExtractedData } from './components/ExtractedData';
import axios from 'axios';
import { Button, Typography} from "@mui/material";
import { NavBar } from './components/NavBar';
import { ProgressBar } from './components/ProgressBar';

function App() {
  const [search, setSearch] = useState('');
  const [extractedData, setExtractedData] = useState({});
  const [showExtractedData, setShowExtractedData] = useState(true);

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
  const extractCaseButton = () => {
    const caseNumber = search;
    setSearch('');
    setShowExtractedData(false);
    if (caseNumber === '') {

    } else {
      axios.post('http://localhost:8000/case', {caseNumber: caseNumber})
      .then(res => {
        console.log(res.data.data);
        setExtractedData(res.data.data);
        setShowExtractedData(true);
      })
    }
  }






  return (
    <div className="App">
      <NavBar />
      <br /><br /><br />
      
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
  );
}

export default App;
