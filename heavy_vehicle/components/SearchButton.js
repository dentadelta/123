import { Button, Stack, TextField } from '@mui/material';


export const SearchButton = ({extractCase, handleSearch, handleFocus, handleUnfocus}) => {

    return (
        <Stack spacing={2}>
        <TextField style={{background: "rgb(232, 241, 250)"}} id="search_case" label="Search Case" variant="filled" color='secondary' size='small' onChange={handleSearch} 
        onFocus={handleFocus}
        onBlur={handleUnfocus}/>
        <Button variant='contained' color='warning' onClick={extractCase}>Search</Button>
        </Stack>
    )

}