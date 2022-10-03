import axios from "axios"
import { useEffect, useState } from "react"

// custom reuseable FastAPI hook
// inputs:
// url: the url of the API
// data: the data to be sent to the API
// buttonClicked: a boolean that is true when the button is clicked

//outputs:
// response: the response from the API
// loading: a boolean that is true when the API is loading
// error: the error message from the API

export function FastAPI(url, data, buttonClicked) {
    const [response, setResponse] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    
    useEffect(() => {
        setLoading(true);
            axios.post(`http://localhost:8000${url}`, data)
            .then(res => {
                setResponse(res.data);
            })
            .catch(err => {
                setError(err);
            })
            .finally(() => {
                setLoading(false);
            })
        
        
    }, [(buttonClicked)]);

    return {response, loading, error}
}
