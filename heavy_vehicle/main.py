from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import time

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Case(BaseModel):
    caseNumber: str

class GetConditions(BaseModel):
    width: float
    height: float
    length: float
    routeList: List[str]

class Condition(BaseModel):
    district: int
    description: str
    criteria: str
    condition: str

class Email(BaseModel):
    to: str
    cc: str
    bcc: str
    subject: str
    body: str



fakeData1 = Condition(district=1, description="test", criteria="test", condition="test")
fakeData2 = Condition(district=2, description="test", criteria="test", condition="test")
fakeData3 = Condition(district=3, description="test", criteria="test", condition="test")
fakeData = [fakeData1, fakeData2, fakeData3]


@app.post("/case")
async def case(ccase: Case):
    return_data = {}
    selenium_case = ccase.caseNumber
    return_data["caseNumber"] = '342324v2'
    return_data["width"] = "Width: 3.5m"
    return_data["height"] = "Height: 2.5m"
    return_data["length"] = "Length: 1.5m"
    return_data["routeList"] = ["Route 1", "Route 2", "Route 3"]
    return_data["imageURL"] = "https://source.unsplash.com/random"
    return_data['vehicleImageURL'] = "https://source.unsplash.com/random"
    time.sleep(2)
    return {"data": return_data}


@app.post("/getConditions")
async def getConditions(request: GetConditions):
    return_data= fakeData
    return {'data': return_data}

@app.post("/export")
async def updateCondition(request: List[Condition]):
    return_data = request
    message = 'Condition updated successfully'
    return {'message': message}

@app.post("/sendEmail")
async def sendEmail(request: Email):
    return_data = request
    message = 'Email sent successfully'
    return {'message': message}





