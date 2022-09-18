from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000"
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class Lot(BaseModel):
    lot_number: str
    lot_description: str
    status: str
    road: str
    start_ch: float
    end_ch: float
    start_date: str
    end_date: str
    measured_quantity: float
    baseline_quantity: float


lot1 = Lot(lot_number="CAP-SWS-SC1-001", lot_description="CAP work on Road 10E, from chainage 0.01km to chainage 0.02km, LHS", status="Closed", road="10E", start_ch=0.0, end_ch=1.0, start_date="2021-01-01", end_date="2021-01-01", measured_quantity=100.0, baseline_quantity=100.0)
lot2 = Lot(lot_number="CAP-SWS-SC1-002", lot_description="CAP work on Road 10E, from chainage 0.02km to chainage 0.03km, LHS", status="Closed", road="10E", start_ch=0.0, end_ch=1.0, start_date="2021-01-01", end_date="2021-01-01", measured_quantity=100.0, baseline_quantity=100.0)

Lot_data = [lot1, lot2]


@app.get("/form")
async def read_item(request: Request):
    return templates.TemplateResponse("lotmanagement.html",  context={'request': request, 'lots': Lot_data})

@app.post("/form")
def form_post(request: Request, 
lot_number: str = Form(...), 
lot_description: str = Form(...),
status: str = Form(...),
road: str = Form(...),
start_ch: float = Form(...),
end_ch: float = Form(...),
start_date: str = Form(...),
end_date: str = Form(...),
measured_quantity: float = Form(...),
baseline_quantity: float = Form(...),):
    
    new_lot = Lot(lot_number=lot_number, lot_description=lot_description, status=status, road=road, start_ch=start_ch, end_ch=end_ch, start_date=start_date, end_date=end_date, measured_quantity=measured_quantity, baseline_quantity=baseline_quantity)
    for i in range(len(Lot_data)):
        lot = Lot_data[i]
        if lot.lot_number == new_lot.lot_number:
            Lot_data[i] = new_lot
            break
    else:
        Lot_data.append(new_lot)
    





    return templates.TemplateResponse("lotmanagement.html",  context={'request': request, 'lots': Lot_data})