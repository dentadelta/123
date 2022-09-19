
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

LOT_REGISTER = r"Lot Register.xlsx" 
TEST_REQUEST_REGISTER = r"Test Request Register.xlsx"
NCR_REGISTER = r"NCR Register.xlsx"

# lot register column names are Lot_number, lot_description, status, road, start_ch, end_ch,start_date,end_date,measured_quantity,baseline_quantity
# ncr register column names are ncr_number, receipient,ncr_title,ncr_details,affected_lot
# test request register column names are test_request_no, lot_number, test_request_description, due_date, other_lots, test_type_and_number_of_test  

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

class TestRequest(BaseModel):
    lot_number: str
    test_request_number: str
    test_request_description: str
    due_date: str
    other_lots: str
    test_request_type_number_of_test: str
        
class NCRRegister(BaseModel):
    ncr_number: str
    receipient: str
    ncr_title: str
    ncr_details: str
    affected_lot: str

df = pd.read_excel(LOT_REGISTER)
dic = df.to_dict(orient='record')
Lot_data = []
for d in dic:
    lot = Lot(lot_number=d['lot_number'],
             lot_description=d['lot_description'],
             status=d['status'],
             road=d['road'],
             start_ch=float(d['start_ch']),
             end_ch=float(d['end_ch']),
             start_date=str(d['start_date'])[:-9],
             end_date=str(d['end_date'])[:-9],
             measured_quantity=float(d['measured_quantity']),
             baseline_quantity=float(d['baseline_quantity'])
             )
    Lot_data.append(lot)

@app.get("/form")
async def read_item(request: Request):
    ds = pd.read_excel(TEST_REQUEST_REGISTER)
    test_request_data = ds.to_dict(orient='record')
    dn = pd.read_excel(NCR_REGISTER)
    ncr_request_data = dn.to_dict(orient='record')
    ncr_data = []
    for n in ncr_request_data:
        affected_lots = n['affected_lot']
        affected_lots = affected_lots.split('\n')
        n['affected_lot'] = affected_lots
        ncr_data.append(n)
    
    return templates.TemplateResponse("lotmanagement.html",  context={'request': request, 
                                                                      'lots': Lot_data, 
                                                                      'test_request_data':test_request_data,
                                                                      'ncr_data':ncr_data})
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
baseline_quantity: float = Form(...)):
    ds = pd.read_excel(TEST_REQUEST_REGISTER)
    test_request_data = ds.to_dict(orient='record')
    dn = pd.read_excel(NCR_REGISTER)
    ncr_request_data = dn.to_dict(orient='record')
    ncr_data = []
    for n in ncr_request_data:
        affected_lots = n['affected_lot']
        affected_lots = affected_lots.split('\n')
        n['affected_lot'] = affected_lots
        ncr_data.append(n)
    
    
    
    
    new_lot = Lot(lot_number=lot_number, lot_description=lot_description, status=status, road=road, start_ch=start_ch, end_ch=end_ch, start_date=start_date, end_date=end_date, measured_quantity=measured_quantity, baseline_quantity=baseline_quantity)
    for i in range(len(Lot_data)):
        lot = Lot_data[i]
        if lot.lot_number == new_lot.lot_number:
            Lot_data[i] = new_lot
            break
    else:
        num_rows = len(df)
        df.at[num_rows,'lot_number'] = lot_number
        df.at[num_rows,'lot_description'] = lot_description
        df.at[num_rows,'status'] = status
        df.at[num_rows,'road'] = road
        df.at[num_rows,'start_ch'] = start_ch
        df.at[num_rows,'end_ch'] = end_ch
        df.at[num_rows,'start_date'] = start_date
        df.at[num_rows,'end_date'] = start_date
        df.at[num_rows,'measured_quantity'] = measured_quantity
        df.at[num_rows,'baseline_quantity'] = baseline_quantity
        df.to_excel(LOT_REGISTER,
                   index=False)
        Lot_data.append(new_lot)
    return templates.TemplateResponse("lotmanagement.html", context={'request': request, 
                                                                      'lots': Lot_data, 
                                                                      'test_request_data':test_request_data,
                                                                      'ncr_data':ncr_data})

@app.post("/test_request")
def received_test_request(testRequest: TestRequest):
    ds = pd.read_excel(TEST_REQUEST_REGISTER)
    num_rows = len(ds)
    ds.at[num_rows,'test_request_no'] = testRequest.test_request_number
    ds.at[num_rows,'lot_number'] = testRequest.lot_number
    ds.at[num_rows,'test_request_description'] = testRequest.test_request_description
    ds.at[num_rows,'due_date'] = testRequest.due_date
    ds.at[num_rows,'other_lots'] = testRequest.other_lots
    ds.at[num_rows,'test_type_and_number_of_test'] = testRequest.test_request_type_number_of_test
    
    ds.to_excel(TEST_REQUEST_REGISTER, index=False)
    return {'data':'test request received'}

@app.post("/raisencr")
def raise_non_conformance(ncr: NCRRegister):
    dn = pd.read_excel(NCR_REGISTER)
    num_rows = len(dn)
    dn.at[num_rows,'ncr_number'] = ncr.ncr_number
    dn.at[num_rows,'receipient'] = ncr.receipient
    dn.at[num_rows,'ncr_title'] = ncr.ncr_title
    dn.at[num_rows,'ncr_details'] = ncr.ncr_details
    dn.at[num_rows,'affected_lot'] = ncr.affected_lot
    dn.to_excel(NCR_REGISTER)

    return {'data':'ncr sent'}


@app.post("/search")
def search(request: Request, search_text: str = Form(...)):
    ds = pd.read_excel(TEST_REQUEST_REGISTER)
    test_request_data = ds.to_dict(orient='record')
    dn = pd.read_excel(NCR_REGISTER)
    ncr_request_data = dn.to_dict(orient='record')
    ncr_data = []
    for n in ncr_request_data:
        affected_lots = n['affected_lot']
        affected_lots = affected_lots.split('\n')
        n['affected_lot'] = affected_lots
        ncr_data.append(n)
    
    Lot_data = pd.read_excel(LOT_REGISTER).to_dict(orient='record')
    search_results = []
    for lot in Lot_data:
        if search_text in lot['lot_number']:
            search_results.append(lot)
    return templates.TemplateResponse("lotmanagement.html", context={'request': request, 
                                                                      'lots': search_results, 
                                                                      'test_request_data':test_request_data,
                                                                      'ncr_data':ncr_data})