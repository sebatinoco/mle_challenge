import fastapi
from fastapi.responses import JSONResponse
import pandas as pd

from .model import DelayModel
from .utils import FlightsRequest

model = DelayModel()

app = fastapi.FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: FlightsRequest) -> dict:

    # gather flight data from the request
    flights = request.flights
    flights_data = pd.DataFrame([flight.dict() for flight in flights])

    try:
        # preprocess the flight data
        features = model.preprocess(flights_data)

        # make predictions using the preprocessed features
        content = {
            "predict": model.predict(features),
        }

        return JSONResponse(content=content, status_code=200)
    
    except ValueError as e:
        # handle any exceptions that occur during prediction
        return JSONResponse(
            status_code=400,
            content={
                "error": str(e),
                "message": "Invalid input data. Please check the flight details."
            }
        )