from datetime import datetime
from pydantic import BaseModel
from typing import List

def get_min_diff(data):
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff
class Flight(BaseModel):
    OPERA: str # Airline operator
    TIPOVUELO: str # Flight type (e.g., domestic, international)
    MES: int # Month of the flight (1-12)

class FlightsRequest(BaseModel):
    flights: List[Flight] # List of flights