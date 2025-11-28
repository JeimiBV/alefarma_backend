from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import pickle
import requests
import pandas as pd
from datetime import datetime

app = FastAPI(
    title="Prophet Sales Forecast API",
    description="API para predecir ventas usando Prophet",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL = "https://github.com/JeimiBV/alefarma/raw/refs/heads/main/output/modelo_prophet.pkl"
MODEL_PATH = Path("modelo_prophet.pkl")

if not MODEL_PATH.exists():
    r = requests.get(MODEL_URL)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

class ForecastRequest(BaseModel):
    days: int
    last_date: str = None

@app.get("/")
def root():
    return {"message": "API de predicción Prophet funcionando"}

@app.post("/forecast")
def forecast(request: ForecastRequest):
    if request.last_date:
        start_date = pd.to_datetime(request.last_date)
    else:
        start_date = model.history['ds'].max()

    future_dates = pd.date_range(
        start=start_date + pd.Timedelta(days=1),
        periods=request.days,
        freq='D'
    )
    future = pd.DataFrame({'ds': future_dates})
    forecast_df = model.predict(future)

    forecast_df[['yhat','yhat_lower','yhat_upper']] = forecast_df[['yhat','yhat_lower','yhat_upper']].clip(lower=0)

    result = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    result['ds'] = result['ds'].dt.strftime('%Y-%m-%d')

    return {"forecast": result.to_dict(orient='records')}


class ForecastRangeRequest(BaseModel):
    start: str
    end: str
    
@app.post("/forecast/range")
def forecast_range(request: ForecastRangeRequest):
    start_date = pd.to_datetime(request.start)
    end_date = pd.to_datetime(request.end)
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future = pd.DataFrame({'ds': future_dates})
    forecast_df = model.predict(future)

    forecast_df[['yhat','yhat_lower','yhat_upper']] = forecast_df[['yhat','yhat_lower','yhat_upper']].clip(lower=0)

    forecast_df['ds'] = forecast_df['ds'].dt.strftime('%Y-%m-%d')
    return {
        "start": request.start,
        "end": request.end,
        "forecast": forecast_df[['ds','yhat','yhat_lower','yhat_upper']].to_dict(orient='records')
    }

@app.get("/last_training_date")
def last_training_date():
    last_training_date = model.history['ds'].max()

    return {"last_training_date": last_training_date.strftime('%Y-%m-%d')}
