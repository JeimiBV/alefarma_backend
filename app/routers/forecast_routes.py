from fastapi import APIRouter
import pandas as pd
from app.models.prophet_loader import model
from app.schemas.forecast_schemas import ForecastRequest
from app.schemas.forecast_schemas import ForecastRangeRequest

router = APIRouter()

@router.post("/forecast")
def forecast(request: ForecastRequest):
    if request.last_date:
        start_date = pd.to_datetime(request.last_date)
    else:
        start_date = model.history['ds'].max()

    future_dates = pd.date_range(
        start=start_date + pd.Timedelta(days=1),
        periods=request.days,
        freq="D"
    )

    future = pd.DataFrame({"ds": future_dates})
    forecast_df = model.predict(future)

    forecast_df[["yhat", "yhat_lower", "yhat_upper"]] = forecast_df[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0)

    result = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result["ds"] = result["ds"].dt.strftime("%Y-%m-%d")

    return {"forecast": result.to_dict(orient="records")}


@router.post("/forecast/range")
def forecast_range(request: ForecastRangeRequest):
    start_date = pd.to_datetime(request.start)
    end_date = pd.to_datetime(request.end)

    future_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    future = pd.DataFrame({"ds": future_dates})

    forecast_df = model.predict(future)
    forecast_df[["yhat", "yhat_lower", "yhat_upper"]] = forecast_df[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0)

    forecast_df["ds"] = forecast_df["ds"].dt.strftime("%Y-%m-%d")

    return {
        "start": request.start,
        "end": request.end,
        "forecast": forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")
    }


@router.get("/last_training_date")
def last_training_date():
    last_date = model.history['ds'].max()
    return {"last_training_date": last_date.strftime("%Y-%m-%d")}
