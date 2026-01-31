import pandas as pd

def forecast_by_days(model, days, last_date):
    if last_date:
        start_date = pd.to_datetime(last_date)
    else:
        start_date = model.history["ds"].max()

    future_dates = pd.date_range(
        start=start_date + pd.Timedelta(days=1),
        periods=days,
        freq="D"
    )
    
    future = pd.DataFrame({"ds": future_dates})
    forecast_df = model.predict(future)
    forecast_df[['yhat','yhat_lower','yhat_upper']] = \
        forecast_df[['yhat','yhat_lower','yhat_upper']].clip(lower=0)

    forecast_df['ds'] = forecast_df['ds'].dt.strftime('%Y-%m-%d')
    return forecast_df[['ds','yhat','yhat_lower','yhat_upper']]
