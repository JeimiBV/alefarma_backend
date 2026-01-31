from pydantic import BaseModel

class ForecastRequest(BaseModel):
    days: int
    last_date: str | None = None

class ForecastRangeRequest(BaseModel):
    start: str
    end: str
