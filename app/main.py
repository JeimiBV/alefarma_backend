from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.forecast_routes import router as forecast_router

app = FastAPI(
    title="Prophet Sales Forecast API",
    description="API para predecir ventas usando Prophet",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas
app.include_router(forecast_router)

@app.get("/")
def root():
    return {"message": "API de predicción Prophet funcionando"}
