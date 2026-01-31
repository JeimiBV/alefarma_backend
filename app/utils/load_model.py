from pathlib import Path
import pickle
import requests

MODEL_URL = "https://github.com/JeimiBV/alefarma/raw/refs/heads/main/output/modelo_prophet.pkl"
MODEL_PATH = Path("modelo_prophet.pkl")

def load_prophet_model():
    if not MODEL_PATH.exists():
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model
