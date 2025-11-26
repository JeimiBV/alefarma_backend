from functools import lru_cache
import pandas as pd

DATA_URL = "https://tu-url-del-csv.com/dataset.csv"

@lru_cache()
def load_data():
    """Carga el CSV y lo guarda en caché para evitar múltiples requests."""
    print("Cargando datos desde la URL...")
    df = pd.read_csv(DATA_URL)
    return df
