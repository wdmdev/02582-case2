import os
import pandas as pd

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_PATH, '..', '..', 'data', 'processed', 'data.pkl')

def load_features(path: str=DATA_FILE) -> pd.DataFrame:
    return pd.read_pickle(path)

def save_features(df: pd.DataFrame, path: str=DATA_FILE):
    df.to_pickle(path)
