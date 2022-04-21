import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


from src.features.serialization import load_features
from labels import plot_data_dists

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_PATH, '..', '..', 'reports', 'figures')


if __name__ == "__main__":
    df = load_features()
    plot_data_dists(df, OUT_PATH)
