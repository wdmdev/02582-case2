import os

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from serialization import save_features

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_PATH, '..', '..', 'data', 'raw')

TRAIN_SPLIT = 0.8

def convert_to_dataframe(path: str=RAW_PATH):
    # Load simple features
    df = pd.read_csv(os.path.join(path, "labels.csv"), header=None, names=("age", "gender", "race"))

    # Load images
    get_img = lambda row: np.array(Image.open(os.path.join(RAW_PATH, "Faces", f"{row.name}.jpg")))
    tqdm.pandas(desc="Loading image arrays")
    df["image"] = df.progress_apply(get_img, axis=1)

    # Divide into train and test
    df["train"] = False
    idx = np.arange(n := len(df))
    np.random.shuffle(idx)
    train_idx = idx[:int(TRAIN_SPLIT*n)]
    df.loc[train_idx, "train"] = True
    return df

if __name__ == "__main__":
    df = convert_to_dataframe()
    save_features(df)
    print(f"Saved processed data frame of {len(df)} images.")
