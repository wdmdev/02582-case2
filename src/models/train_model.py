import os

import numpy as np
import pandas as pd
from skimage import color
from sklearn.decomposition import DictionaryLearning

from src.features.serialization import load_features

def divide_into_agegroups(df: pd.DataFrame, n_groups: int):
    age_lists = np.array_split(np.arange(df["age"].min(), df["age"].max()), n_groups)
    datasets = [df[[age in age_list for age in df["age"]]] for age_list in age_lists]
    return datasets

def build_data_matrix(df: pd.DataFrame):
    X = np.stack(df["image"])
    X = X / 255 # <- converts to float
    X = color.rgb2gray(X)
    X = X.reshape(X.shape[0], -1)
    return X

def train(X: np.ndarray):
    model = DictionaryLearning()
    Z = model.fit_transform(X)
    return Z

def main():
    N = 10
    df = load_features()
    df = df.sample(1000) # <- memory limitation
    age_dfs = divide_into_agegroups(df, N)
    for age_df in age_dfs:
        print(f"Running for age group {age_df.age.min()}-{age_df.age.max()}...")
        X = build_data_matrix(age_df)
        Z = train(X)


if __name__ == "__main__":
    main()
