import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import pandas as pd
from skimage import color

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score


from src.features.serialization import load_features

EMBEDDERS = (
    PCA(n_components=100),
)

CLASSIFIERS = (
    KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
)
REGRESSORS = (
    KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
)

def run_feature_robustness_test(model, df: pd. DataFrame, e_train: np.ndarray, e_test: np.ndarray, feature: str):
    print(f"\tTesting {feature} robustness of embeddings using {model}...")
    y = df[feature].values
    y_train, y_test = y[df["train"]], y[~df["train"]]

    # Train on embeddings
    model.fit(e_train, y_train)
    pred_train, pred_test = model.predict(e_train), model.predict(e_test)

    if feature == "age":
        # Regression stats
        print(f"Train MSE, r^2 {mean_squared_error(y_train, pred_train):.2f}, {r2_score(y_train, pred_train):.2f}")
        print(f"Test MSE, r^2 {mean_squared_error(y_test, pred_test):.2f}, {r2_score(y_test, pred_test):.2f}")
    else:
        # Classification stats
        print("Train conf. mat., report", confusion_matrix(y_train, pred_train), classification_report(y_train, pred_train), sep="\n")
        print("Test conf. mat., report", confusion_matrix(y_test, pred_test), classification_report(y_test, pred_test), sep="\n")

def build_data_matrix(df: pd.DataFrame):
    X = np.stack(df["image"])
    X = X / 255 # <- converts to float
    X = color.rgb2gray(X)
    X = X.reshape(X.shape[0], -1)
    return X

def main():
    df = load_features()
    df = df.sample(1000) # <- memory limitation
    Xtrain, Xtest = build_data_matrix(df[df["train"]]), build_data_matrix(df[~df["train"]])
    for embedder in EMBEDDERS:
        print(f"Creating face embeddings using {embedder}...")
        train_embeddings = embedder.fit_transform(Xtrain)
        test_embeddings = embedder.transform(Xtest)
        for classifier in CLASSIFIERS:
            classifier = clone(classifier) # Make sure there is no leakage between runs
            for f in "race", "gender":
                run_feature_robustness_test(classifier, df, train_embeddings, test_embeddings, f)
        for regressor in REGRESSORS:
            regressor = clone(regressor)
            run_feature_robustness_test(regressor, df, train_embeddings, test_embeddings, "age")

if __name__ == "__main__":
    main()
