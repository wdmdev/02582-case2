import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import pandas as pd
from skimage import color
import datetime as dt

from sklearn.decomposition import PCA, MiniBatchDictionaryLearning, NMF
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score

from src.models.model import Model
from src.features.serialization import load_features

EMBEDDERS = [
    PCA(n_components=100)
    # MiniBatchDictionaryLearning(n_components=15, alpha=0.1, 
    #     n_iter=50, batch_size=3, random_state=np.random.RandomState(0)),
    # NMF(n_components=100, init="nndsvda", tol=5e-3)
    
]

# We create a 5NN predictor for each embedding algorithm
MODELS = [
    Model('model_100000',
        embedder=embedder,
        race_classifier=KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        gender_classifier=KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        age_regressor=KNeighborsRegressor(n_neighbors=3, n_jobs=-1))
    for embedder in EMBEDDERS
]

BASE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
REPORT_PATH = os.path.join(BASE_PATH, 'reports')

def run_feature_robustness_test(model, model_name:str, df: pd. DataFrame, e_train: np.ndarray, e_test: np.ndarray, feature: str):
    print(f"\tTesting {feature} robustness of embeddings using {model}...")
    y = df[feature].values
    y_train, y_test = y[df["train"]], y[~df["train"]]

    # Train on embeddings
    model.fit(e_train, y_train)
    pred_train, pred_test = model.predict(e_train), model.predict(e_test)

    with open(os.path.join(REPORT_PATH, model_name+'.txt'), 'a') as report_file:
        if feature == "age":
            # Regression stats
            msg = f'''
                    Train MSE, r^2 {mean_squared_error(y_train, pred_train):.2f}, {r2_score(y_train, pred_train):.2f}\n
                    Test MSE, r^2 {mean_squared_error(y_test, pred_test):.2f}, {r2_score(y_test, pred_test):.2f}\n\n
                    '''
            report_file.write(msg)
            print(msg)
        else:
            # Classification stats
            msg = f'''
                    train conf. mat., report\n{confusion_matrix(y_train, pred_train)}\n
                    train conf. mat., report\n{classification_report(y_train, pred_train)}\n
                    test conf. mat., report\n{confusion_matrix(y_test, pred_test)}\n
                    test conf. mat., report\n{classification_report(y_test, pred_test)}\n\n
                    '''
            report_file.write(msg)
            print(msg)

def build_data_matrix(df: pd.DataFrame):
    X = np.stack(df["image"])
    X = X / 255 # <- converts to float
    X = color.rgb2gray(X)
    X = X.reshape(X.shape[0], -1)
    return X

def main():
    df = load_features()
    df = df.sample(100000) # <- memory limitation
    Xtrain, Xtest = build_data_matrix(df[df["train"]]), build_data_matrix(df[~df["train"]])
    for model in MODELS:
        with open(os.path.join(REPORT_PATH, model.name+'.txt'), 'a') as report_file:
            report_file.write('####################################################')
            report_file.write(f'Run model: {model.name}, time: {dt.datetime.now()}')
        print(f"Creating face embeddings using {model.embedder}...")
        train_embeddings = model.embedder.fit_transform(Xtrain)
        test_embeddings = model.embedder.transform(Xtest)
        run_feature_robustness_test(model.race_classifier, model.name, df, train_embeddings, test_embeddings, 'race')
        run_feature_robustness_test(model.gender_classifier, model.name, df, train_embeddings, test_embeddings, 'gender')
        run_feature_robustness_test(model.age_regressor, model.name, df, train_embeddings, test_embeddings, "age")
        model.save()

if __name__ == "__main__":
    main()
