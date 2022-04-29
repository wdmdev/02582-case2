import os
import sys
import argparse
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
    Model('model_all',
        embedder=embedder,
        race_classifier=KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        gender_classifier=KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        age_regressor=KNeighborsRegressor(n_neighbors=5, n_jobs=-1))
    for embedder in EMBEDDERS
]

S_AND_P_AMOUNT = 0.5

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

def gauss(img: np.ndarray) -> np.ndarray:
    return img + np.random.normal(img.std(), size=img.shape)

def salt_and_pepper(img: np.ndarray) -> np.ndarray:
    noisy = img.copy()
    # Salt mode
    num_salt = np.ceil(S_AND_P_AMOUNT * img.size * 0.5)
    noisy[np.random.randint(0, img.size - 1, int(num_salt))] = 1
    # Pepper mode
    num_pepper = np.ceil(S_AND_P_AMOUNT * img.size * 0.5)
    noisy[np.random.randint(0, img.size - 1, int(num_pepper))] = 0
    return noisy

def run_noise_robustness_test(model: Model, Xtest: np.ndarray, e_test: np.ndarray):
    for noisename, noisefunc in zip(
            (f"Gaussian", f"Salt and Pepper ({S_AND_P_AMOUNT})"),
            (gauss, salt_and_pepper)
        ):
        preds, truth = np.zeros(Xtest.shape[0]), np.arange(Xtest.shape[0])
        for i, img in enumerate(Xtest):
            noisy = noisefunc(img)
            new_emb = model.embedder.transform(noisy.reshape(1, -1))
            dists = ((new_emb - e_test)**2).sum(1)
            # TODO: What about top K-performance?
            preds[i] = dists.argmin()

        with open(os.path.join(REPORT_PATH, model.name + '.txt'), 'a') as report_file:
            msg = f'''
            Test set {noisename} noise prediction accuracy {(preds == truth).mean()*100:.2f}
            '''
            report_file.write(msg)
            print(msg)



def build_data_matrix(df: pd.DataFrame):
    X = np.stack(df["image"])
    X = X / 255 # <- converts to float
    X = color.rgb2gray(X)
    X = X.reshape(X.shape[0], -1)
    return X

def main(run_feature: bool=True, run_noise: bool=True):
    df = load_features()
    X = build_data_matrix(df)
    # df = df.sample(100000) # <- memory limitation
    Xtrain, Xtest = build_data_matrix(df[df["train"]]), build_data_matrix(df[~df["train"]])
    for model in MODELS:
        with open(os.path.join(REPORT_PATH, model.name+'.txt'), 'a') as report_file:
            report_file.write('####################################################')
            report_file.write(f'Run model: {model.name}, time: {dt.datetime.now()}')
        print(f"Creating face embeddings using {model.embedder}...")
        train_embeddings = model.embedder.fit_transform(Xtrain)
        test_embeddings = model.embedder.transform(Xtest)
        if run_feature:
            run_feature_robustness_test(model.race_classifier, model.name, df, train_embeddings, test_embeddings, 'race')
            run_feature_robustness_test(model.gender_classifier, model.name, df, train_embeddings, test_embeddings, 'gender')
            run_feature_robustness_test(model.age_regressor, model.name, df, train_embeddings, test_embeddings, "age")
        if run_noise:
            run_noise_robustness_test(model, Xtest, test_embeddings)

        # Train on full data and save final model
        print('Fitting and saving final model...')
        embedding = model.embedder.fit_transform(X)
        model.set_embedding(embedding)
        model.race_classifier.fit(embedding, df['race'].values)
        model.gender_classifier.fit(embedding, df['gender'].values)
        model.age_regressor.fit(embedding, df['age'].values)
        model.save()
        print('Final model saved!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--test", default="all", help="What tests to run")
    t = parser.parse_args().test
    main(run_feature=t=="all" or t=="feature", run_noise=t=="all" or t=="noise")
