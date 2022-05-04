import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import pandas as pd
import pickle as pkl
from PIL import Image
from skimage import color
from sklearn.metrics import pairwise_distances


def load_images(img_folder):
    imgs = []
    img_names = []
    for img_file in sorted(os.listdir(img_folder)):
        img = np.array(Image.open(os.path.join(img_folder, img_file)))
        imgs.append(img)
        img_names.append(img_file)
    
    X = np.stack(imgs)
    X = X / 255 # <- converts to float
    X = color.rgb2gray(X)
    X = X.reshape(X.shape[0], -1)
    return X, img_names


def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        return pkl.load(model_file)


def compute_dists(img_folder, model_path):
    model = load_model(model_path)
    X, img_names = load_images(img_folder)
    img_names = [' '.join(n[:-7].split('_')).title() for n in img_names]

    embedding = model.embedder.transform(X)
    races = model.race_classifier.predict(embedding)
    genders = model.gender_classifier.predict(embedding)
    ages = model.age_regressor.predict(embedding)
    dists = pairwise_distances(embedding)
    df = pd.DataFrame(dists, columns=img_names, index=img_names)
    df['Predicted Race'] = races
    df['Predicted Gender'] = genders
    df['Predicted Age'] = ages
    return df

if __name__ == '__main__':
    base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
    face_folder = 'hollywood_faces'
    img_folder = os.path.join(base_path, 'data', face_folder)
    model_path = os.path.join(base_path, 'models', 'soeren', 'model-minibatchdictionarylearning.pkl')

    df = compute_dists(img_folder, model_path)
    df.to_csv(os.path.join(base_path, 'reports', f'{face_folder}_distances.csv'))