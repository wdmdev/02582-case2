import os
import numpy as np
from PIL import Image
from skimage import color

def img_gray(img):
    # Sometimes last dimension is 4, but we only keep the 3 first
    X = img / 255 # <- converts to float
    if X.shape[2] > 3:
        X = X[:,:,:3]
    X = color.rgb2gray(X)
    X = X.reshape(X.shape[0], -1)
    return X

def load_img_as_gray(img_path):
    base_size = (200,200)
    img = np.array(Image.open(img_path).resize(base_size))
    img = img_gray(img)
    return img