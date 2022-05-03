import os
import numpy as np
import pickle as pkl
from PIL import Image

class Model():

    def __init__(self, name, embedder, race_classifier, 
                        gender_classifier, age_regressor) -> None:
        self.name = name
        self.embedder = embedder
        self.race_classifier = race_classifier
        self.gender_classifier = gender_classifier
        self.age_regressor = age_regressor

    def save(self):
        '''
        Serializes the model and saves it as a pickle (.pkl) file
        '''
        save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'models', self.name + '.pkl')
        with open(save_path, 'wb') as model:
            pkl.dump(self, model)

    def predict(self, image):
        '''
        Predicts the age, race and gender of the given image

        :param image:       Image to predict on

        :returns:           Triplet of predicted (age, race, gender)
        '''
        embedding = self.embedder.transform(image)
        age = self.age_regressor.predict(embedding)
        race = self.race_classifier.predict(embedding)
        gender = self.gender_classifier.predict(embedding)

        return age, race, gender
    
    def set_embedding(self, embedding):
        self.embedding = embedding

    
    def find_n_best_face_match(self, image, n):
        '''
        Find the face image that is closest to the given image in the model face space embedding.

        :param model:       Model with face space embedding
        :param image:       Image of face to find best match for in the embedded face space

        :returns:           Image of best matching face in embedded face space as a numpy array
        '''
        img = image.reshape(1,-1)
        img_emb = self.embedder.transform(img)
        dists = ((img_emb - self.embedding)**2).sum(1)
        best_match_ids = np.sort(dists)[:n]

        # Load best matching image
        best_matches = []
        for id in best_match_ids:
            best_img_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 
                                        'data', 'raw', 'Faces', f'{id}.jpg')
            img = np.array(Image.open(best_img_path))
            best_matches.append(img)

        return list(zip(best_matches, best_match_ids.tolist()))