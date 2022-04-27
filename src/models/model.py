import os
import pickle as pkl

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

