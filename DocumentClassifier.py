import numpy as np

from DataLoader import DataLoader

from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score


class DocumentClassifier(object):
    """Class that separates the main functionality
       into four stages: load data, build model, train model,
       and evaluate model

    """

    def __init__(self):
        """Constructor for Document Classifier

        Note:
            Instantiates a DataLoader object to help load train and test data.

        """

        self.dataloader = DataLoader()

    def load_data(self):
        """Loads data using the DataLoader object and sets up label encoder"""

        self.X_train, self.y_train = self.dataloader.load_data('train')
        self.X_test, self.y_test = self.dataloader.load_data('test')
        self.label_encoder = self.dataloader.label_encoder

    def build_model(self, feature_choice, model_choice):
        """Sets up classifier pipeline using the feature and model of choice

        Args:
            feature_choice (tuple[string, sklearn.Vectorizer]): name of feature and vectorizer
            model_choice (tuple[string, sklearn.Model]): name of model and 

        Returns:
            None

        """

        self.features = feature_choice
        self.model = model_choice
        self.classifier = Pipeline([self.features, self.model])



    def train_model(self):
        """Trains and fits classifier using on training data"""

        self.classifier.fit(self.X_train, self.y_train)


    def evaluate_model(self):
        """Uses classifer to generate predictions using test data
        
        Compares the prediction with the true values and calculates
        the macro f1-score of the classifier
        """

        # Pull model from saved pickle
        self.y_predict = self.classifier.predict(self.X_test)
        feature_name = self.features[0]
        mode_name = self.model[0]

        # Save Macro Average F1 Score
        self.f1_score = f1_score(self.y_test, self.y_predict, average='macro')
