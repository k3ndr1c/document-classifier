import os
import spacy
import sklearn
import pandas as pd
import numpy as np
import csv
import pickle

from string import punctuation

from sklearn import preprocessing

from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokenizer import Tokenizer

# Spacy's pretrained NLP models to help parse text
nlp = spacy.load("en_core_web_sm")

# Stop words for tokenizer
stop_words = set(STOP_WORDS)

# Symbols set to help clean data
symbols = set(punctuation + '\n') 


class DataLoader(object):
    """Class that loads training and testing data and preprocesses it if
       not already preprocessed 

    """

    def __init__(self):
        """Initializes label encoder"""

        self.label_encoder = self._get_label_encoder()
    
    def load_data(self, data_type):
        """Returns x and y data based on data_type

        Args:
            data_type (DataType): data type choice (either DataType.Train or DataType.Test)

        Returns:
            X (np.array): X data (cleaned text data)
            y (np.array): y data (label data)

        """

        # Validate data type
        if data_type not in ['test', 'train']:
            raise ValueError('Not a valid data type')

        # Check if already preprocessed
        pickle_file_path = f'preprocessing/{data_type}/{data_type}_data.pkl'
        if not os.path.exists(pickle_file_path):
            self._preprocess_data(data_type)

        X, y = self._read_from_pickle(pickle_file_path)
        return X, y


    def _read_from_pickle(self, pickle_file_path):
        """Reads already processed data from pickle

        Args:
            pickle_file_path (str): location of pickle file

        Returns:
            X (np.array): X data (cleaned text data)
            y (np.array): y data (label data)

        """
        df = pd.read_pickle(pickle_file_path)

        X = df['text'].fillna(' ').to_numpy()
        y = df['label'].to_numpy()
        return X, y

    def _preprocess_data(self, data_type):
        """Writes cleaned text data into csv, transforms data into pandas dataframe
           and stores dataframe into pickle file

        Args:
            data_type (DataType): data type choice (either DataType.Train or DataType.Test)

        Returns:
            None

        """

        folder_path = f'data/{data_type}'
        csv_file_path = f'preprocessing/{data_type}/{data_type}_data.csv'
        pickle_file_path = f'preprocessing/{data_type}/{data_type}_data.pkl'

        # Initialize cleaned data csv file
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["label", "text"])

        # Go through each folder and write cleaned data into csv
        with os.scandir(folder_path) as folders:
            # Go through each category
            for category_folder in folders:
                label = self.label_encoder.transform([category_folder.name])[0]

                # Go through each data file in folder
                with os.scandir(category_folder.path) as data_folder:
                    self._process_text_file(data_folder, data_type, label)

        # Create a pandas dataframe from csv data
        df = pd.read_csv(csv_file_path)

        # Cache dataframe into pickle
        df.to_pickle(pickle_file_path)

    def _process_text_file(self, data_folder, data_type, label):
        """Cleans data file based on predicate filters
            and appends (label, cleaned_text) to csv
        
        Args:
            data_folder (Dir): directory object where data is held
            dt (str): data type
            label (str): label for data

        Returns:
            None

        """

        for data_file in data_folder:
            # Skips header of email
            found_first_new_line = True
            cleaned_text = []
            with open(data_file, encoding="utf8", errors="replace") as f:
                for line in f:
                    if line == '\n':
                        # found_first_new_line = True
                        continue
                    elif found_first_new_line:
                        # Separates text data into tokens and parses out stop words and symbols
                        tokens = nlp(line)
                        words = [token.lemma_.lower() for token in tokens]
                        words = [w for w in words if self._predicate_filter(w)]
                        for w in words:
                            cleaned_text.append(w)
        
            # Append data into csv
            with open(f'preprocessing/{data_type}/{data_type}_data.csv', 'a+', newline='') as write_file:
                writer = csv.writer(write_file)
                cleaned_text = ' '.join(cleaned_text)
                writer.writerow([label, cleaned_text])

    def _predicate_filter(self, word):
        """Returns whether or not the word should be added

        Args:
            word (str): input word
        
        Returns:
            bool

        """
        
        # boolean filters
        IS_ALPHA = word.isalpha()
        NOT_IN_STOP_WORDS = word not in stop_words
        NOT_IN_SYMBOLS = word not in symbols

        return IS_ALPHA and NOT_IN_STOP_WORDS and NOT_IN_STOP_WORDS

    def _get_label_encoder(self, folder_path='data/train'):
        """Returns label encoder based on folder name inside data folder

        Args:
            folder_path (str): directory location where folder labels are located
        
        Returns:
            None

        """

        # Go through all folders names to create label encoder
        labels = []
        label_encoder = preprocessing.LabelEncoder()

        with os.scandir(folder_path) as folders:
            for fldr in folders:
                labels.append(fldr.name)
        label_encoder.fit(labels)
        return label_encoder

