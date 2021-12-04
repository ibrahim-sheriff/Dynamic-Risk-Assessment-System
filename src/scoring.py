"""
Author: Ibrahim Sherif
Date: December, 2021
This script used for scoring the model
"""
import os
import sys
import pickle
import logging
import pandas as pd
from sklearn.metrics import f1_score

from config import MODEL_PATH, TEST_DATA_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def score_model():
    """
    Loads a trained model and the test data, and calculate an F1 score
    for the model on the test data and saves the result to
    the latestscore.txt file
    """
    logging.info("Loading testdata.csv")
    test_df = pd.read_csv(os.path.join(TEST_DATA_PATH, 'testdata.csv'))

    logging.info("Loading trained model")
    model = pickle.load(
        open(
            os.path.join(
                MODEL_PATH,
                'trainedmodel.pkl'),
            'rb'))

    logging.info("Preparing test data")
    y_true = test_df.pop('exited')
    X_df = test_df.drop(['corporation'], axis=1)

    logging.info("Predicting test data")
    y_pred = model.predict(X_df)
    f1 = f1_score(y_true, y_pred)
    print(f"f1 score = {f1}")

    logging.info("Saving scores to text file")
    with open(os.path.join(MODEL_PATH, 'latestscore.txt'), 'w') as file:
        file.write(f"f1 score = {f1}")


if __name__ == '__main__':
    logging.info("Running scoring.py")
    score_model()
