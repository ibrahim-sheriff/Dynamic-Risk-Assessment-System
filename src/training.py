"""
Author: Ibrahim Sherif
Date: December, 2021
This script used for training model on the ingested data
"""
import os
import sys
import pickle
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import DATA_PATH, MODEL_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def train_model():
    """
    Train logistic regression model on ingested data and
    and saves the model
    """
    logging.info("Loading and preparing finaldata.csv")
    data_df = pd.read_csv(os.path.join(DATA_PATH, 'finaldata.csv'))
    y_df = data_df.pop('exited')
    X_df = data_df.drop(['corporation'], axis=1)

    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    logging.info("Training model")
    model.fit(X_df, y_df)

    logging.info("Saving trained model")
    pickle.dump(
        model,
        open(
            os.path.join(
                MODEL_PATH,
                'trainedmodel.pkl'),
            'wb'))


if __name__ == '__main__':
    logging.info("Running training.py")
    train_model()
