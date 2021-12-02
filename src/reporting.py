"""
Author: Ibrahim Sherif
Date: December, 2021
This script used to generate a confusion matrix
"""
import os
import sys
import logging
import pandas as pd

from config import MODEL_PATH, TEST_DATA_PATH
from diagnostics import model_predictions
from pretty_confusion_matrix import plot_confusion_matrix_from_data


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def plot_confusion_matrix():
    """
    Calculate a confusion matrix using the test data and the deployed model
    plot the confusion matrix using pretty confusion matrix to the workspace
    """
    logging.info("Loading and preparing testdata.csv")
    test_df = pd.read_csv(os.path.join(TEST_DATA_PATH, 'testdata.csv'))

    y_true = test_df.pop('exited')
    X_df = test_df.drop(['corporation'], axis=1)

    logging.info("Predicting test data")
    y_pred = model_predictions(X_df)

    logging.info("Plotting and saving confusion matrix")
    fig, ax = plot_confusion_matrix_from_data(
        y_true, y_pred, columns=[0, 1], cmap='Blues')

    ax.set_title("Model Confusion Matrix")
    fig.savefig(os.path.join(MODEL_PATH, 'confusionmatrix.png'))


if __name__ == '__main__':
    logging.info("Running reporting.py")
    plot_confusion_matrix()
