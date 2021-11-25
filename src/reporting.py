import os
import json
import pickle
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from pretty_confusion_matrix import plot_confusion_matrix_from_data

from diagnostics import model_predictions


# Load config.json and get path variables
with open('../config.json', 'r') as f:
    config = json.load(f)

data_path = os.path.join(os.path.abspath('../'), 'data', config['output_folder_path'])
test_data_path = os.path.join(os.path.abspath('../'), 'data', config['test_data_path'])
model_path = os.path.join(os.path.abspath('../'), 'model', config['output_model_path']) 

# Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace

    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    y_true = test_df.pop('exited')
    X_df = test_df.drop(['corporation'], axis=1)

    y_pred = model_predictions(X_df)

    #ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')
    #plt.savefig(os.path.join(model_path, 'confusionmatrix.png'))
   
    fig, ax = plot_confusion_matrix_from_data(y_true, y_pred, columns=[0, 1], cmap='Blues')
    
    ax.set_title("Model Confusion Matrix")
    fig.savefig(os.path.join(model_path, 'confusionmatrix.png'))


if __name__ == '__main__':
    score_model()
