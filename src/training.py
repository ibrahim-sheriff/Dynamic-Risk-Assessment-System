import os
import json
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Load config.json and get path variables
with open('../config.json', 'r') as f:
    config = json.load(f)

data_path = os.path.join(os.path.abspath('../'), 'data', config['output_folder_path'])
model_path = os.path.join(os.path.abspath('../'), 'model', config['output_model_path']) 


# Function for training the model
def train_model():

    # Load data
    data_df = pd.read_csv(os.path.join(data_path, 'finaldata.csv'))
    y_df = data_df.pop('exited')
    X_df = data_df.drop(['corporation'], axis=1)
    
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)

    #fit the logistic regression to your data
    model.fit(X_df, y_df)

    #write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(model, open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb'))


if __name__ == '__main__':
    train_model()
    