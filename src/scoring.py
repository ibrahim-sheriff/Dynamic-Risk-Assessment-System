import os
import json
import pickle
import pandas as pd
from sklearn.metrics import f1_score


# Load config.json and get input and output paths
with open('../config.json', 'r') as f:
    config = json.load(f) 

test_data_path = os.path.join(os.path.abspath('../'), 'data', config['test_data_path'])
model_path = os.path.join(os.path.abspath('../'), 'model', config['output_model_path']) 


# Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    model = pickle.load(open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb'))

    y_true = test_df.pop('exited')
    X_df = test_df.drop(['corporation'], axis=1)

    y_pred = model.predict(X_df)
    f1 = f1_score(y_true, y_pred)
    print(f"f1 score = {f1}")

    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as file:
        file.write(str(f1))

if __name__ == '__main__':
    score_model()