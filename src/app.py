from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import subprocess
import json
import os

import diagnostics


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

# Load config.json and get input and output paths
with open('../config.json', 'r') as f:
    config = json.load(f) 

data_path = os.path.join(os.path.abspath('../'), 'data', config['output_folder_path'])


@app.route('/')
def index():
    return "Hello World"

# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():        
    # call the prediction function you created in Step 3
    #filepath = request.args.get('filepath')
    filepath = request.get_json()['filepath']
    
    df = pd.read_csv(filepath)
    df = df.drop(['corporation', 'exited'], axis=1)
    
    preds = diagnostics.model_predictions(df)
    return jsonify(preds.tolist()) # add return value for prediction outputs
    
# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    output = subprocess.run(['python', 'scoring.py'], capture_output=True).stdout
    return output #add return value (a single F1 score number)

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    return jsonify(diagnostics.dataframe_summary()) #return a list of all calculated summary statistics

# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diag():        
    #check timing and percent NA values
    missing = diagnostics.missing_percentage()
    time = diagnostics.execution_time()
    outdated = diagnostics.outdated_packages_list()
    
    ret = {
        'missing_percentage': missing,
        'execution_time': time,
        'outdated_packages': outdated
    }
    
    return jsonify(ret)  #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
