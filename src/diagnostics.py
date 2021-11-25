import os
import json
import pickle
import timeit
import subprocess
import pandas as pd
import numpy as np


# Load config.json and get input and output paths
with open('../config.json', 'r') as f:
    config = json.load(f) 

data_path = os.path.join(os.path.abspath('../'), 'data', config['output_folder_path'])
test_data_path = os.path.join(os.path.abspath('../'), 'data', config['test_data_path'])
model_path = os.path.join(os.path.abspath('../'), 'model', config['output_model_path']) 

# Function to get model predictions
def model_predictions(X_df):
    # read the deployed model and a test dataset, calculate predictions
    model = pickle.load(open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb'))

    y_pred = model.predict(X_df)
    return y_pred

# Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    data_df = pd.read_csv(os.path.join(data_path, 'finaldata.csv'))
    data_df = data_df.drop(['exited'], axis=1)
    data_df = data_df.select_dtypes('number')
    
    statistics_list = []
    for col in data_df.columns:
        mean = data_df[col].mean()
        median = data_df[col].median()
        std = data_df[col].std()
        
        statistics_list.append({'name': col, 'mean': mean, 'median': median, 'std': std})
    
    return statistics_list

# Function to get missing values percentage
def missing_percentage():
    data_df = pd.read_csv(os.path.join(data_path, 'finaldata.csv'))
    data_df = data_df.drop(['corporation', 'exited'], axis=1)
    
    missing_list = [{'name': col, 'percentage': perc} for col, perc in zip(data_df.columns, data_df.isna().sum() / data_df.shape[0] * 100)]
    
    return missing_list


def ingestion_timing():
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    timing = timeit.default_timer() - starttime
    return timing


def training_timing():
    starttime = timeit.default_timer()
    os.system('python training.py')
    timing = timeit.default_timer() - starttime   
    return timing


# Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    ingestion_time = []
    for _ in range(20):
        time = ingestion_timing()
        ingestion_time.append(time)
    
    training_time = []
    for _ in range(20):
        time = training_timing()
        training_time.append(time)
    
    ret_list = [
        {'ingest_time_mean': np.mean(ingestion_time)},
        {'train_time_mean': np.mean(training_time)}
    ]
    
    return ret_list

# Function to check dependencies
def outdated_packages_list():
    #get a list of 
 
    dependencies = subprocess.run('pip-outdated ../requirements.txt', stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    
    return dependencies.stdout


if __name__ == '__main__':
    
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X_df = test_df.drop(['corporation', 'exited'], axis=1)

    print("Model predictions:", model_predictions(X_df), end='\n\n')

    print("Summary statistics")
    print(json.dumps(dataframe_summary(), indent=4), end='\n\n')

    print("Missing percentage")
    print(json.dumps(missing_percentage(), indent=4), end='\n\n')

    print("Execution time")
    print(json.dumps(execution_time(), indent=4), end='\n\n')


    print(outdated_packages_list())
