"""
Author: Ibrahim Sherif
Date: December, 2021
This script used to call the APIs and generate a report file that includes
Model predictions and scores on test data
Summary statistics and missing data percentage on ingested train data
Execution time for training and ingestion functions
Outdated packages list
"""
import os
import sys
import logging
import requests

from config import TEST_DATA_PATH, MODEL_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

URL = "http://127.0.0.1:8000"

logging.info(
    f"Post request /prediction for {os.path.join(TEST_DATA_PATH, 'testdata.csv')}")
# response_pred =
# requests.get(f'{URL}/prediction?filepath={os.path.join(TEST_DATA_PATH,
# 'testdata.csv')}').text
response_pred = requests.post(
    f'{URL}/prediction',
    json={
        'filepath': os.path.join(TEST_DATA_PATH, 'testdata.csv')}).text

logging.info("Get request /scoring")
response_scor = requests.get(f'{URL}/scoring').text

logging.info("Get request /summarystats")
response_stat = requests.get(f'{URL}/summarystats').text

logging.info("Get request /diagnostics")
response_diag = requests.get(f'{URL}/diagnostics').text

logging.info("Generating report text file")
with open(os.path.join(MODEL_PATH, 'apireturns.txt'), 'w') as file:
    file.write('Ingested Data\n\n')
    file.write('Statistics Summary\n')
    file.write(response_stat)
    file.write('\nDiagnostics Summary\n')
    file.write(response_diag)
    file.write('\n\nTest Data\n\n')
    file.write('Model Predictions\n')
    file.write(response_pred)
    file.write('\nModel Score\n')
    file.write(response_scor)
