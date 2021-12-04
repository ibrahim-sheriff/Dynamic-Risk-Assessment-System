import os
import json


# Load config.json and correct path variable
with open('../config.json', 'r') as file:
    CONFIG = json.load(file)

INPUT_FOLDER_PATH = os.path.join(
    os.path.abspath('../'),
    'data',
    CONFIG['input_folder_path'])
DATA_PATH = os.path.join(
    os.path.abspath('../'),
    'data',
    CONFIG['output_folder_path'])
TEST_DATA_PATH = os.path.join(
    os.path.abspath('../'),
    'data',
    CONFIG['test_data_path'])
MODEL_PATH = os.path.join(
    os.path.abspath('../'),
    'model',
    CONFIG['output_model_path'])
PROD_DEPLOYMENT_PATH = os.path.join(os.path.abspath(
    '../'), 'model', CONFIG['prod_deployment_path'])
