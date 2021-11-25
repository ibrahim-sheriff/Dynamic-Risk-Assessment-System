import os
import json
import requests

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Load config.json and get input and output paths
with open('../config.json', 'r') as f:
    config = json.load(f) 

test_data_path = os.path.join(os.path.abspath('../'), 'data', config['test_data_path'], 'testdata.csv')
model_path = os.path.join(os.path.abspath('../'), 'model', config['output_model_path']) 

#Call each API endpoint and store the responses
#response_pred = requests.get(f'{URL}/prediction?filepath={test_data_path}').text
response_pred = requests.post(f'{URL}/prediction', json={'filepath': test_data_path}).text
response_scor = requests.get(f'{URL}/scoring').text
response_stat = requests.get(f'{URL}/summarystats').text
response_diag = requests.get(f'{URL}/diagnostics').text

#combine all API responses
#responses = #combine reponses here

#write the responses to your workspace
with open(os.path.join(model_path, 'apireturns.txt'), 'w') as file:
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
