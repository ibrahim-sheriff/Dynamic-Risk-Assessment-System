import os
import json
import shutil


# Load config.json and correct path variable
with open('../config.json', 'r') as f:
    config = json.load(f)

data_path = os.path.join(os.path.abspath('../'), 'data', config['output_folder_path'])
model_path = os.path.join(os.path.abspath('../'), 'model', config['output_model_path']) 
prod_deployment_path = os.path.join(os.path.abspath('../'), 'model', config['prod_deployment_path']) 


# Function for deployment
def deploy_model():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    shutil.copy(os.path.join(data_path, 'ingestedfiles.txt'), prod_deployment_path)
    shutil.copy(os.path.join(model_path, 'trainedmodel.pkl'), prod_deployment_path)
    shutil.copy(os.path.join(model_path, 'latestscore.txt'), prod_deployment_path)


if __name__ == '__main__':
    deploy_model()