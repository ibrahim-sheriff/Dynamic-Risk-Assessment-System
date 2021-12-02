"""
Author: Ibrahim Sherif
Date: December, 2021
This script used to deploy the trained model
"""
import os
import sys
import shutil
import logging

from config import DATA_PATH, MODEL_PATH, PROD_DEPLOYMENT_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def deploy_model():
    """
    Copy the latest model pickle file, the latestscore.txt value,
    and the ingestfiles.txt file into the deployment directory
    """
    logging.info("Deploying trained model to production")
    logging.info(
        "Copying trainedmodel.pkl, ingestfiles.txt and latestscore.txt")
    shutil.copy(
        os.path.join(
            DATA_PATH,
            'ingestedfiles.txt'),
        PROD_DEPLOYMENT_PATH)
    shutil.copy(
        os.path.join(
            MODEL_PATH,
            'trainedmodel.pkl'),
        PROD_DEPLOYMENT_PATH)
    shutil.copy(
        os.path.join(
            MODEL_PATH,
            'latestscore.txt'),
        PROD_DEPLOYMENT_PATH)


if __name__ == '__main__':
    logging.info("Running deployment.py")
    deploy_model()
