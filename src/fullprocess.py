import os
import re
import sys
import logging
import pandas as pd
from sklearn.metrics import f1_score

import scoring
import training
import ingestion
import reporting
import deployment
import diagnostics
from config import INPUT_FOLDER_PATH, PROD_DEPLOYMENT_PATH, DATA_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    # Check and read new data
    logging.info("Checking for new data")

    # First, read ingestedfiles.txt
    with open(os.path.join(PROD_DEPLOYMENT_PATH, "ingestedfiles.txt")) as file:
        ingested_files = {line.strip('\n') for line in file.readlines()[1:]}

    # Second, determine whether the source data folder has files that aren't
    # listed in ingestedfiles.txt
    source_files = set(os.listdir(INPUT_FOLDER_PATH))

    # Deciding whether to proceed, part 1
    # If you found new data, you should proceed. otherwise, do end the process
    # here
    if len(source_files.difference(ingested_files)) == 0:
        logging.info("No new data found")
        return None

    # Ingesting new data
    logging.info("Ingesting new data")
    ingestion.merge_multiple_dataframe()

    # Checking for model drift
    logging.info("Checking for model drift")

    # Check whether the score from the deployed model is different from the
    # score from the model that uses the newest ingested data
    with open(os.path.join(PROD_DEPLOYMENT_PATH, "latestscore.txt")) as file:
        deployed_score = re.findall(r'\d*\.?\d+', file.read())[0]
        deployed_score = float(deployed_score)

    data_df = pd.read_csv(os.path.join(DATA_PATH, 'finaldata.csv'))
    y_df = data_df.pop('exited')
    X_df = data_df.drop(['corporation'], axis=1)

    y_pred = diagnostics.model_predictions(X_df)
    new_score = f1_score(y_df.values, y_pred)

    # Deciding whether to proceed, part 2
    logging.info(f"Deployed score = {deployed_score}")
    logging.info(f"New score = {new_score}")

    # If you found model drift, you should proceed. otherwise, do end the
    # process here
    if(new_score >= deployed_score):
        logging.info("No model drift occurred")
        return None

    # Re-training
    logging.info("Re-training model")
    training.train_model()
    logging.info("Re-scoring model")
    scoring.score_model()

    # Re-deployment
    logging.info("Re-deploying model")

    # If you found evidence for model drift, re-run the deployment.py script
    deployment.deploy_model()

    # Diagnostics and reporting
    logging.info("Running diagnostics and reporting")

    # Run diagnostics.py and reporting.py for the re-deployed model
    reporting.plot_confusion_matrix()
    reporting.generate_pdf_report()
    os.system("python apicalls.py")


if __name__ == '__main__':
    main()
