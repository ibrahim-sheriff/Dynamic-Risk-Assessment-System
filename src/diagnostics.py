"""
Author: Ibrahim Sherif
Date: December, 2021
This script used to create functions for model predictions
and diagnostics
"""
import os
import sys
import json
import pickle
import timeit
import logging
import subprocess
import numpy as np
import pandas as pd

from config import DATA_PATH, TEST_DATA_PATH, MODEL_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def model_predictions(X_df):
    """
    Loads deployed model to predict on data provided

    Args:
        X_df (pandas.DataFrame): Dataframe with features

    Returns:
        y_pred: Model predictions
    """
    logging.info("Loading deployed model")
    model = pickle.load(
        open(
            os.path.join(
                MODEL_PATH,
                'trainedmodel.pkl'),
            'rb'))

    logging.info("Running predictions on data")
    y_pred = model.predict(X_df)
    return y_pred


def dataframe_summary():
    """
    Loads finaldata.csv and calculates mean, median and std
    on numerical data

    Returns:
        list[dict]: Each dict contains column name, mean, median and std
    """
    logging.info("Loading and preparing finaldata.csv")
    data_df = pd.read_csv(os.path.join(DATA_PATH, 'finaldata.csv'))
    data_df = data_df.drop(['exited'], axis=1)
    data_df = data_df.select_dtypes('number')

    logging.info("Calculating statistics for data")
    statistics_list = []
    for col in data_df.columns:
        mean = data_df[col].mean()
        median = data_df[col].median()
        std = data_df[col].std()

        statistics_list.append(
            {'name': col, 'mean': mean, 'median': median, 'std': std})

    return statistics_list


def missing_percentage():
    """
    Calculates percentage of missing data for each column
    in finaldata.csv

    Returns:
        list[dict]: Each dict contains column name and percentage
    """
    logging.info("Loading and preparing finaldata.csv")
    data_df = pd.read_csv(os.path.join(DATA_PATH, 'finaldata.csv'))
    data_df = data_df.drop(['corporation', 'exited'], axis=1)

    logging.info("Calculating missing data percentage")
    missing_list = [{'name': col, 'percentage': perc} for col, perc in zip(
        data_df.columns, data_df.isna().sum() / data_df.shape[0] * 100)]

    return missing_list


def _ingestion_timing():
    """
    Runs ingestion.py script and measures execution time

    Returns:
        float: running time
    """
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    timing = timeit.default_timer() - starttime
    return timing


def _training_timing():
    """
    Runs training.py script and measures execution time

    Returns:
        float: running time
    """
    starttime = timeit.default_timer()
    os.system('python training.py')
    timing = timeit.default_timer() - starttime
    return timing


def execution_time():
    """
    Gets average execution time for data ingestion and model training
    by running each 25 times

    Returns:
        list[dict]: mean of execution times for each script
    """
    logging.info("Calculating time for ingestion.py")
    ingestion_time = []
    for _ in range(20):
        time = _ingestion_timing()
        ingestion_time.append(time)
    logging.info(f"Mean time: {np.mean(ingestion_time)}")

    logging.info("Calculating time for training.py")
    training_time = []
    for _ in range(20):
        time = _training_timing()
        training_time.append(time)
    logging.info(f"Mean time: {np.mean(training_time)}")

    ret_list = [
        {'ingest_time_mean': np.mean(ingestion_time)},
        {'train_time_mean': np.mean(training_time)}
    ]

    return ret_list


def outdated_packages_list():
    """
    Check dependencies status from requirements.txt file using pip-outdated
    which checks each package status if it is outdated or not

    Returns:
        str: stdout of the pip-outdated command
    """
    logging.info("Checking outdated dependencies")
    dependencies = subprocess.run(
        'pip-outdated ../requirements.txt',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8')

    return dependencies.stdout


if __name__ == '__main__':

    logging.info("Loading and preparing testdata.csv")
    test_df = pd.read_csv(os.path.join(TEST_DATA_PATH, 'testdata.csv'))
    X_df = test_df.drop(['corporation', 'exited'], axis=1)

    print("Model predictions on testdata.csv:",
          model_predictions(X_df), end='\n\n')

    print("Summary statistics")
    print(json.dumps(dataframe_summary(), indent=4), end='\n\n')

    print("Missing percentage")
    print(json.dumps(missing_percentage(), indent=4), end='\n\n')

    print("Execution time")
    print(json.dumps(execution_time(), indent=4), end='\n\n')

    print(outdated_packages_list())
