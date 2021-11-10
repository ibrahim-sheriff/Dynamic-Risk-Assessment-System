import os
import json
import pandas as pd
from datetime import datetime


# Load config.json and get input and output paths
with open('../config.json', 'r') as f:
    config = json.load(f)

input_folder_path = os.path.join(
    os.path.abspath('../'),
    'data',
    config['input_folder_path'])
output_folder_path = os.path.join(
    os.path.abspath('../'),
    'data',
    config['output_folder_path'])


# Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    df = pd.DataFrame()
    file_names = []

    for file in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, file)
        df_tmp = pd.read_csv(file_path)

        file_names.append(file)
        df = df.append(df_tmp, ignore_index=True)

    df = df.drop_duplicates().reset_index(drop=1)

    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as file:
        file.write(f"Ingestion date: {datetime.now()}\n")
        file.write("\n".join(file_names))

    df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)


if __name__ == '__main__':
    merge_multiple_dataframe()
