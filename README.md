# Dynamic Risk Assessment System
The fourth project for [ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) by Udacity.

## Description
This project is part of Unit 5: Machine Learning Model Scoring and Monitoring. The problem is to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's clients. Also setting up processes to re-train, re-deploy, monitor and report on the ML model.

## Prerequisites
- Python 3 required
- Linux environment may be needed within windows through WSL

## Dependencies
This project dependencies is available in the ```requirements.txt``` file.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies from the ```requirements.txt```. Its recommended to install it in a separate virtual environment.

```bash
pip install -r requirements.txt
```

## Project Structure
```bash
📦Dynamic-Risk-Assessment-System
 ┣
 ┣ 📂data
 ┃ ┣ 📂ingesteddata                 # Contains csv and metadata of the ingested data
 ┃ ┃ ┣ 📜finaldata.csv
 ┃ ┃ ┗ 📜ingestedfiles.txt
 ┃ ┣ 📂practicedata                 # Data used for practice mode initially
 ┃ ┃ ┣ 📜dataset1.csv
 ┃ ┃ ┗ 📜dataset2.csv
 ┃ ┣ 📂sourcedata                   # Data used for production mode
 ┃ ┃ ┣ 📜dataset3.csv
 ┃ ┃ ┗ 📜dataset4.csv
 ┃ ┗ 📂testdata                     # Test data
 ┃ ┃ ┗ 📜testdata.csv
 ┣ 📂model
 ┃ ┣ 📂models                       # Models pickle, score, and reports for production mode
 ┃ ┃ ┣ 📜apireturns.txt
 ┃ ┃ ┣ 📜confusionmatrix.png
 ┃ ┃ ┣ 📜latestscore.txt
 ┃ ┃ ┣ 📜summary_report.pdf
 ┃ ┃ ┗ 📜trainedmodel.pkl
 ┃ ┣ 📂practicemodels               # Models pickle, score, and reports for practice mode
 ┃ ┃ ┣ 📜apireturns.txt
 ┃ ┃ ┣ 📜confusionmatrix.png
 ┃ ┃ ┣ 📜latestscore.txt
 ┃ ┃ ┣ 📜summary_report.pdf
 ┃ ┃ ┗ 📜trainedmodel.pkl
 ┃ ┗ 📂production_deployment        # Deployed models and model metadata needed
 ┃ ┃ ┣ 📜ingestedfiles.txt
 ┃ ┃ ┣ 📜latestscore.txt
 ┃ ┃ ┗ 📜trainedmodel.pkl
 ┣ 📂src
 ┃ ┣ 📜apicalls.py                  # Runs app endpoints
 ┃ ┣ 📜app.py                       # Flask app
 ┃ ┣ 📜config.py                    # Config file for the project which depends on config.json
 ┃ ┣ 📜deployment.py                # Model deployment script
 ┃ ┣ 📜diagnostics.py               # Model diagnostics script
 ┃ ┣ 📜fullprocess.py               # Process automation
 ┃ ┣ 📜ingestion.py                 # Data ingestion script
 ┃ ┣ 📜pretty_confusion_matrix.py   # Plots confusion matrix
 ┃ ┣ 📜reporting.py                 # Generates confusion matrix and PDF report
 ┃ ┣ 📜scoring.py                   # Scores trained model
 ┃ ┣ 📜training.py                  # Model training
 ┃ ┗ 📜wsgi.py
 ┣ 📜config.json                    # Config json file
 ┣ 📜cronjob.txt                    # Holds cronjob created for automation
 ┣ 📜README.md
 ┗ 📜requirements.txt               # Projects required dependencies
```

## Steps Overview
1. **Data ingestion:** Automatically check if new data that can be used for model training. Compile all training data to a training dataset and save it to folder. 
2. **Training, scoring, and deploying:** Write scripts that train an ML model that predicts attrition risk, and score the model. Saves the model and the scoring metrics.
3. **Diagnostics:** Determine and save summary statistics related to a dataset. Time the performance of some functions. Check for dependency changes and package updates.
4. **Reporting:** Automatically generate plots and PDF document that report on model metrics and diagnostics. Provide an API endpoint that can return model predictions and metrics.
5. **Process Automation:** Create a script and cron job that automatically run all previous steps at regular intervals.

<img src="images/fullprocess.jpg" width=550 height=300>

## Usage

### 1- Edit config.json file to use practice data

```bash
"input_folder_path": "practicedata",
"output_folder_path": "ingesteddata", 
"test_data_path": "testdata", 
"output_model_path": "practicemodels", 
"prod_deployment_path": "production_deployment"
```

### 2- Run data ingestion
```python
cd src
python ingestion.py
```
Artifacts output:
```
data/ingesteddata/finaldata.csv
data/ingesteddata/ingestedfiles.txt
```

### 3- Model training
```python
python training.py
```
Artifacts output:
```
models/practicemodels/trainedmodel.pkl
```

###  4- Model scoring 
```python
python scoring.py
```
Artifacts output: 
```
models/practicemodels/latestscore.txt
``` 

### 5- Model deployment
```python
python deployment.py
```
Artifacts output:
```
models/prod_deployment_path/ingestedfiles.txt
models/prod_deployment_path/trainedmodel.pkl
models/prod_deployment_path/latestscore.txt
``` 

### 6- Run diagnostics
```python
python diagnostics.py
```

### 7- Run reporting
```python
python reporting.py
```
Artifacts output:
```
models/practicemodels/confusionmatrix.png
models/practicemodels/summary_report.pdf
```

### 8- Run Flask App
```python
python app.py
```

### 9- Run API endpoints
```python
python apicalls.py
```
Artifacts output:
```
models/practicemodels/apireturns.txt
```

### 11- Edit config.json file to use production data

```bash
"input_folder_path": "sourcedata",
"output_folder_path": "ingesteddata", 
"test_data_path": "testdata", 
"output_model_path": "models", 
"prod_deployment_path": "production_deployment"
```

### 10- Full process automation
```python
python fullprocess.py
```
### 11- Cron job

Start cron service
```bash
sudo service cron start
```

Edit crontab file
```bash
sudo crontab -e
```
   - Select **option 3** to edit file using vim text editor
   - Press **i** to insert a cron job
   - Write the cron job in ```cronjob.txt``` which runs ```fullprocces.py``` every 10 mins
   - Save after editing, press **esc key**, then type **:wq** and press enter
  
View crontab file
```bash
sudo crontab -l
```

## License
Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See ```LICENSE``` for more information.

## Resources

- Flask
  - https://pythonbasics.org/flask-http-methods/
  - https://www.sqlshack.com/create-rest-apis-in-python-using-flask/
  - https://medium.com/@shanakachathuranga/end-to-end-machine-learning-pipeline-with-mlops-tools-mlflow-dvc-flask-heroku-evidentlyai-github-c38b5233778c

- Reportlab
  - https://www.youtube.com/playlist?list=PLOGAj7tCqHx-IDg2x6cWzqN0um8Z4plQT
  - https://www.reportlab.com/docs/reportlab-userguide.pdf
