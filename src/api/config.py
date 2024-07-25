from pathlib import Path


# ENV when using standalone uvicorn server running FastAPI
ENV_PATH = Path('../../env/online.env')


DESCRIPTION = """
This API identifies ICU patients at risk of developing sepsis using XGBoost Classifier and Random Forest Classifier.
The models were trained on datasets from The John Hopkins University.
https://www.kaggle.com/datasets/chaunguynnghunh/sepsis?select=README.md

SEPSIS FEATURES
PRG: Plasma glucose
PL: Blood Work Result-1 (mu U/ml)
PR: Blood Pressure (mm Hg)
SK: Blood Work Result-2 (mm)
TS: Blood Work Result-3 (mu U/ml)
M11: Body mass index (weight in kg/(height in m)^2
BD2: Blood Work Result-4 (mu U/ml)
Age: patients age (years)
Insurance: If a patient holds a valid insurance card
 
EXAMPLE RESPONSE BODY- a JSON object:
{
    "execution_msg": "Execution was successful",
    "execution_code": 1,
    "result": {
    "prediction": "Negative",
    "probability": 79.18
    }
}

Sepsis prediction: 'Positive' if a patient in ICU will develop a sepsis, and 'Negative' otherwise
Sepsis probability: In percentage
 
"""
