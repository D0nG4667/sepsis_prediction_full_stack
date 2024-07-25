from pathlib import Path

# ENV when using standalone uvicorn server running FastAPI in api directory
ENV_PATH = Path('../../env/online.env')

ONE_DAY_SEC = 24*60*60

ONE_WEEK_SEC = ONE_DAY_SEC*7

PIPELINE_FUNCTION_URL = "https://raw.githubusercontent.com/D0nG4667/sepsis_prediction_full_stack/main/dev/models/pipeline_func/pipeline_functions.joblib"

RANDOM_FOREST_URL = "https://raw.githubusercontent.com/D0nG4667/sepsis_prediction_full_stack/main/dev/models/RandomForestClassifier.joblib"

XGBOOST_URL = "https://raw.githubusercontent.com/D0nG4667/sepsis_prediction_full_stack/main/dev/models/XGBClassifier.joblib"

ADABOOST_URL = "https://raw.githubusercontent.com/D0nG4667/sepsis_prediction_full_stack/main/dev/models/AdaBoostClassifier.joblib"

CATBOOST_URL = "https://raw.githubusercontent.com/D0nG4667/sepsis_prediction_full_stack/main/dev/models/CatBoostClassifier.joblib"

DECISION_TREE_URL = "https://raw.githubusercontent.com/D0nG4667/sepsis_prediction_full_stack/main/dev/models/DecisionTreeClassifier.joblib"

KNN_URL = "https://raw.githubusercontent.com/D0nG4667/sepsis_prediction_full_stack/main/dev/models/KNeighborsClassifier.joblib"

LGBM_URL = "https://raw.githubusercontent.com/D0nG4667/sepsis_prediction_full_stack/main/dev/models/LGBMClassifier.joblib"

LOG_REG_URL = "https://raw.githubusercontent.com/D0nG4667/sepsis_prediction_full_stack/main/dev/models/LogisticRegression.joblib"

SVC_URL = "https://raw.githubusercontent.com/D0nG4667/sepsis_prediction_full_stack/main/dev/models/SVC.joblib"

ENCODER_URL = "https://raw.githubusercontent.com/D0nG4667/sepsis_prediction_full_stack/main/dev/models/enc/encoder.joblib"

ALL_MODELS = {
    "AdaBoostClassifier": ADABOOST_URL,
    "CatBoostClassifier": CATBOOST_URL,
    "DecisionTreeClassifier": DECISION_TREE_URL,
    "KNeighborsClassifier": KNN_URL,
    "LGBMClassifier": LGBM_URL,
    "LogisticRegression": LOG_REG_URL,
    "RandomForestClassifier": RANDOM_FOREST_URL,
    "SupportVectorClassifier": SVC_URL,
    "XGBoostClassifier": XGBOOST_URL
}

DESCRIPTION = """
This API identifies ICU patients at risk of developing sepsis using `9 models` of which `Random Forest Classifier` and `XGBoost Classifier` are the best.\n

The models were trained on [The John Hopkins University datasets at Kaggle](https://www.kaggle.com/datasets/chaunguynnghunh/sepsis?select=README.md).\n

### Features
`PRG:` Plasma glucose\n
`PL:` Blood Work Result-1 (mu U/ml)\n
`PR:` Blood Pressure (mm Hg)\n
`SK:` Blood Work Result-2 (mm)\n
`TS:` Blood Work Result-3 (mu U/ml)\n
`M11:` Body mass index (weight in kg/(height in m)^2\n
`BD2:` Blood Work Result-4 (mu U/ml)\n
`Age:` patients age (years)\n
`Insurance:` If a patient holds a valid insurance card\n
 
### Results 
**Sepsis prediction:** *Positive* if a patient in ICU will develop a sepsis, and *Negative* otherwise\n

**Sepsis probability:** In percentage\n

### GraphQL API
To explore the GraphQL sub-application (built-with strawberry) to this RESTFul API click the link below.\n
🍓[GraphQL](/graphql)

### Let's Connect
👨‍⚕️ `Gabriel Okundaye`\n
[<img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" width="20" height="20">  LinkendIn](https://www.linkedin.com/in/dr-gabriel-okundaye)

[<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" width="20" height="20">  GitHub](https://github.com/D0nG4667/sepsis_prediction_full_stack)
 
"""
