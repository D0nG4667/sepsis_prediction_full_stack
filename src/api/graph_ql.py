import strawberry
from strawberry.asgi import GraphQL

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing._label import LabelEncoder

import httpx
from io import BytesIO

from typing import Tuple, List, Optional, Union
from enum import Enum


from config import RANDOM_FOREST_URL, XGBOOST_URL, ENCODER_URL

import logging


# API input features

@strawberry.enum
class ModelChoice(Enum):
    RandomForestClassifier = RANDOM_FOREST_URL
    XGBoostClassifier = XGBOOST_URL


@strawberry.input
class SepsisFeatures:
    prg: List[int]
    pl: List[int]
    pr: List[int]
    sk: List[int]
    ts: List[int]
    m11: List[float]
    bd2: List[float]
    age: List[int]
    insurance: List[int]


@strawberry.type
class Url:
    url: str
    pipeline_url: str
    encoder_url: str


@strawberry.type
class ResultData:
    prediction: List[str]
    probability: List[float]


@strawberry.type
class PredictionResponse:
    execution_msg: str
    execution_code: int
    result: ResultData


@strawberry.type
class ErrorResponse:
    execution_msg: str
    execution_code: int
    error: Optional[str]


logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


async def url_to_data(url: Url) -> BytesIO:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()  # Ensure we catch any HTTP errors
        # Convert response content to BytesIO object
        data = BytesIO(response.content)
        return data


# Load the model pipelines and encoder
async def load_pipeline(pipeline_url: Url, encoder_url: Url) -> Tuple[Pipeline, LabelEncoder]:
    pipeline, encoder = None, None
    try:
        pipeline: Pipeline = joblib.load(await url_to_data(pipeline_url))
        encoder: LabelEncoder = joblib.load(await url_to_data(encoder_url))
    except Exception as e:
        logging.error(
            "Omg, an error occurred in loading the pipeline resources: %s", e)
    finally:
        return pipeline, encoder


async def pipeline_classifier(pipeline: Pipeline, encoder: LabelEncoder, data: SepsisFeatures) -> Union[ErrorResponse, PredictionResponse]:
    msg = 'Execution failed'
    code = 0
    output = ErrorResponse(**{'execution_msg': msg,
                              'execution_code': code, 'error': None})
    try:
        # Create dataframe
        df = pd.DataFrame.from_dict(data.__dict__)

        # Make prediction
        preds = pipeline.predict(df)
        preds_int = [int(pred) for pred in preds]

        predictions = encoder.inverse_transform(preds_int)
        probabilities_np = pipeline.predict_proba(df)

        probabilities = [round(float(max(prob)*100), 2)
                         for prob in probabilities_np]

        result = ResultData(**{"prediction": predictions,
                               "probability": probabilities}
                            )

        msg = 'Execution was successful'
        code = 1
        output = PredictionResponse(
            **{'execution_msg': msg,
               'execution_code': code, 'result': result}
        )

    except Exception as e:
        error = f"Omg, pipeline classifier and/or encoder failure. {e}"

        output = ErrorResponse(**{'execution_msg': msg,
                                  'execution_code': code, 'error': error})

    finally:
        return output


@strawberry.type
class Query:
    @strawberry.field
    async def predict_sepsis(self, model: ModelChoice, data: SepsisFeatures) -> Union[ErrorResponse, PredictionResponse]:
        pipeline_url: Url = model.value
        pipeline, encoder = await load_pipeline(pipeline_url, ENCODER_URL)

        output = await pipeline_classifier(pipeline, encoder, data)

        return output


# Create the GraphQL Schema
schema = strawberry.Schema(query=Query)

# Create the GraphQL application
graphql_app = GraphQL(schema)
