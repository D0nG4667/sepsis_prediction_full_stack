import os
from dotenv import load_dotenv

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.coder import PickleCoder
from fastapi_cache.decorator import cache
import logging

from redis import asyncio as aioredis

from pydantic import BaseModel, Field
from typing import Tuple, Union, Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing._label import LabelEncoder
import joblib

import pandas as pd

import httpx
from io import BytesIO


from config import ONE_DAY_SEC, ONE_WEEK_SEC, XGBOOST_URL, RANDOM_FOREST_URL, ENCODER_URL, ENV_PATH, DESCRIPTION, ALL_MODELS

load_dotenv(ENV_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    url = os.getenv("REDIS_URL")
    username = os.getenv("REDIS_USERNAME")
    password = os.getenv("REDIS_PASSWORD")
    redis = aioredis.from_url(url=url, username=username,
                              password=password, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
    yield


# FastAPI Object
app = FastAPI(
    title='Sepsis classification',
    version='1.0.0',
    description=DESCRIPTION,
    lifespan=lifespan,
)

app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    file_name = "favicon.ico"
    file_path = os.path.join(app.root_path, "assets", file_name)
    return FileResponse(path=file_path, headers={"Content-Disposition": "attachment; filename=" + file_name})


# API input features

class SepsisFeatures(BaseModel):
    prg: int = Field(description="PRG: Plasma glucose")
    pl: int = Field(description="PL: Blood Work Result-1 (mu U/ml)")
    pr: int = Field(description="PR: Blood Pressure (mm Hg)")
    sk: int = Field(description="SK: Blood Work Result-2 (mm)")
    ts: int = Field(description="TS: Blood Work Result-3 (mu U/ml)")
    m11: float = Field(
        description="M11: Body mass index (weight in kg/(height in m)^2")
    bd2: float = Field(description="BD2: Blood Work Result-4 (mu U/ml)")
    age: int = Field(description="Age: patients age (years)")
    insurance: int = Field(
        description="Insurance: If a patient holds a valid insurance card")


class Url(BaseModel):
    url: str
    pipeline_url: str
    encoder_url: str


class ResultData(BaseModel):
    prediction: str
    probability: float


class PredictionResponse(BaseModel):
    execution_msg: str
    execution_code: int
    result: ResultData


class ErrorResponse(BaseModel):
    execution_msg: str
    execution_code: int
    error: Optional[str]


logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Load the model pipelines and encoder
# Cache for 1 day
@cache(expire=ONE_DAY_SEC, namespace='pipeline_resource', coder=PickleCoder)
async def load_pipeline(pipeline_url: Url, encoder_url: Url) -> Tuple[Pipeline, LabelEncoder]:
    async def url_to_data(url: Url):
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()  # Ensure we catch any HTTP errors
            # Convert response content to BytesIO object
            data = BytesIO(response.content)
            return data

    pipeline, encoder = None, None
    try:
        pipeline: Pipeline = joblib.load(await url_to_data(pipeline_url))
        encoder: LabelEncoder = joblib.load(await url_to_data(encoder_url))
    except Exception as e:
        logging.error(
            "Omg, an error occurred in loading the pipeline resources: %s", e)
    finally:
        return pipeline, encoder


# Endpoints

# Status endpoint: check if api is online
@app.get('/')
@cache(expire=ONE_WEEK_SEC, namespace='status_check')  # Cache for 1 week
async def status_check():
    return {"Status": "API is online..."}


@cache(expire=ONE_DAY_SEC, namespace='pipeline_classifier')  # Cache for 1 day
async def pipeline_classifier(pipeline: Pipeline, encoder: LabelEncoder, data: SepsisFeatures) -> Union[ErrorResponse, PredictionResponse]:
    msg = 'Execution failed'
    code = 0
    output = ErrorResponse(**{'execution_msg': msg,
                              'execution_code': code, 'error': None})

    try:
        # Create dataframe
        df = pd.DataFrame([data.model_dump()])

        # Make prediction
        prediction = pipeline.predict(df)

        pred_int = int(prediction[0])

        prediction = encoder.inverse_transform([pred_int])[0]

        # Get the probability of the predicted class
        probability = round(
            float(pipeline.predict_proba(df)[0][pred_int] * 100), 2)

        result = ResultData(**{"prediction": prediction,
                            "probability": probability})

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


# Random forest endpoint: classify sepsis with random forest
@app.post('/api/v1/random_forest/prediction', tags=['Random Forest'])
async def random_forest_classifier(data: SepsisFeatures) -> Union[ErrorResponse, PredictionResponse]:
    random_forest_pipeline, encoder = await load_pipeline(RANDOM_FOREST_URL, ENCODER_URL)
    output = await pipeline_classifier(random_forest_pipeline, encoder, data)
    return output


# Xgboost endpoint: classify sepsis with xgboost
@app.post('/api/v1/xgboost/prediction', tags=['XGBoost'])
async def xgboost_classifier(data: SepsisFeatures) -> Union[ErrorResponse, PredictionResponse]:
    xgboost_pipeline, encoder = await load_pipeline(XGBOOST_URL, ENCODER_URL)
    output = await pipeline_classifier(xgboost_pipeline, encoder, data)
    return output


@app.post('/api/v1/prediction', tags=['All Models'])
async def query_sepsis_prediction(data: SepsisFeatures, model: str = Query('RandomForestClassifier', enum=list(ALL_MODELS.keys()))) -> Union[ErrorResponse, PredictionResponse]:
    pipeline_url: Url = ALL_MODELS[model]
    pipeline, encoder = await load_pipeline(pipeline_url, ENCODER_URL)
    output = await pipeline_classifier(pipeline, encoder, data)
    return output
