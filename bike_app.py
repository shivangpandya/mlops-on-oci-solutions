"""Serve the saved BikeOps model and a small browser learning interface."""
from contextlib import asynccontextmanager
from datetime import date as Date
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from train import ROOT, CATEGORICAL, NUMERIC

logger = logging.getLogger(__name__)
WEATHER = {'clear': 1, 'mist': 2, 'light_precipitation': 3, 'severe_weather': 4}


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    date: Date
    hour: int = Field(ge=0, le=23, strict=True)
    holiday: bool = Field(strict=True)
    temperature_c: float = Field(ge=0, le=41)
    humidity_pct: float = Field(ge=0, le=100)
    windspeed_kmh: float = Field(ge=0, le=67)
    weather: Literal['clear', 'mist', 'light_precipitation', 'severe_weather']


def feature_frame(inputs: PredictionInput) -> pd.DataFrame:
    month_day = (inputs.date.month, inputs.date.day)
    # These fixed boundaries match the source CSV's season transitions.
    season = 1
    if (3, 21) <= month_day < (6, 21):
        season = 2
    elif (6, 21) <= month_day < (9, 23):
        season = 3
    elif (9, 23) <= month_day < (12, 21):
        season = 4
    record = {
        'season': season, 'mnth': inputs.date.month, 'hr': inputs.hour,
        'holiday': int(inputs.holiday), 'weekday': (inputs.date.weekday() + 1) % 7,
        'workingday': int(inputs.date.weekday() < 5 and not inputs.holiday),
        'weathersit': WEATHER[inputs.weather], 'temp': inputs.temperature_c / 41,
        'hum': inputs.humidity_pct / 100, 'windspeed': inputs.windspeed_kmh / 67,
    }
    return pd.DataFrame([record], columns=CATEGORICAL + NUMERIC)


def create_app(model_path=None):
    path = Path(model_path or os.environ.get('BIKEOPS_MODEL_PATH', ROOT / 'artifacts/model.joblib'))

    @asynccontextmanager
    async def lifespan(application):
        application.state.model = None
        application.state.model_version = None
        try:
            # Hash and deserialize the same bytes so the version identifies this artifact.
            import io
            artifact = path.read_bytes()
            model = joblib.load(io.BytesIO(artifact))
            if list(model.feature_names_in_) != CATEGORICAL + NUMERIC:
                raise ValueError('Model feature schema differs from serving schema.')
            probe = PredictionInput(date='2012-10-01', hour=8, holiday=False,
                                    temperature_c=20, humidity_pct=60, windspeed_kmh=10, weather='clear')
            result = float(model.predict(feature_frame(probe))[0])
            if not math.isfinite(result) or result < 0:
                raise ValueError('Model returned an invalid readiness prediction.')
            application.state.model = model
            application.state.model_version = 'sha256:' + hashlib.sha256(artifact).hexdigest()
        except Exception:
            logger.warning('BikeOps model could not be loaded; predictions are unavailable.', exc_info=True)
        yield
        application.state.model = None

    application = FastAPI(title='BikeOps prediction API', version='0.1.0', lifespan=lifespan)
    application.mount('/static', StaticFiles(directory=ROOT / 'bike_static'), name='static')

    @application.get('/', include_in_schema=False)
    def index():
        return FileResponse(ROOT / 'bike_static/index.html')

    @application.get('/health')
    def health():
        return {'status': 'alive'}

    @application.get('/ready')
    def ready():
        if application.state.model is None:
            raise HTTPException(503, 'Model is unavailable. Train or configure a valid model and restart the service.')
        return {'status': 'ready', 'model_version': application.state.model_version}

    @application.post('/predict')
    def predict(inputs: PredictionInput):
        ready()
        frame = feature_frame(inputs)
        value = float(application.state.model.predict(frame)[0])
        if not math.isfinite(value) or value < 0:
            raise HTTPException(503, 'Model returned an invalid prediction.')
        return {'predicted_rentals': value, 'model_version': application.state.model_version,
                'model_inputs': frame.iloc[0].to_dict()}

    return application


app = create_app()
