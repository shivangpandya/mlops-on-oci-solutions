"""ReviewOps UI/API with local inference or an explicitly configured KServe backend."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from review_inference import SentimentModel, InvalidReviewError, ModelUnavailableError, MODEL_ID

from review_remote import RemoteSentimentModel

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

class ReviewInput(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=2000, strict=True)

def create_app(model_dir=None, predictor_url=None):
    endpoint = os.environ.get('REVIEWOPS_PREDICTOR_URL', '') if predictor_url is None else predictor_url
    remote = RemoteSentimentModel(endpoint) if endpoint else None
    backend_name = 'kserve' if remote else 'local'
    directory = Path(model_dir or os.environ.get('REVIEWOPS_MODEL_DIR', ROOT / 'artifacts/review_model'))
    @asynccontextmanager
    async def lifespan(application):
        application.state.model = None
        try:
            application.state.model = remote if remote is not None else SentimentModel(directory)
        except ModelUnavailableError:
            logger.warning('ReviewOps model could not load; predictions are unavailable.', exc_info=True)
        yield
        application.state.model = None
    application = FastAPI(title='ReviewOps sentiment API', version='0.4.0', lifespan=lifespan)
    application.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
    @application.get('/', include_in_schema=False)
    def index():
        return FileResponse(ROOT / 'static/index.html')
    @application.get('/health')
    def health():
        return {'status': 'alive', 'inference_backend': backend_name}
    @application.get('/ready')
    def ready():
        if application.state.model is None:
            raise HTTPException(503, 'Model unavailable. Download or configure the checkpoint and restart the service.')
        if remote is not None:
            try:
                return remote.ready()
            except ModelUnavailableError:
                raise HTTPException(503, 'KServe predictor unavailable or model identity mismatch.')
        return {'status': 'ready', 'model_id': MODEL_ID, 'model_version': application.state.model.model_version,
                'inference_backend': backend_name}
    @application.post('/predict')
    def predict(inputs: ReviewInput):
        if application.state.model is None:
            raise HTTPException(503, 'Model unavailable.')
        try:
            return application.state.model.predict(inputs.text)
        except InvalidReviewError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ModelUnavailableError as exc:
            raise HTTPException(503, 'Predictor unavailable or returned an invalid result.') from exc
    return application

app = create_app()
