"""Auditable MLflow model code; weights are separate verified artifacts."""
import pandas as pd
import mlflow
from review_inference import SentimentModel, InvalidReviewError

class ReviewPythonModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = SentimentModel(context.artifacts['checkpoint'])

    def predict(self, context, model_input, params=None):
        if not isinstance(model_input, pd.DataFrame) or list(model_input.columns) != ['text']:
            raise InvalidReviewError('Expected a DataFrame with one column: text.')
        if params:
            raise InvalidReviewError('Inference parameters are not supported.')
        return pd.DataFrame([self.model.predict(text) for text in model_input['text']])

mlflow.models.set_model(ReviewPythonModel())
