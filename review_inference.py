"""Shared, offline CPU inference for ReviewOps."""
import hashlib
import json
import math
from pathlib import Path
import re
import time

MODEL_ID = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / 'artifacts/review_model'

class InvalidReviewError(ValueError):
    """Input does not satisfy the serving contract."""

class ModelUnavailableError(RuntimeError):
    """Checkpoint verification or inference failed."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(directory):
    directory = Path(directory)
    try:
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['model_id'] != MODEL_ID or not re.fullmatch('[0-9a-f]{40}', manifest['revision']):
            raise ValueError('Unexpected model identity.')
        required = {'config.json', 'model.safetensors', 'tokenizer_config.json', 'vocab.txt'}
        if not required.issubset(manifest['sha256']):
            raise ValueError('Incomplete model manifest.')
        for name, expected in manifest['sha256'].items():
            if Path(name).name != name or name in {'.', '..'}:
                raise ValueError('Invalid manifest filename.')
            if file_sha256(directory / name) != expected:
                raise ValueError(f'Checkpoint checksum mismatch: {name}')
        return manifest
    except Exception as exc:
        raise ModelUnavailableError(f'Checkpoint verification failed: {exc}') from exc


class SentimentModel:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR):
        self.directory = Path(model_dir)
        self.manifest = verify_checkpoint(self.directory)
        self.model_version = 'hf:' + self.manifest['revision']
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            torch.set_num_threads(2)
            self.tokenizer = AutoTokenizer.from_pretrained(self.directory, local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.directory, local_files_only=True, trust_remote_code=False, use_safetensors=True)
            self.model.to('cpu').eval()
            if self.model.config.id2label != {0: 'NEGATIVE', 1: 'POSITIVE'}:
                raise ValueError('Unexpected sentiment labels.')
            self.predict('A wonderful movie.')
        except Exception as exc:
            raise ModelUnavailableError(f'Model loading failed: {exc}') from exc

    def encode(self, text):
        if not isinstance(text, str):
            raise InvalidReviewError('Review must be a string.')
        text = text.strip()
        if not 1 <= len(text) <= 2000:
            raise InvalidReviewError('Review must contain 1–2,000 characters.')
        encoded = self.tokenizer(text, return_tensors='pt', truncation=False, verbose=False)
        if encoded['input_ids'].shape[1] > 512:
            raise InvalidReviewError('Review exceeds 512 tokens. Please shorten it; text is not silently truncated.')
        return encoded

    def predict(self, text):
        import torch
        start = time.perf_counter()
        encoded = self.encode(text)
        try:
            with torch.inference_mode():
                logits = self.model(**encoded).logits[0]
                if not torch.isfinite(logits).all():
                    raise ValueError('Non-finite model logits.')
                score, index = torch.max(torch.softmax(logits, dim=-1), dim=0)
            value = float(score.item())
            if not math.isfinite(value):
                raise ValueError('Non-finite model score.')
        except Exception as exc:
            raise ModelUnavailableError(f'Inference failed: {exc}') from exc
        return {'label': self.model.config.id2label[int(index.item())], 'score': value,
                'model_id': MODEL_ID, 'model_version': self.model_version,
                'token_count': encoded['input_ids'].shape[1],
                'inference_ms': round((time.perf_counter() - start) * 1000, 2)}
