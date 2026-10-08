"""Bounded HTTP client for the verified internal ReviewOps predictor."""
import http.client
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from review_inference import MODEL_ID, InvalidReviewError, ModelUnavailableError

EXPECTED_MODEL_VERSION = 'hf:714eb0fa89d2f80546fda750413ed43d93601a13'
MAX_RESPONSE_BYTES = 32768

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

class RemoteSentimentModel:
    def __init__(self, base_url, timeout=5):
        parsed = urllib.parse.urlsplit(base_url)
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in {'', '/'} or parsed.query or parsed.fragment):
            raise ValueError('Predictor URL must be an HTTP(S) service origin without credentials, path, query, or fragment.')
        # Accessing port also validates a malformed/out-of-range port.
        parsed.port
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Predictor timeout must be positive and finite.')
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.model_version = EXPECTED_MODEL_VERSION
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def _request(self, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base_url+path, data=data,
                                         headers={'Accept':'application/json', 'Content-Type':'application/json'})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise ModelUnavailableError('Predictor unavailable.')
                raw = response.read(MAX_RESPONSE_BYTES+1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ModelUnavailableError('Predictor returned an oversized response.')
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise ModelUnavailableError('Predictor returned an invalid response.')
                return result
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            if code == 422 and payload is not None:
                raise InvalidReviewError('Review exceeds predictor input limits.') from None
            raise ModelUnavailableError('Predictor unavailable.') from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, RecursionError, http.client.HTTPException) as exc:
            raise ModelUnavailableError('Predictor unavailable or returned invalid JSON.') from exc

    def _verify_identity(self, result):
        if result.get('model_id') != MODEL_ID or result.get('model_version') != self.model_version:
            raise ModelUnavailableError('Predictor model identity differs from the verified release.')

    def ready(self):
        result = self._request('/ready')
        self._verify_identity(result)
        if result.get('status') != 'ready':
            raise ModelUnavailableError('Predictor is not ready.')
        return {'status':'ready', 'model_id':MODEL_ID, 'model_version':self.model_version,
                'inference_backend':'kserve'}

    def predict(self, text):
        result = self._request('/predict', {'text':text})
        self._verify_identity(result)
        score, tokens, elapsed = result.get('score'), result.get('token_count'), result.get('inference_ms')
        def numeric(value):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
            try:
                return math.isfinite(value)
            except OverflowError:
                return False
        if (result.get('label') not in ('POSITIVE', 'NEGATIVE') or not numeric(score) or not .5 <= score <= 1
                or type(tokens) is not int or not 1 <= tokens <= 512 or not numeric(elapsed) or elapsed < 0):
            raise ModelUnavailableError('Predictor returned an invalid prediction.')
        # Only forward the documented response fields, rather than arbitrary backend content.
        output = {key:result[key] for key in ['label','score','model_id','model_version','token_count','inference_ms']}
        output['inference_backend'] = 'kserve'
        return output
