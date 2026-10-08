"""Regression checks at a real HTTP boundary, without loading model weights."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from review_app import create_app

VERSION = 'hf:714eb0fa89d2f80546fda750413ed43d93601a13'
PREDICTION = {'label': 'POSITIVE', 'score': .95,
              'model_id': 'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
              'model_version': VERSION, 'token_count': 6, 'inference_ms': 12.5}

@contextmanager
def predictor_server(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.respond('ready')
        def do_POST(self):
            state.setdefault('received', []).append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            self.respond('predict')
        def respond(self, route):
            time.sleep(state.get('delay', 0))
            value = state.get(route, {'status': 'ready', 'model_id': PREDICTION['model_id'], 'model_version': VERSION}
                              if route == 'ready' else PREDICTION)
            status = state.get('status', 200)
            data = value if isinstance(value, bytes) else json.dumps(value).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                if state.get('truncated_chunk'):
                    self.send_header('Transfer-Encoding', 'chunked')
                    self.end_headers()
                    self.wfile.write(b'20\r\nshort')
                    self.close_connection = True
                    return
                if status == 302:
                    self.send_header('Location', state['redirect'])
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

class RemoteFrontendTests(unittest.TestCase):
    def test_forwarding_preserves_input_identity_and_does_not_load_weights(self):
        state = {}
        with predictor_server(state) as url, patch('review_app.SentimentModel', side_effect=AssertionError('frontend loaded weights')):
            with TestClient(create_app(predictor_url=url)) as client:
                self.assertEqual(client.get('/health').json()['inference_backend'], 'kserve')
                self.assertEqual(client.get('/ready').json()['model_version'], VERSION)
                result = client.post('/predict', json={'text': '  Wonderful movie.  '})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()['label'], 'POSITIVE')
                self.assertEqual(result.json()['score'], .95)
                self.assertEqual(result.json()['inference_backend'], 'kserve')
                self.assertEqual(state['received'], [('/predict', {'text': 'Wonderful movie.'})])

    def test_invalid_frontend_inputs_never_reach_predictor(self):
        state = {}
        with predictor_server(state) as url, TestClient(create_app(predictor_url=url)) as client:
            for body in [{'text': ''}, {'text': 12}, {'text': 'x'*2001}, {'text': 'Hi.', 'extra': 1}]:
                self.assertEqual(client.post('/predict', json=body).status_code, 422)
        self.assertNotIn('received', state)

    def test_upstream_failures_return_503_without_local_fallback(self):
        state = {'status': 503, 'predict': {'detail': 'private backend error'}}
        with predictor_server(state) as url, patch('review_app.SentimentModel', side_effect=AssertionError('fallback')):
            with TestClient(create_app(predictor_url=url)) as client:
                self.assertEqual(client.get('/health').status_code, 200)
                self.assertEqual(client.get('/ready').status_code, 503)
                response = client.post('/predict', json={'text': 'Good movie.'})
                self.assertEqual(response.status_code, 503)
                self.assertNotIn('private', response.text)

    def test_upstream_token_limit_remains_422(self):
        with predictor_server({'status': 422}) as url, TestClient(create_app(predictor_url=url)) as client:
            self.assertEqual(client.post('/predict', json={'text': 'a '*600}).status_code, 422)

    def test_bad_payloads_and_model_mismatch_fail_closed(self):
        values = [b'not json', b'['*10000+b'0'+b']'*10000, {'label': 'MAYBE'}, dict(PREDICTION, score=float('nan')),
                  dict(PREDICTION, score=True), dict(PREDICTION, label=['POSITIVE']), dict(PREDICTION, model_version='hf:'+'0'*40),
                  dict(PREDICTION, model_id='other'), dict(PREDICTION, token_count=513),
                  dict(PREDICTION, inference_ms=-1), dict(PREDICTION, score=10**400),
                  dict(PREDICTION, inference_ms=10**400), b'x'*33000]
        for value in values:
            with self.subTest(value=str(value)[:40]), predictor_server({'predict': value}) as url:
                with TestClient(create_app(predictor_url=url)) as client:
                    self.assertEqual(client.post('/predict', json={'text': 'Good movie.'}).status_code, 503)

    def test_truncated_http_response_returns_503(self):
        with predictor_server({'truncated_chunk': True}) as url, TestClient(create_app(predictor_url=url), raise_server_exceptions=False) as client:
            self.assertEqual(client.get('/ready').status_code, 503)
            self.assertEqual(client.post('/predict', json={'text': 'Good movie.'}).status_code, 503)

    def test_readiness_rejects_wrong_model_revision(self):
        state={'ready': {'status':'ready', 'model_id': PREDICTION['model_id'], 'model_version':'hf:'+'0'*40}}
        with predictor_server(state) as url, TestClient(create_app(predictor_url=url)) as client:
            self.assertEqual(client.get('/ready').status_code, 503)

    def test_invalid_predictor_configuration_does_not_fall_back(self):
        for url in ['file:///etc/passwd', 'http://user:password@localhost', 'http://localhost/predict', 'http://localhost?x=1']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                create_app(predictor_url=url)

    def test_timeout_is_bounded_and_redirects_are_not_followed(self):
        from review_remote import RemoteSentimentModel
        from review_inference import ModelUnavailableError
        with predictor_server({'delay': .25}) as url:
            backend = RemoteSentimentModel(url, timeout=.05)
            started = time.monotonic()
            with self.assertRaises(ModelUnavailableError):
                backend.predict('Good movie.')
            self.assertLess(time.monotonic()-started, .5)
        destination={}
        with predictor_server(destination) as target, predictor_server({'status':302,'redirect':target+'/predict'}) as url:
            with self.assertRaises(ModelUnavailableError):
                RemoteSentimentModel(url).predict('Good movie.')
        self.assertNotIn('received', destination)

if __name__ == '__main__':
    unittest.main()
