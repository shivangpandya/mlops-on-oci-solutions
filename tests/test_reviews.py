import tempfile
import unittest
from fastapi.testclient import TestClient
from review_app import create_app


class ReviewApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(create_app())
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_real_pretrained_model_positive_and_negative(self):
        for text, label in [('The movie was wonderful. I loved it.', 'POSITIVE'),
                            ('A boring movie with terrible acting.', 'NEGATIVE')]:
            with self.subTest(text=text):
                response = self.client.post('/predict', json={'text': text})
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertEqual(result['label'], label)
                self.assertGreater(result['score'], .5)
                self.assertLessEqual(result['score'], 1)
                self.assertRegex(result['model_version'], r'^hf:[0-9a-f]{40}$')

    def test_invalid_requests_do_not_reach_model(self):
        for body in [{'text': ''}, {'text': '   '}, {'text': 12}, {'text': 'x' * 2001},
                     {'text': 'Good movie', 'unexpected': True}]:
            with self.subTest(body=str(body)[:60]):
                self.assertEqual(self.client.post('/predict', json=body).status_code, 422)

    def test_token_limit_rejects_instead_of_silently_truncating(self):
        self.assertEqual(self.client.post('/predict', json={'text': 'a ' * 600}).status_code, 422)

    def test_live_ready_and_browser(self):
        for route in ['/', '/health', '/ready']:
            self.assertEqual(self.client.get(route).status_code, 200)

    def test_missing_model_cannot_serve_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertLogs('review_app', level='WARNING'), TestClient(create_app(folder)) as client:
                self.assertEqual(client.get('/health').status_code, 200)
                self.assertEqual(client.get('/ready').status_code, 503)
                self.assertEqual(client.post('/predict', json={'text': 'Great movie'}).status_code, 503)
