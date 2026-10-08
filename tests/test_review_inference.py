import json
import math
from pathlib import Path
import tempfile
import unittest
from fastapi.testclient import TestClient
from review_app import create_app
from review_inference import SentimentModel, InvalidReviewError, ModelUnavailableError

ROOT = Path(__file__).resolve().parents[1]

class SharedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SentimentModel(ROOT / 'artifacts/review_model')

    def test_real_predictions_match_api(self):
        with TestClient(create_app()) as client:
            for text, label in [('The movie was wonderful. I loved it.', 'POSITIVE'),
                                ('A boring movie with terrible acting.', 'NEGATIVE')]:
                result = self.model.predict(text)
                response = client.post('/predict', json={'text': text}).json()
                self.assertEqual(result['label'], label)
                self.assertEqual(result['label'], response['label'])
                self.assertAlmostEqual(result['score'], response['score'], places=7)
                self.assertTrue(math.isfinite(result['score']))
                self.assertRegex(result['model_version'], r'^hf:[0-9a-f]{40}$')

    def test_rejects_invalid_inputs_consistently(self):
        for value in [None, 12, '', '  ', 'x' * 2001, 'a ' * 600]:
            with self.subTest(value=str(value)[:30]), self.assertRaises(InvalidReviewError):
                self.model.predict(value)
        self.assertEqual(self.model.predict('  Wonderful movie.  ')['label'],
                         self.model.predict('Wonderful movie.')['label'])

    def test_missing_and_corrupt_model_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ModelUnavailableError):
                SentimentModel(Path(folder))
            manifest = json.loads((ROOT / 'artifacts/review_model/manifest.json').read_text())
            (Path(folder) / 'manifest.json').write_text(json.dumps(manifest))
            (Path(folder) / 'config.json').write_text('corrupted')
            with self.assertRaises(ModelUnavailableError):
                SentimentModel(Path(folder))

if __name__ == '__main__':
    unittest.main()
