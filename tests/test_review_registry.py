import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import mlflow
from mlflow import MlflowClient
import pandas as pd
from review_inference import SentimentModel, InvalidReviewError, DEFAULT_MODEL_DIR, file_sha256
from evaluate_reviews import evaluate_sample, record_run, REGISTERED_NAME

class ReviewRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SentimentModel(DEFAULT_MODEL_DIR)
        rows = [{'source_row_id': 0, 'text': 'A wonderful movie.', 'label': 'POSITIVE'},
                {'source_row_id': 1, 'text': 'A boring movie with terrible acting.', 'label': 'NEGATIVE'}]
        cls.report = evaluate_sample(cls.model, rows, {'sample_sha256': 'fixture', 'dataset_revision': 'fixture'})
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.uri = 'sqlite:///' + str(cls.root / 'tracking.db')
        cls.release = None

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_passing_registration_reload_and_validation(self):
        result = record_run(self.report, DEFAULT_MODEL_DIR, self.uri, self.root / 'passing')
        self.__class__.release = result
        self.assertTrue(result['gate']['passed'])
        version = str(result['release']['model_version'])
        client = MlflowClient(tracking_uri=self.uri)
        self.assertEqual(int(client.get_model_version_by_alias(REGISTERED_NAME, 'candidate').version), int(version))
        with patch('huggingface_hub.HfApi.model_info', side_effect=AssertionError('Network not allowed')):
            loaded = mlflow.pyfunc.load_model(f'models:/{REGISTERED_NAME}/{version}')
            result = loaded.predict(pd.DataFrame({'text': ['A wonderful movie.']})).iloc[0]
        expected = self.model.predict('A wonderful movie.')
        self.assertEqual(result['label'], expected['label'])
        self.assertAlmostEqual(result['score'], expected['score'], places=7)
        self.assertEqual(result['model_version'], expected['model_version'])
        for text in [' ', 'x' * 2001, 'a ' * 600, None, 123]:
            with self.subTest(text=str(text)[:15]), self.assertRaises(Exception):
                loaded.predict(pd.DataFrame({'text': [text]}))
        self.assertTrue((self.root / 'passing/report.json').exists())

    def test_failed_gate_does_not_register(self):
        uri = 'sqlite:///' + str(self.root / 'failed.db')
        report = json.loads(json.dumps(self.report))
        report['gate']['passed'] = False
        report['metrics']['accuracy'] = .5
        result = record_run(report, DEFAULT_MODEL_DIR, uri, self.root / 'failed')
        self.assertIsNone(result['release'])
        client = MlflowClient(tracking_uri=uri)
        self.assertEqual(list(client.search_registered_models()), [])
        self.assertEqual(client.get_run(result['run_id']).data.tags['release_gate'], 'failed')

    def test_changed_checkpoint_after_evaluation_is_not_registered(self):
        directory = self.root / 'changed-model'
        directory.mkdir()
        manifest = json.loads((DEFAULT_MODEL_DIR / 'manifest.json').read_text())
        for name in manifest['sha256']:
            shutil.copy2(DEFAULT_MODEL_DIR / name, directory / name)
        (directory / 'README.md').write_text('Changed after evaluation.')
        manifest['sha256']['README.md'] = file_sha256(directory / 'README.md')
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        uri = 'sqlite:///' + str(self.root / 'changed.db')
        with self.assertRaisesRegex(ValueError, 'changed since evaluation'):
            record_run(self.report, directory, uri, self.root / 'changed-run')
        self.assertEqual(list(MlflowClient(tracking_uri=uri).search_registered_models()), [])

    def test_registry_failure_never_writes_success(self):
        uri = 'sqlite:///' + str(self.root / 'broken.db')
        with patch('mlflow.register_model', side_effect=RuntimeError('Registry unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'Registry unavailable'):
                record_run(self.report, DEFAULT_MODEL_DIR, uri, self.root / 'broken')
        saved = json.loads((self.root / 'broken/report.json').read_text())
        self.assertIsNone(saved['release'])
        self.assertIn('Registry unavailable', saved['release_error'])

if __name__ == '__main__':
    unittest.main()
