import json
from pathlib import Path
import tempfile
import unittest
from evaluate_reviews import record_run, evaluate_sample, REGISTERED_NAME
from review_inference import SentimentModel, DEFAULT_MODEL_DIR, ModelUnavailableError
from export_review_model import export_registered_model, verify_export

class ReviewExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.uri = 'sqlite:///' + str(cls.root / 'export.db')
        cls.model = SentimentModel(DEFAULT_MODEL_DIR)
        rows = [{'source_row_id': 0, 'text': 'Wonderful movie.', 'label': 'POSITIVE'},
                {'source_row_id': 1, 'text': 'A boring movie with terrible acting.', 'label': 'NEGATIVE'}]
        report = evaluate_sample(cls.model, rows, {'dataset_revision': 'fixture', 'sample_sha256': 'fixture'})
        cls.result = record_run(report, DEFAULT_MODEL_DIR, cls.uri, cls.root / 'run')
        cls.version = cls.result['release']['model_version']

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_export_verifies_identity_and_prediction_parity(self):
        directory = self.root / 'valid-release'
        metadata = export_registered_model(self.uri, self.version, directory)
        self.assertEqual(metadata['model_version'], self.version)
        self.assertEqual(metadata['run_id'], self.result['run_id'])
        self.assertEqual(metadata['model_revision'], self.model.manifest['revision'])
        self.assertEqual(verify_export(directory), metadata)
        model = SentimentModel(directory / 'checkpoint')
        expected = self.model.predict('Wonderful movie.')
        actual = model.predict('Wonderful movie.')
        self.assertEqual(expected['label'], actual['label'])
        self.assertAlmostEqual(expected['score'], actual['score'], places=7)
        with self.assertRaises(FileExistsError):
            export_registered_model(self.uri, self.version, directory)

    def test_corrupt_export_rejected(self):
        directory = self.root / 'corrupt-release'
        export_registered_model(self.uri, self.version, directory)
        (directory / 'checkpoint/config.json').write_text('tampered')
        with self.assertRaises((ValueError, ModelUnavailableError)):
            verify_export(directory)

    def test_invalid_or_missing_version_has_no_partial_output(self):
        for i, version in enumerate(['candidate', '1', 0, True, 999]):
            directory = self.root / f'invalid-{i}'
            with self.assertRaises(Exception):
                export_registered_model(self.uri, version, directory)
            self.assertFalse(directory.exists())
