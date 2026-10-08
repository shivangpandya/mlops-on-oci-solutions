import math
import unittest
from evaluate_reviews import compute_metrics, passes_gate

class ReviewEvaluationTests(unittest.TestCase):
    def test_known_metrics_and_confusion_order(self):
        result = compute_metrics(['NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE'],
                                 ['NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE'])
        self.assertEqual(result['accuracy'], .75)
        self.assertAlmostEqual(result['macro_f1'], (.6666666666666666 + .8) / 2)
        self.assertEqual(result['confusion_matrix'], [[1, 1], [0, 2]])
        self.assertEqual(result['labels'], ['NEGATIVE', 'POSITIVE'])
        self.assertEqual(result['per_class']['NEGATIVE']['recall'], .5)
        self.assertAlmostEqual(result['per_class']['POSITIVE']['precision'], 2/3)

    def test_threshold_and_failure_cases(self):
        labels = {'NEGATIVE', 'POSITIVE'}
        self.assertTrue(passes_gate({'accuracy': .8, 'macro_f1': .8}, labels, 0))
        for accuracy, f1, present, failures in [(.79, .8, labels, 0), (.8, .79, labels, 0),
            (.8, .8, {'POSITIVE'}, 0), (.8, .8, labels, 1), (math.nan, .8, labels, 0),
            (.8, math.inf, labels, 0)]:
            self.assertFalse(passes_gate({'accuracy': accuracy, 'macro_f1': f1}, present, failures))
        for value in [-1, 1.1, math.nan]:
            with self.assertRaises(ValueError):
                passes_gate({'accuracy': .8, 'macro_f1': .8}, labels, 0, min_accuracy=value)

    def test_empty_mismatched_or_unknown_labels_rejected(self):
        for truth, predicted in [([], []), (['POSITIVE'], []), (['NEUTRAL'], ['POSITIVE'])]:
            with self.assertRaises(ValueError):
                compute_metrics(truth, predicted)

class PreparationFailureTests(unittest.TestCase):
    def test_warmup_failure_is_reported_with_failed_gate(self):
        from unittest.mock import patch
        from review_inference import SentimentModel, DEFAULT_MODEL_DIR, ModelUnavailableError
        from evaluate_reviews import evaluate_sample
        model = SentimentModel(DEFAULT_MODEL_DIR)
        with patch.object(model, 'predict', side_effect=ModelUnavailableError('Non-finite logits')):
            report = evaluate_sample(model, [{'source_row_id': 0, 'text': 'Good.', 'label': 'POSITIVE'}],
                                     {'dataset_revision': 'fixture', 'sample_sha256': 'fixture'})
        self.assertFalse(report['gate']['passed'])
        self.assertEqual(report['inference_failure_count'], 1)
        self.assertEqual(report['inference_failures'][0]['phase'], 'warmup')
        self.assertIsNone(report['release'])
        self.assertIn('requirements.txt', report['requirements_sha256'])
        self.assertIn('requirements-mlops.txt', report['requirements_sha256'])
        self.assertIn('pyarrow', report['dependencies'])

    def test_startup_failure_creates_failed_mlflow_run(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from evaluate_reviews import main
        from review_inference import ModelUnavailableError, DEFAULT_MODEL_DIR
        from mlflow import MlflowClient
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'sample_manifest.json'
            manifest.write_text(json.dumps({'dataset_revision': 'fixture', 'sample_sha256': 'fixture'}))
            uri = 'sqlite:///' + str(root / 'tracking.db')
            args = ['--tracking-uri', uri, '--model-dir', str(DEFAULT_MODEL_DIR), '--manifest', str(manifest),
                    '--output-dir', str(root / 'out')]
            with patch('evaluate_reviews.SentimentModel', side_effect=ModelUnavailableError('Non-finite logits')):
                with patch('sys.argv', ['evaluate_reviews.py', *args]):
                    status = main()
            self.assertEqual(status, 1)
            report = json.loads((root / 'out/report.json').read_text())
            self.assertEqual(report['inference_failures'][0]['phase'], 'model_startup')
            self.assertFalse(report['gate']['passed'])
            self.assertIsNone(report['release'])
            client = MlflowClient(tracking_uri=uri)
            self.assertEqual(client.get_run(report['run_id']).data.tags['release_gate'], 'failed')
            self.assertEqual(client.get_run(report['run_id']).info.status, 'FAILED')
            self.assertEqual(list(client.search_registered_models()), [])
