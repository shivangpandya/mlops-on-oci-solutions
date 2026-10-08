from pathlib import Path
import unittest
from review_inference import SentimentModel
from acquire_review_data import select_sample, sample_bytes

ROOT = Path(__file__).resolve().parents[1]

class ReviewDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SentimentModel(ROOT / 'artifacts/review_model')

    def rows(self):
        return [{'source_row_id': i, 'text': f'A wonderful film {i}.' if i % 2 else f'A terrible film {i}.',
                 'label': i % 2} for i in range(240)]

    def test_balanced_repeatable_sample_and_coverage(self):
        rows = self.rows() + [{'source_row_id': 240, 'text': 'x' * 2001, 'label': 0},
                             {'source_row_id': 241, 'text': 'a ' * 600, 'label': 1},
                             {'source_row_id': 242, 'text': ' ', 'label': 0}]
        selected, metadata = select_sample(rows, self.model)
        again, repeated = select_sample(rows, self.model)
        self.assertEqual(sample_bytes(selected), sample_bytes(again))
        self.assertEqual(metadata, repeated)
        self.assertEqual(len(selected), 200)
        self.assertEqual(sum(r['label'] == 'POSITIVE' for r in selected), 100)
        self.assertEqual(metadata['source_rows'], 243)
        self.assertEqual(metadata['eligible_rows'], 240)
        self.assertEqual(metadata['excluded'], {'characters': 2, 'tokens': 1})
        self.assertEqual(len(set(r['source_row_id'] for r in selected)), 200)

    def test_invalid_labels_and_insufficient_data_fail(self):
        for label in [-1, 2, True, 'positive']:
            rows = self.rows()
            rows[0]['label'] = label
            with self.assertRaises(ValueError):
                select_sample(rows, self.model)
        with self.assertRaises(ValueError):
            select_sample(self.rows()[:20], self.model)
        rows = self.rows()
        rows[1]['source_row_id'] = rows[0]['source_row_id']
        with self.assertRaises(ValueError):
            select_sample(rows, self.model)
