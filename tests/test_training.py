import unittest
import pandas as pd
from train import features, split_by_date


class TrainingIntegrityTests(unittest.TestCase):
    def test_features_exclude_target_components_and_identifiers(self):
        frame = pd.DataFrame({
            'season': [1], 'mnth': [1], 'hr': [8], 'holiday': [0],
            'weekday': [1], 'workingday': [1], 'weathersit': [1],
            'temp': [.3], 'hum': [.6], 'windspeed': [.2],
            'cnt': [100], 'casual': [20], 'registered': [80],
            'instant': [1], 'dteday': ['2011-01-01'], 'yr': [0],
        })
        result = features(frame)
        self.assertTrue({'cnt', 'casual', 'registered', 'instant', 'dteday', 'yr'}.isdisjoint(result.columns))
        self.assertEqual(result.loc[0, 'hr'], 8)

    def test_date_boundaries_and_order_are_kept_separate(self):
        frame = pd.DataFrame({
            'dteday': ['2012-10-01', '2012-07-01', '2011-01-01', '2012-06-30', '2012-09-30'],
            'hr': [0, 0, 0, 23, 23],
            'cnt': [10, 20, 30, 40, 50],
        })
        train, validation, test = split_by_date(frame)
        self.assertEqual(train['cnt'].tolist(), [30, 40])
        self.assertEqual(validation['cnt'].tolist(), [20, 50])
        self.assertEqual(test['cnt'].tolist(), [10])
        self.assertEqual(len(train) + len(validation) + len(test), len(frame))

    def test_empty_evaluation_partition_is_rejected(self):
        frame = pd.DataFrame({'dteday': ['2011-01-01'], 'hr': [1], 'cnt': [5]})
        with self.assertRaises(ValueError):
            split_by_date(frame)


if __name__ == '__main__':
    unittest.main()
