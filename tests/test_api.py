import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from bike_app import create_app, feature_frame, PredictionInput
from train import ROOT, features


class PredictionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = {
            'date': '2012-10-01', 'hour': 0, 'holiday': False,
            'temperature_c': 19.68, 'humidity_pct': 94,
            'windspeed_kmh': 0, 'weather': 'mist',
        }
        cls.context = TestClient(create_app())
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_prediction_matches_saved_model_on_real_heldout_record(self):
        frame = pd.read_csv(ROOT / 'data/raw/hour.csv')
        row = frame[(frame.dteday == '2012-10-01') & (frame.hr == 0)].iloc[0]
        payload = dict(self.payload, temperature_c=float(row.temp * 41),
                       humidity_pct=float(row.hum * 100), windspeed_kmh=float(row.windspeed * 67),
                       weather={1: 'clear', 2: 'mist', 3: 'light_precipitation', 4: 'severe_weather'}[int(row.weathersit)])
        expected = float(joblib.load(ROOT / 'artifacts/model.joblib').predict(features(frame.loc[[row.name]]))[0])
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertAlmostEqual(result['predicted_rentals'], expected, places=6)
        self.assertRegex(result['model_version'], r'^sha256:[0-9a-f]{64}$')

    def test_liveness_and_readiness(self):
        self.assertEqual(self.client.get('/health').status_code, 200)
        self.assertEqual(self.client.get('/ready').status_code, 200)

    def test_invalid_and_extra_inputs_rejected(self):
        for changes in [{'hour': 24}, {'humidity_pct': -1}, {'temperature_c': 42},
                        {'windspeed_kmh': 68}, {'weather': 'hurricane'}, {'date': 'bad'},
                        {'hour': 1.5}, {'registered': 80}, {'temperature_c': 'NaN'}]:
            with self.subTest(changes=changes):
                self.assertEqual(self.client.post('/predict', json=dict(self.payload, **changes)).status_code, 422)

    def test_calendar_boundaries_and_unit_conversions(self):
        for date, season in [('2012-03-20', 1), ('2012-03-21', 2),
                             ('2012-06-21', 3), ('2012-09-23', 4), ('2012-12-21', 1)]:
            with self.subTest(date=date):
                inputs = PredictionInput(**dict(self.payload, date=date, temperature_c=20.5,
                                               humidity_pct=50, windspeed_kmh=33.5))
                row = feature_frame(inputs).iloc[0]
                self.assertEqual(row['season'], season)
                for column in ['temp', 'hum', 'windspeed']:
                    self.assertEqual(row[column], .5)
        sunday = feature_frame(PredictionInput(**dict(self.payload, date='2012-09-23'))).iloc[0]
        self.assertEqual(sunday['weekday'], 0)
        self.assertEqual(sunday['workingday'], 0)
        holiday = feature_frame(PredictionInput(**dict(self.payload, holiday=True))).iloc[0]
        self.assertEqual(holiday['workingday'], 0)
        self.assertEqual(holiday['holiday'], 1)

    def test_browser_form_is_available(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.headers['content-type'])

    def test_missing_or_corrupt_model_is_unready_and_cannot_predict(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / 'corrupt.joblib'
            corrupt.write_bytes(b'not a model')
            for path in [Path(directory) / 'missing.joblib', corrupt]:
                with self.subTest(path=path), self.assertLogs('bike_app', level='WARNING'), TestClient(create_app(path)) as client:
                    self.assertEqual(client.get('/health').status_code, 200)
                    self.assertEqual(client.get('/ready').status_code, 503)
                    self.assertEqual(client.post('/predict', json=self.payload).status_code, 503)


if __name__ == '__main__':
    unittest.main()
