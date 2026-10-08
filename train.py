"""Train a leakage-safe first BikeOps model on UCI hourly rental data."""
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
CATEGORICAL = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
NUMERIC = ['temp', 'hum', 'windspeed']


def features(frame):
    # An allowlist prevents target-derived columns from becoming inputs.
    return frame[CATEGORICAL + NUMERIC].copy()


def split_by_date(frame):
    ordered = frame.copy()
    ordered['timestamp'] = pd.to_datetime(ordered['dteday']) + pd.to_timedelta(ordered['hr'], unit='h')
    ordered = ordered.sort_values('timestamp').reset_index(drop=True)
    partitions = (
        ordered[ordered['timestamp'] < '2012-07-01'].copy(),
        ordered[(ordered['timestamp'] >= '2012-07-01') & (ordered['timestamp'] < '2012-10-01')].copy(),
        ordered[ordered['timestamp'] >= '2012-10-01'].copy(),
    )
    if any(part.empty for part in partitions):
        raise ValueError('Dataset must cover training, validation, and test date ranges.')
    return partitions


def main():
    source = ROOT / 'data/raw/hour.csv'
    frame = pd.read_csv(source)
    if frame.isna().any().any():
        raise ValueError('Dataset contains missing values.')
    if not (frame['cnt'] == frame['casual'] + frame['registered']).all():
        raise ValueError('Rental count integrity check failed.')
    if frame.duplicated(['dteday', 'hr']).any():
        raise ValueError('Duplicate hourly records.')
    train, validation, test = split_by_date(frame)
    models = {'mean_baseline': DummyRegressor(strategy='mean')}
    for leaf in [3, 10]:
        preprocessing = ColumnTransformer([
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
            ('numeric', 'passthrough', NUMERIC),
        ])
        models[f'random_forest_leaf_{leaf}'] = Pipeline([
            ('preprocessing', preprocessing),
            ('regressor', RandomForestRegressor(n_estimators=150, max_depth=18,
                                               min_samples_leaf=leaf, random_state=42, n_jobs=2)),
        ])
    validation_scores = {}
    training_seconds = {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(features(train), train['cnt'])
        training_seconds[name] = round(time.perf_counter() - start, 3)
        prediction = model.predict(features(validation))
        validation_scores[name] = float(mean_absolute_error(validation['cnt'], prediction))
    selected = min(validation_scores, key=validation_scores.get)
    test_metrics = {}
    # Evaluate the selected candidate and the baseline only after selection.
    for name in dict.fromkeys(['mean_baseline', selected]):
        prediction = models[name].predict(features(test))
        test_metrics[name] = {
            'mae': float(mean_absolute_error(test['cnt'], prediction)),
            'rmse': float(np.sqrt(mean_squared_error(test['cnt'], prediction))),
            'r2': float(r2_score(test['cnt'], prediction)),
        }
    output = ROOT / 'artifacts'
    output.mkdir(exist_ok=True)
    model_path = output / 'model.joblib'
    joblib.dump(models[selected], model_path)
    restored = joblib.load(model_path)
    sample_features = features(test.head(5))
    np.testing.assert_allclose(restored.predict(sample_features), models[selected].predict(sample_features))
    sample = test[['timestamp', 'cnt']].head(5).copy()
    sample['predicted_rentals'] = np.round(restored.predict(sample_features), 1)
    sample.to_csv(output / 'sample_predictions.csv', index=False)
    split_info = {
        name: {'rows': len(part), 'start': str(part['timestamp'].min()), 'end': str(part['timestamp'].max())}
        for name, part in [('train', train), ('validation', validation), ('test', test)]
    }
    report = {
        'dataset': {'source': 'https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset',
                    'license': 'CC BY 4.0', 'rows': len(frame),
                    'sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
        'target': 'cnt', 'features': CATEGORICAL + NUMERIC,
        'excluded': ['casual', 'registered', 'cnt', 'instant', 'dteday', 'yr', 'atemp'],
        'splits': split_info, 'seed': 42,
        'selected_model': selected, 'validation_mae': validation_scores,
        'training_seconds': training_seconds, 'test_metrics': test_metrics,
        'versions': {'python': platform.python_version(), 'pandas': pd.__version__,
                     'numpy': np.__version__, 'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__},
        'limitations': ['Historical system-wide demand, not station availability.',
                       'Uses observed weather: not a validated future-weather forecast.',
                       'Missing hours are not filled; no live data or OCI deployment yet.',
                       'Selected artifact remains fitted on training partition only.'],
    }
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    baseline_mae = test_metrics['mean_baseline']['mae']
    candidate_mae = test_metrics[selected]['mae']
    improvement = 100 * (baseline_mae - candidate_mae) / baseline_mae
    lines = [
        '# BikeOps: first public-data training run', '',
        'Predict total hourly rentals across Capital Bikeshare using calendar and weather inputs.', '',
        '## Data and evaluation', '',
        f'- Real public data: {len(frame):,} hourly records from 2011–2012.',
        '- Training precedes July 2012; validation is July–September; final test is October–December.',
        '- Model selection uses validation MAE. Final test data is not used to fit or select models.',
        '- Target components `casual` and `registered` are excluded.', '',
        '| Model | Validation MAE | Test MAE | Test RMSE | Test R² |',
        '|---|---:|---:|---:|---:|',
    ]
    for name in dict.fromkeys(['mean_baseline', selected]):
        m = test_metrics[name]
        lines.append(f'| {name} | {validation_scores[name]:.2f} | {m["mae"]:.2f} | {m["rmse"]:.2f} | {m["r2"]:.3f} |')
    lines += ['', f'The selected model reduces test MAE by **{improvement:.1f}%** versus the constant-mean baseline.',
              'MAE is the average absolute error in rentals per hour. This is a simple baseline comparison, not a claim of state-of-the-art performance.', '',
              '## Example predictions', '', '| Historical hour | Actual rentals | Predicted rentals |', '|---|---:|---:|']
    for _, row in sample.iterrows():
        lines.append(f'| {row["timestamp"]} | {row["cnt"]} | {row["predicted_rentals"]} |')
    lines += ['', '## Limitations', ''] + ['- ' + item for item in report['limitations']]
    lines += ['', '## Attribution', '',
              'Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894. CC BY 4.0.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'selected_model': selected, 'splits': split_info, 'validation_mae': validation_scores,
                      'test_metrics': test_metrics, 'mae_improvement_percent': improvement}, indent=2))


if __name__ == '__main__':
    main()
