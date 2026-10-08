"""Evaluate pinned short IMDb reviews and register eligible local MLflow releases."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from collections import Counter
import sys
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from review_inference import ROOT, DEFAULT_MODEL_DIR, SentimentModel, ModelUnavailableError, MODEL_ID, file_sha256
from acquire_review_data import DEFAULT_DATA_DIR

LABELS = ['NEGATIVE', 'POSITIVE']
DEFAULT_TRACKING_URI = 'sqlite:///' + str(ROOT / 'artifacts/mlflow/reviewops.db')
REGISTERED_NAME = 'reviewops-distilbert'
EXPERIMENT_NAME = 'reviewops-evaluation'


def compute_metrics(labels, predictions):
    if not labels or len(labels) != len(predictions) or not set(labels + predictions).issubset(LABELS):
        raise ValueError('Metrics require nonempty, matching binary label sequences.')
    report = classification_report(labels, predictions, labels=LABELS, output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(labels, predictions)),
            'macro_f1': float(f1_score(labels, predictions, labels=LABELS, average='macro', zero_division=0)),
            'labels': LABELS, 'confusion_matrix': confusion_matrix(labels, predictions, labels=LABELS).tolist(),
            'per_class': {label: report[label] for label in LABELS},
            'error_count': sum(a != b for a, b in zip(labels, predictions)), 'evaluated_rows': len(labels)}


def passes_gate(metrics, labels_present, failure_count, min_accuracy=.80, min_macro_f1=.80):
    for threshold in (min_accuracy, min_macro_f1):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Gate thresholds must be finite values between 0 and 1.')
    accuracy, f1 = metrics.get('accuracy', float('nan')), metrics.get('macro_f1', float('nan'))
    return (labels_present == set(LABELS) and failure_count == 0
            and math.isfinite(accuracy) and math.isfinite(f1)
            and min_accuracy <= accuracy <= 1 and min_macro_f1 <= f1 <= 1)


def load_sample(dataset_path, manifest_path, model):
    manifest = json.loads(Path(manifest_path).read_text())
    if file_sha256(dataset_path) != manifest['sample_sha256']:
        raise ValueError('Dataset sample checksum mismatch.')
    if (manifest['model_revision'] != model.manifest['revision']
            or manifest['model_manifest_sha256'] != file_sha256(model.directory / 'manifest.json')):
        raise ValueError('Dataset eligibility was computed with different model files.')
    rows = [json.loads(line) for line in Path(dataset_path).read_text().splitlines()]
    counts = dict(Counter(row['label'] for row in rows))
    ids = [row['source_row_id'] for row in rows]
    if (len(rows) != 200 or counts != {'NEGATIVE': 100, 'POSITIVE': 100}
            or ids != manifest['source_row_ids'] or len(set(ids)) != 200):
        raise ValueError('Expected 200 unique selected reviews, 100 per class.')
    for row in rows:
        model.encode(row['text'])
    return rows, manifest


def evaluate_sample(model, rows, manifest, min_accuracy=.8, min_macro_f1=.8):
    # Validate threshold options before spending time on inference.
    passes_gate({}, set(), 0, min_accuracy, min_macro_f1)
    predictions, failures = [], []
    try:
        model.predict('A wonderful movie.')
    except Exception as exc:
        failures.append({'phase': 'warmup', 'error': str(exc)})
    for row in ([] if failures else rows):
        try:
            result = model.predict(row['text'])
            predictions.append({'source_row_id': row['source_row_id'], 'expected': row['label'], **result})
        except Exception as exc:
            failures.append({'source_row_id': row['source_row_id'], 'error': str(exc)})
    metrics = compute_metrics([r['expected'] for r in predictions], [r['label'] for r in predictions]) if predictions else {}
    timings = [r['inference_ms'] for r in predictions]
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'model_id': model.manifest['model_id'],
              'model_revision': model.manifest['revision'],
              'model_manifest_sha256': file_sha256(model.directory / 'manifest.json'),
              'dataset': manifest, 'metrics': metrics,
              'inference_failure_count': len(failures), 'inference_failures': failures,
              'latency_ms': {'p50': float(np.percentile(timings, 50)), 'p95': float(np.percentile(timings, 95))} if timings else {},
              'machine': {'platform': platform.platform(), 'processor': platform.machine(),
                          'python': platform.python_version(), 'device': 'cpu', 'torch_threads': 2},
              **environment_metadata(),
              'gate': {'min_accuracy': min_accuracy, 'min_macro_f1': min_macro_f1,
                       'teaching_thresholds': True,
                       'passed': passes_gate(metrics, {r['label'] for r in rows}, len(failures), min_accuracy, min_macro_f1)},
              'predictions': predictions, 'release': None}
    return report


def environment_metadata():
    return {
        'dependencies': {name: importlib.metadata.version(name) for name in
            ['torch', 'transformers', 'mlflow', 'pandas', 'scikit-learn', 'pyarrow',
             'huggingface-hub', 'numpy', 'safetensors']},
        'requirements_sha256': {p.name: file_sha256(p) for p in
            [ROOT / 'requirements.txt', ROOT / 'requirements-mlops.txt']},
        'installed_environment': dict(sorted((d.metadata['Name'], d.version)
            for d in importlib.metadata.distributions() if d.metadata['Name'])),
        'source_sha256': {p.name: file_sha256(p) for p in ROOT.glob('*.py')},
    }


def preparation_failure_report(model_dir, manifest_path, phase, error, min_accuracy, min_macro_f1):
    # Declared source identities are retained, but no successful verification is claimed.
    def read_manifest(path):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return {}
    checkpoint = read_manifest(Path(model_dir) / 'manifest.json')
    dataset = read_manifest(manifest_path)
    dataset.setdefault('dataset_revision', 'unavailable')
    dataset.setdefault('sample_sha256', 'unavailable')
    failures = [{'phase': phase, 'error': str(error)}]
    return {'created_at': datetime.now(timezone.utc).isoformat(), 'model_id': MODEL_ID,
            'model_revision': checkpoint.get('revision', 'unavailable'), 'model_verified': False,
            'dataset': dataset, 'metrics': {}, 'inference_failure_count': 1 if phase == 'model_startup' else 0,
            'inference_failures': failures if phase == 'model_startup' else [],
            'preparation_failures': failures, 'latency_ms': {}, 'predictions': [], 'release': None,
            'gate': {'min_accuracy': min_accuracy, 'min_macro_f1': min_macro_f1,
                     'teaching_thresholds': True, 'passed': False}, **environment_metadata()}


def record_run(report, model_dir, tracking_uri, output_dir):
    """Persist evaluation evidence; register only successful quality-gated runs."""
    import copy
    import shutil
    import tempfile
    import mlflow
    import pandas as pd
    from mlflow import MlflowClient
    from mlflow.models import ModelSignature
    from mlflow.types.schema import Schema, ColSpec
    from review_inference import verify_checkpoint
    report = copy.deepcopy(report)
    # Recompute the gate so a stale boolean cannot force a release.
    report['gate']['passed'] = passes_gate(report['metrics'],
        {r['expected'] for r in report['predictions']}, report['inference_failure_count'],
        report['gate']['min_accuracy'], report['gate']['min_macro_f1'])
    report['release'] = None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    def save():
        target = output_dir / 'report.json'
        temp = target.with_suffix('.json.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temp.replace(target)
    # Replace stale successful evidence even when the tracking store is unavailable.
    save()
    if tracking_uri.startswith('sqlite:///'):
        Path(tracking_uri.removeprefix('sqlite:///')).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME,
            artifact_location=(output_dir / 'mlflow-artifacts').resolve().as_uri())
    else:
        experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id) as run:
        report['run_id'] = run.info.run_id
        report['tracking_uri'] = tracking_uri
        mlflow.set_tags({'release_gate': 'passed' if report['gate']['passed'] else 'failed',
                         'model_revision': report['model_revision'], 'project': 'ReviewOps'})
        mlflow.log_params({'model_id': report['model_id'], 'model_revision': report['model_revision'],
                           'dataset_revision': report['dataset']['dataset_revision'],
                           'sample_sha256': report['dataset']['sample_sha256'],
                           'seed': report['dataset'].get('seed', 42),
                           'min_accuracy': report['gate']['min_accuracy'],
                           'min_macro_f1': report['gate']['min_macro_f1']})
        logged = {k: report['metrics'][k] for k in ['accuracy', 'macro_f1', 'error_count', 'evaluated_rows']
                  if k in report['metrics'] and math.isfinite(report['metrics'][k])}
        logged['inference_failure_count'] = report['inference_failure_count']
        logged.update({f'latency_{k}_ms': v for k, v in report['latency_ms'].items()})
        for label, metrics in report['metrics'].get('per_class', {}).items():
            logged.update({f'{label.lower()}_{k}': float(v) for k, v in metrics.items()})
        mlflow.log_metrics(logged)
        try:
            if report['gate']['passed']:
                manifest = verify_checkpoint(model_dir)
                if (manifest['revision'] != report['model_revision']
                        or file_sha256(Path(model_dir) / 'manifest.json') != report['model_manifest_sha256']):
                    raise ValueError('Model revision changed since evaluation.')
                with tempfile.TemporaryDirectory(dir=output_dir) as folder:
                    checkpoint = Path(folder) / 'checkpoint'
                    checkpoint.mkdir()
                    # No Hugging Face cache, credentials, or dataset in the model package.
                    for name in ['manifest.json', *manifest['sha256']]:
                        shutil.copy2(Path(model_dir) / name, checkpoint / name)
                    model = SentimentModel(checkpoint)
                    example = pd.DataFrame({'text': ['A wonderful movie.']})
                    requirements = [f'{name}=={importlib.metadata.version(name)}' for name in
                        ['mlflow', 'torch', 'transformers', 'pandas', 'safetensors', 'huggingface-hub']]
                    info = mlflow.pyfunc.log_model(name='sentiment-model',
                        python_model=str(ROOT / 'review_mlflow_model.py'),
                        code_paths=[str(ROOT / 'review_inference.py')],
                        artifacts={'checkpoint': str(checkpoint)},
                        signature=ModelSignature(inputs=Schema([ColSpec('string', 'text')]),
                            outputs=Schema([ColSpec('string', 'label'), ColSpec('double', 'score'),
                                ColSpec('string', 'model_id'), ColSpec('string', 'model_version'),
                                ColSpec('long', 'token_count'), ColSpec('double', 'inference_ms')])),
                        input_example=example,
                        pip_requirements=requirements,
                        metadata={'model_revision': manifest['revision'],
                                  'sample_sha256': report['dataset']['sample_sha256']})
                # Do not report success before both registry creation and alias assignment succeed.
                version = mlflow.register_model(info.model_uri, REGISTERED_NAME)
                client.set_model_version_tag(REGISTERED_NAME, version.version, 'model_revision', manifest['revision'])
                client.set_model_version_tag(REGISTERED_NAME, version.version, 'sample_sha256', report['dataset']['sample_sha256'])
                client.set_registered_model_alias(REGISTERED_NAME, 'candidate', version.version)
                report['release'] = {'registered_name': REGISTERED_NAME, 'model_version': int(version.version),
                                     'model_uri': f'models:/{REGISTERED_NAME}/{version.version}',
                                     'run_id': run.info.run_id, 'alias': 'candidate'}
            save()
            mlflow.log_dict(report, 'evaluation/report.json')
        except Exception as exc:
            report['release'] = None
            report['release_error'] = str(exc)
            mlflow.set_tag('release_status', 'failed')
            save()
            mlflow.log_dict(report, 'evaluation/report.json')
            raise
    if report['inference_failure_count'] or report.get('preparation_failures'):
        client.set_terminated(report['run_id'], status='FAILED')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATA_DIR / 'imdb_sample.jsonl')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_DATA_DIR / 'imdb_sample_manifest.json')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--tracking-uri', default=DEFAULT_TRACKING_URI)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/review_evaluation')
    parser.add_argument('--min-accuracy', type=float, default=.80)
    parser.add_argument('--min-macro-f1', type=float, default=.80)
    args = parser.parse_args()
    passes_gate({}, set(), 0, args.min_accuracy, args.min_macro_f1)
    try:
        model = SentimentModel(args.model_dir)
    except ModelUnavailableError as exc:
        report = preparation_failure_report(args.model_dir, args.manifest, 'model_startup', exc,
                                            args.min_accuracy, args.min_macro_f1)
    else:
        try:
            rows, manifest = load_sample(args.dataset, args.manifest, model)
        except (OSError, ValueError, KeyError) as exc:
            report = preparation_failure_report(args.model_dir, args.manifest, 'dataset_preparation', exc,
                                                args.min_accuracy, args.min_macro_f1)
        else:
            report = evaluate_sample(model, rows, manifest, args.min_accuracy, args.min_macro_f1)
    result = record_run(report, args.model_dir, args.tracking_uri, args.output_dir)
    print(json.dumps({k: result[k] for k in ['run_id', 'metrics', 'gate', 'release']}, indent=2))
    return 0 if result['gate']['passed'] else 1

if __name__ == '__main__':
    sys.exit(main())
