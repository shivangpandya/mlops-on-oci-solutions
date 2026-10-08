"""Export one numeric MLflow model version as a verified container-ready checkpoint."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import mlflow
from mlflow import MlflowClient
import yaml
from evaluate_reviews import DEFAULT_TRACKING_URI, REGISTERED_NAME
from review_inference import ROOT, SentimentModel, verify_checkpoint, file_sha256


def verify_export(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'release.json').read_text())
    if type(metadata['model_version']) is not int or metadata['model_version'] < 1:
        raise ValueError('Release requires a positive numeric model version.')
    manifest = verify_checkpoint(directory / 'checkpoint')
    if metadata['model_revision'] != manifest['revision'] or metadata['registered_name'] != REGISTERED_NAME:
        raise ValueError('Exported model identity mismatch.')
    actual = {str(p.relative_to(directory)): file_sha256(p)
              for p in (directory / 'checkpoint').iterdir() if p.is_file()}
    if actual != metadata['sha256']:
        raise ValueError('Exported file checksums mismatch.')
    return metadata


def export_registered_model(tracking_uri, version, output_dir):
    if type(version) is not int or version < 1:
        raise ValueError('Use a positive numeric registry version, not an alias.')
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Release destination already exists: {output_dir}')
    client = MlflowClient(tracking_uri=tracking_uri)
    registered = client.get_model_version(REGISTERED_NAME, version)
    mlflow.set_tracking_uri(tracking_uri)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reviewops-export-', dir=output_dir.parent) as folder:
        staging = Path(folder) / 'release'
        staging.mkdir()
        package = Path(mlflow.artifacts.download_artifacts(
            artifact_uri=f'models:/{REGISTERED_NAME}/{version}', dst_path=str(Path(folder) / 'download')))
        config = yaml.safe_load((package / 'MLmodel').read_text())
        relative = config['flavors']['python_function']['artifacts']['checkpoint']['path']
        checkpoint = (package / relative).resolve()
        if not checkpoint.is_relative_to(package.resolve()):
            raise ValueError('Checkpoint path escapes the model package.')
        manifest = verify_checkpoint(checkpoint)
        if manifest['revision'] != registered.tags.get('model_revision'):
            raise ValueError('Registry revision differs from packaged checkpoint.')
        target = staging / 'checkpoint'
        target.mkdir()
        for name in ['manifest.json', *manifest['sha256']]:
            shutil.copy2(checkpoint / name, target / name)
        model = SentimentModel(target)
        probe = model.predict('A wonderful movie.')
        metadata = {'registered_name': REGISTERED_NAME, 'model_version': version,
                    'model_uri': f'models:/{REGISTERED_NAME}/{version}', 'run_id': registered.run_id,
                    'model_id': manifest['model_id'], 'model_revision': manifest['revision'],
                    'sample_sha256': registered.tags.get('sample_sha256'),
                    'sha256': {str(p.relative_to(staging)): file_sha256(p) for p in target.iterdir() if p.is_file()},
                    'verification_prediction': {k: probe[k] for k in ['label', 'score', 'model_version']}}
        (staging / 'release.json').write_text(json.dumps(metadata, indent=2) + '\n')
        verify_export(staging)
        # Destination becomes visible only once all files and inference pass verification.
        staging.rename(output_dir)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracking-uri', default=DEFAULT_TRACKING_URI)
    parser.add_argument('--version', required=True, type=int)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export_registered_model(args.tracking_uri, args.version, args.output_dir), indent=2))

if __name__ == '__main__':
    main()
