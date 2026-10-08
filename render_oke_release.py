"""Render digest-pinned OKE manifests locally; never contacts Kubernetes/OCI."""
import argparse
import json
from pathlib import Path
import re
import tempfile
import yaml
from export_review_model import verify_export
from review_inference import ROOT
from review_remote import RemoteSentimentModel


def render_release(image_digest, context, platform, output_dir, predictor_url=None):
    if predictor_url is not None:
        if not predictor_url:
            raise ValueError('An explicit predictor service origin is required.')
        RemoteSentimentModel(predictor_url)  # Configuration validation only; no network call.
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', context)
            or any(word in context.lower() for word in ['replace_with', 'changeme', 'placeholder'])):
        raise ValueError('An explicit kubeconfig context is required.')
    if not re.fullmatch(r'[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9][a-z0-9/_.-]*@sha256:[0-9a-f]{64}', image_digest):
        raise ValueError('Provide a fully qualified registry image with an immutable SHA-256 digest.')
    if any(word in image_digest.lower() for word in ['replace_with','changeme','placeholder']):
        raise ValueError('Placeholder image is not allowed.')
    if platform not in {'linux/amd64', 'linux/arm64'}:
        raise ValueError('Only Linux AMD64 and Arm64 are supported; publish the matching tested image.')
    release = verify_export(ROOT / 'artifacts/releases/v2')
    documents = list(yaml.safe_load_all((ROOT / 'k8s/base/resources.yaml').read_text()))
    deployment = next(d for d in documents if d['kind'] == 'Deployment')
    template = deployment['spec']['template']
    expected = {'reviewops/model-version': str(release['model_version']),
                'reviewops/mlflow-run': release['run_id'], 'reviewops/model-revision': release['model_revision']}
    if template['metadata']['annotations'] != expected:
        raise ValueError('Deployment annotations differ from the verified release.')
    template['spec']['containers'][0]['image'] = image_digest
    template['spec']['nodeSelector']['kubernetes.io/arch'] = platform.split('/')[1]
    if predictor_url is not None:
        template['metadata']['annotations']['reviewops/inference-backend'] = 'kserve'
        template['spec']['containers'][0].setdefault('env', []).append(
            {'name':'REVIEWOPS_PREDICTOR_URL', 'value':predictor_url.rstrip('/')})
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'context': context, 'oci_profile': 'axon', 'image': image_digest, 'platform': platform,
                'app_mode': 'kserve-frontend' if predictor_url is not None else 'local-inference',
                'predictor_url': predictor_url, 'release': release, 'status': 'rendered-not-deployed',
                'registry_digest_confirmation': 'caller must confirm this digest is the published tested image'}
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix='.reviewops-oke-') as folder:
        staging = Path(folder) / 'bundle'
        staging.mkdir()
        (staging / 'resources.yaml').write_text(yaml.safe_dump_all(documents, sort_keys=False))
        (staging / 'release.json').write_text(json.dumps(metadata, indent=2) + '\n')
        staging.rename(output_dir)
    return metadata

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--context', required=True)
    parser.add_argument('--platform', default='linux/amd64')
    parser.add_argument('--predictor-url', help='Internal predictor service origin; enables frontend delegation.')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render_release(args.image, args.context, args.platform, args.output_dir, args.predictor_url), indent=2))
