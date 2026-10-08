"""Download and pin the public sentiment checkpoint; no training required."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'


def main():
    directory = ROOT / 'artifacts/review_model'
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        revision = json.loads(manifest_path.read_text())['revision']
    else:
        revision = HfApi(token=False).model_info(MODEL_ID).sha
    snapshot_download(repo_id=MODEL_ID, revision=revision, token=False, local_dir=directory,
                      cache_dir=ROOT / '.hf-cache',
                      allow_patterns=['config.json', 'model.safetensors', 'tokenizer_config.json',
                                      'tokenizer.json', 'special_tokens_map.json', 'vocab.txt', 'README.md'])
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in directory.iterdir() if path.is_file() and path.name != 'manifest.json'}
    manifest = {'model_id': MODEL_ID, 'revision': revision, 'license': 'Apache-2.0', 'sha256': hashes}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (ROOT / 'artifacts/review_model_source.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'model_id': MODEL_ID, 'revision': revision, 'files': list(hashes)}, indent=2))


if __name__ == '__main__':
    main()
