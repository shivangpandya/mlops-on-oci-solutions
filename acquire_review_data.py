"""Acquire a pinned public IMDb test sample; no training or fabricated data."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
from review_inference import SentimentModel, DEFAULT_MODEL_DIR, ROOT, InvalidReviewError, file_sha256

DATASET_ID = 'stanfordnlp/imdb'
SOURCE_FILE = 'plain_text/test-00000-of-00001.parquet'
DEFAULT_DATA_DIR = ROOT / 'data/reviews'


def sample_bytes(rows):
    return ''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')


def select_sample(rows, model, seed=42):
    eligible = {'NEGATIVE': [], 'POSITIVE': []}
    excluded = {'characters': 0, 'tokens': 0}
    ids = set()
    for row in rows:
        label = row['label']
        if type(label) is not int or label not in (0, 1):
            raise ValueError('IMDb labels must be integers 0 or 1.')
        row_id = row['source_row_id']
        if type(row_id) is not int or row_id < 0 or row_id in ids:
            raise ValueError('Source row IDs must be unique non-negative integers.')
        ids.add(row_id)
        if not isinstance(row['text'], str):
            raise ValueError('Dataset text must be a string.')
        text = row['text'].strip()
        if not 1 <= len(text) <= 2000:
            excluded['characters'] += 1
            continue
        try:
            model.encode(text)
        except InvalidReviewError:
            excluded['tokens'] += 1
            continue
        name = 'POSITIVE' if label else 'NEGATIVE'
        eligible[name].append({'source_row_id': row_id, 'text': text, 'label': name})
    rng = random.Random(seed)
    selected = []
    for name in ('NEGATIVE', 'POSITIVE'):
        if len(eligible[name]) < 100:
            raise ValueError(f'Need 100 eligible reviews for {name}; found {len(eligible[name])}.')
        selected.extend(rng.sample(eligible[name], 100))
    rng.shuffle(selected)
    metadata = {'source_rows': len(rows), 'eligible_rows': sum(map(len, eligible.values())),
                'eligible_by_label': {k: len(v) for k, v in eligible.items()}, 'excluded': excluded,
                'selected_rows': 200, 'selected_by_label': dict(Counter(r['label'] for r in selected)),
                'source_row_ids': [r['source_row_id'] for r in selected], 'seed': seed,
                'sample_sha256': hashlib.sha256(sample_bytes(selected)).hexdigest()}
    return selected, metadata


def acquire(output_dir=DEFAULT_DATA_DIR, model_dir=DEFAULT_MODEL_DIR, seed=42):
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'imdb_sample_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    revision = previous['dataset_revision'] if previous else HfApi(token=False).dataset_info(DATASET_ID).sha
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Dataset revision must be a pinned commit.')
    if previous and previous['dataset_id'] != DATASET_ID:
        raise ValueError('Unexpected previous dataset identity.')
    source = Path(hf_hub_download(DATASET_ID, SOURCE_FILE, repo_type='dataset', revision=revision,
                                  token=False, cache_dir=ROOT / '.hf-cache'))
    card = Path(hf_hub_download(DATASET_ID, 'README.md', repo_type='dataset', revision=revision,
                                token=False, cache_dir=ROOT / '.hf-cache'))
    frame = pd.read_parquet(source)
    rows = [dict(row, source_row_id=i) for i, row in enumerate(frame.to_dict('records'))]
    model = SentimentModel(model_dir)
    selected, metadata = select_sample(rows, model, seed)
    metadata.update({'dataset_id': DATASET_ID, 'dataset_revision': revision, 'split': 'test',
                     'source_files': {SOURCE_FILE: file_sha256(source), 'README.md': file_sha256(card)},
                     'model_id': model.manifest['model_id'], 'model_revision': model.manifest['revision'],
                     'model_manifest_sha256': file_sha256(Path(model_dir) / 'manifest.json'),
                     'eligibility': {'max_characters': 2000, 'max_tokens': 512, 'strip_whitespace': True},
                     'attribution_url': 'https://huggingface.co/datasets/stanfordnlp/imdb'})
    (directory / 'imdb_sample.jsonl').write_bytes(sample_bytes(selected))
    (directory / 'DATASET_CARD.md').write_bytes(card.read_bytes())
    manifest_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(acquire(args.output_dir, args.model_dir, args.seed), indent=2))

if __name__ == '__main__':
    main()
