"""Create an allowlisted, verified ReviewOps container build context."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from export_review_model import verify_export
from review_inference import ROOT, MODEL_ID, file_sha256


def prepare_build_context(release_dir, output_dir):
    release_dir, output_dir = Path(release_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    release = verify_export(release_dir)
    if (release['model_id'] != MODEL_ID or release['model_version'] != 2
            or release['run_id'] != 'a42731d0210a442a8b6d687f17d998c9'
            or release['model_revision'] != '714eb0fa89d2f80546fda750413ed43d93601a13'):
        raise ValueError('Missing or unsupported release identity.')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix='.reviewops-build-') as folder:
        staging = Path(folder) / 'context'
        staging.mkdir()
        for name in ['app.py', 'review_app.py', 'review_inference.py', 'review_remote.py', 'requirements-serving.txt', 'Dockerfile', '.dockerignore']:
            shutil.copy2(ROOT / name, staging / name)
        (staging / 'static').mkdir()
        for name in ['index.html', 'app.js', 'style.css']:
            shutil.copy2(ROOT / 'static' / name, staging / 'static' / name)
        (staging / 'checkpoint').mkdir()
        for relative in release['sha256']:
            source = release_dir / relative
            if source.is_symlink():
                raise ValueError('Checkpoint symlinks are not allowed in build context.')
            shutil.copy2(source, staging / relative)
        shutil.copy2(release_dir / 'release.json', staging / 'release.json')
        if verify_export(staging) != release:
            raise ValueError('Release changed while preparing build context.')
        metadata = {'release': release, 'sha256': {str(p.relative_to(staging)): file_sha256(p)
            for p in staging.rglob('*') if p.is_file()}, 'status': 'prepared-not-built'}
        (staging / 'build-manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
        staging.rename(output_dir)
    return metadata

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, default=ROOT / 'artifacts/releases/v2')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_build_context(args.release_dir, args.output_dir), indent=2))
