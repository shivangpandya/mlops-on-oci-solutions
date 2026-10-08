"""Standard-library live smoke checks for a local ReviewOps container."""
import argparse
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request


def request(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base.rstrip('/') + path, data=data,
                                 headers={'Content-Type': 'application/json'} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def smoke(base, revision, expect_unready=False, expect_backend=None):
    deadline = time.monotonic() + 120
    target = '/health' if expect_unready else '/ready'
    while True:
        try:
            status, payload = request(base, target)
            if status == 200:
                break
        except urllib.error.URLError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError('Container did not become available within 120 seconds.')
        time.sleep(1)
    status, health_payload = request(base, '/health')
    assert status == 200
    if expect_backend:
        assert json.loads(health_payload).get('inference_backend') == expect_backend
    if expect_unready:
        assert request(base, '/ready')[0] == 503
        assert request(base, '/predict', {'text': 'A good movie.'})[0] == 503
        return {'base_url': base, 'missing_model_checks': 'passed'}
    ready = json.loads(payload)
    assert ready['model_version'] == 'hf:' + revision
    if expect_backend:
        assert ready.get('inference_backend') == expect_backend
    assert b'ReviewOps' in request(base, '/')[1]
    results = []
    for text, label in [('The movie was wonderful. I loved it.', 'POSITIVE'),
                        ('A boring movie with terrible acting.', 'NEGATIVE')]:
        status, payload = request(base, '/predict', {'text': text})
        assert status == 200
        result = json.loads(payload)
        assert result['label'] == label and result['model_version'] == 'hf:' + revision
        if expect_backend == 'kserve':
            assert result.get('inference_backend') == expect_backend
        assert math.isfinite(result['score']) and .5 < result['score'] <= 1
        results.append(result)
    for body in [{'text': ''}, {'text': ' '}, {'text': 12}, {'text': 'x' * 2001},
                 {'text': 'a ' * 600}, {'text': 'Good.', 'extra': True}]:
        assert request(base, '/predict', body)[0] == 422
    return {'base_url': base, 'status': 'passed', 'predictions': results,
            'checks': ['health', 'readiness', 'UI', 'positive', 'negative', 'input-limits', 'model-revision']
                      + (['inference-backend'] if expect_backend else [])}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8002')
    parser.add_argument('--revision', default='714eb0fa89d2f80546fda750413ed43d93601a13')
    parser.add_argument('--expect-backend', choices=['kserve', 'local'])
    parser.add_argument('--expect-unready', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = smoke(args.base_url, args.revision, args.expect_unready, args.expect_backend)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
