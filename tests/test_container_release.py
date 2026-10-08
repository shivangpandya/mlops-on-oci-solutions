import json
from pathlib import Path
import tempfile
import unittest
from prepare_container import prepare_build_context

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'artifacts/releases/v2'

class ContainerPreparationTests(unittest.TestCase):
    def test_verified_context_only_contains_serving_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'build'
            metadata = prepare_build_context(RELEASE, out)
            self.assertEqual(metadata['release']['model_version'], 2)
            self.assertTrue((out / 'checkpoint/model.safetensors').exists())
            self.assertTrue((out / 'static/index.html').exists())
            self.assertEqual({p.name for p in out.iterdir()}, {'app.py','review_app.py','review_inference.py','review_remote.py',
                'static','checkpoint','release.json','requirements-serving.txt','Dockerfile','.dockerignore','build-manifest.json'})
            self.assertFalse(any(p.suffix in {'.db','.jsonl'} for p in out.rglob('*')))
            with self.assertRaises(FileExistsError):
                prepare_build_context(RELEASE, out)

    def test_inconsistent_or_tampered_release_is_refused(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            release = Path(folder) / 'release'
            shutil.copytree(RELEASE, release)
            metadata = json.loads((release / 'release.json').read_text())
            metadata['model_revision'] = '0' * 40
            (release / 'release.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                prepare_build_context(release, Path(folder) / 'bad')
            self.assertFalse((Path(folder) / 'bad').exists())
            metadata = json.loads((RELEASE / 'release.json').read_text())
            metadata['model_version'] = 1
            (release / 'release.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                prepare_build_context(release, Path(folder) / 'wrong-version')
            shutil.copy2(RELEASE / 'release.json', release / 'release.json')
            (release / 'checkpoint/config.json').write_text('tampered')
            with self.assertRaises(Exception):
                prepare_build_context(release, Path(folder) / 'bad')

class OkeRenderTests(unittest.TestCase):
    def test_digest_platform_and_identity_render(self):
        import yaml
        from render_oke_release import render_release
        image = 'iad.ocir.io/exampletenant/reviewops@sha256:' + 'a' * 64
        with tempfile.TemporaryDirectory() as folder:
            result = render_release(image, 'context-cef-oke-liftoff-ch4yy43wxtq', 'linux/arm64', Path(folder) / 'rendered')
            self.assertEqual(result['platform'], 'linux/arm64')
            documents = list(yaml.safe_load_all((Path(folder) / 'rendered/resources.yaml').read_text()))
            deployment = next(d for d in documents if d['kind'] == 'Deployment')
            pod = deployment['spec']['template']['spec']
            self.assertEqual(pod['nodeSelector']['kubernetes.io/arch'], 'arm64')
            self.assertEqual(pod['containers'][0]['image'], image)
            self.assertFalse(pod['automountServiceAccountToken'])
            self.assertTrue(pod['containers'][0]['securityContext']['readOnlyRootFilesystem'])
            self.assertEqual(pod['containers'][0]['readinessProbe']['httpGet']['path'], '/ready')

    def test_amd64_release_selects_amd64_workers(self):
        import yaml
        from render_oke_release import render_release
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'amd64'
            metadata = render_release('iad.ocir.io/exampletenant/reviewops@sha256:' + 'b' * 64,
                                      'context-cef-oke-liftoff-ch4yy43wxtq', 'linux/amd64', out)
            self.assertEqual(metadata['platform'], 'linux/amd64')
            deployment = next(d for d in yaml.safe_load_all((out / 'resources.yaml').read_text())
                              if d['kind'] == 'Deployment')
            self.assertEqual(deployment['spec']['template']['spec']['nodeSelector'],
                             {'kubernetes.io/arch': 'amd64'})

    def test_frontend_release_sets_image_and_backend_together(self):
        import yaml
        from render_oke_release import render_release
        image = 'iad.ocir.io/idtfszw3ugfy/reviewops@sha256:' + 'c'*64
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'frontend'
            result = render_release(image, 'context-cef-oke-liftoff-ch4yy43wxtq', 'linux/amd64', out,
                                    predictor_url='http://reviewops-sentiment-predictor.reviewops.svc.cluster.local')
            deployment = next(d for d in yaml.safe_load_all((out/'resources.yaml').read_text()) if d['kind']=='Deployment')
            container = deployment['spec']['template']['spec']['containers'][0]
            self.assertEqual(container['image'], image)
            self.assertIn({'name':'REVIEWOPS_PREDICTOR_URL',
                           'value':'http://reviewops-sentiment-predictor.reviewops.svc.cluster.local'}, container['env'])
            self.assertEqual(result['app_mode'], 'kserve-frontend')
            self.assertEqual(deployment['spec']['template']['metadata']['annotations']['reviewops/inference-backend'], 'kserve')
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            render_release(image, 'context', 'linux/amd64', Path(folder)/'bad', predictor_url='file:///etc/passwd')

    def test_invalid_cloud_destinations_and_unverified_platform_rejected(self):
        from render_oke_release import render_release
        valid = 'iad.ocir.io/exampletenant/reviewops@sha256:' + 'a' * 64
        for image, context, platform in [(valid, 'REPLACE_WITH_CONTEXT', 'linux/arm64'), (valid, '', 'linux/arm64'), ('repo:latest','context','linux/arm64'),
            ('<registry>/reviewops@sha256:'+'a'*64,'context','linux/arm64'),
            (valid,'context','linux/mips')]:
            with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
                render_release(image, context, platform, Path(folder) / 'out')


class AnnotationIntegrityTests(unittest.TestCase):
    def test_changed_deployment_identity_is_rejected(self):
        from unittest.mock import patch
        import yaml
        from render_oke_release import render_release
        documents = list(yaml.safe_load_all((ROOT / 'k8s/base/resources.yaml').read_text()))
        deployment = next(d for d in documents if d['kind'] == 'Deployment')
        deployment['spec']['template']['metadata']['annotations']['reviewops/mlflow-run'] = 'changed'
        with tempfile.TemporaryDirectory() as folder, patch('render_oke_release.yaml.safe_load_all', return_value=documents):
            with self.assertRaisesRegex(ValueError, 'annotations differ'):
                render_release('iad.ocir.io/test/reviewops@sha256:' + 'a' * 64, 'context',
                               'linux/arm64', Path(folder) / 'out')
