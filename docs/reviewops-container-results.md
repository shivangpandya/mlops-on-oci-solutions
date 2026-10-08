# ReviewOps: verified AMD64 deployment image

The local results below describe the initial container milestone. Update: the user subsequently completed OCIR publication, OKE application smoke checks, and KServe predictor checks. The new frontend release is pending publication and rollout; see [its guide](reviewops-kserve-frontend.md).

## AMD64 OKE target — verified 2026-10-06

The OKE deployment target is `linux/amd64`. The AMD64 image was built and tested under emulation on this Arm64 Mac; native OKE performance, sizing, image pulls, and rollout remain unverified.

| Evidence | Value |
|---|---|
| Image | `localhost/reviewops:mlflow-v2-amd64` |
| Platform | Linux AMD64 (`x86_64` at runtime) |
| Image ID | `4b7299acf129b13a38e63718364596330657421085d39eae41bf54b1811e0881` |
| Local manifest digest | `sha256:8f1120f22011ab47b53836c59c84fd50c4f93128f79f86757419504eb465a0bb` |
| Uncompressed size | 1,533,748,683 bytes |
| Local AMD64 demo | http://127.0.0.1:8004 |

Live positive/negative predictions, input limits, UI, health and readiness passed. Real inference also passed with networking disabled, a read-only filesystem, UID 10001, and CPU-only PyTorch. MLflow is absent from the serving image. Missing-model checks returned health 200 and readiness/prediction 503; that temporary container was removed.

All **37 regression tests passed**. The AMD64 renderer regression was observed failing against the old platform guard, then passed after implementation. The default Dockerfile base, renderer CLI platform, and Kubernetes node selector now target AMD64. Arm64 remains available for local testing.

Evidence: `artifacts/container/image-amd64.json`, `smoke-amd64.json`, `offline-amd64.json`, `missing-model-amd64.json`, `installed-serving-amd64.json`, and `build-amd64.log`. Verified build inputs are in `dist/reviewops-v2-amd64/build-manifest.json`. No OCIR publication or OKE deployment occurred. Use the digest from the actual OCIR push when rendering the deployment.

## Earlier Arm64 local verification

| Evidence | Value |
|---|---|
| Local image | `localhost/reviewops:mlflow-v2-arm64` |
| Platform | Linux Arm64 |
| Image ID | `de9e96ba0921164aa5fc10596452546842354112475652d17f65827bd9d00f58` |
| Local manifest digest | `sha256:61f8fe4d275d17f69b46c3353cb50e4e67d3f378f9b82ed0f56cbb0d1bd9b0ae` |
| Image size | 1,432,509,408 bytes; about 1.43 GB uncompressed |
| Runtime user | UID/GID 10001 |
| Model | Exported MLflow version 2, run `a42731d0210a442a8b6d687f17d998c9` |
| Model revision | `714eb0fa89d2f80546fda750413ed43d93601a13` |
| PyTorch | `2.14.1+cpu`; no CUDA |
| Live demo | http://127.0.0.1:8002 |

Image size describes storage, not working memory or registry transfer size. The local manifest digest is not claimed as a published OCIR digest; publication may serialize/compress the manifest differently. Record the digest returned by the actual push.

## Verified behavior

- The exact exported checkpoint and its checksums are preserved; preparation uses an allowlist.
- Positive and negative reviews produce the expected labels and scores matching prior model verification.
- Blank, non-string, oversized, extra-field, and over-token inputs return HTTP 422.
- The application runs as non-root with a read-only root filesystem and temporary writable `/tmp`.
- Networking-disabled execution performs real CPU inference successfully.
- A missing-model test returns health 200, readiness 503, and prediction 503. The temporary negative-test container was removed.
- MLflow is not installed in the serving image. Dataset and local registry files are omitted.
- Base Kubernetes manifests render locally; the digest/context/release renderer rejects invalid input and unsupported architectures.

The full regression suite passed 36 tests after final-review fixes; project and image-build dependency checks passed. A fresh independent review checked the package hashes, recorded image identity, configuration, tests, and instructions. Its separate-terminal context finding was corrected; placeholder-context validation was hardened with a failing regression test, then verified green. No cloud tests were performed.

## Inspectable files

Local evidence under `artifacts/container/` includes:

- `image-arm64.json`: image configuration, labels, digest, platform, size.
- `installed-serving-arm64.json`: installed runtime packages.
- `smoke-arm64.json`: real live API check outputs.
- `offline-arm64.json`: networking-disabled CPU prediction, UID, and CUDA status.
- `missing-model.json`: readiness failure verification.
- `build-arm64.log`: actual Linux build and dependency-check output.

The prepared source/checkpoint hashes are in `dist/reviewops-v2-arm64/build-manifest.json`. Deployment artifacts are `k8s/base/` and `render_oke_release.py`; the base contains a placeholder and must not be applied directly.

## Next cloud handoff

Use [the beginner runbook](reviewops-oke-runbook.md). Intended context: `context-cef-oke-liftoff-ch4yy43wxtq`; OCI profile: `axon`. Confirm the OCIR repository, target cluster connectivity/permissions, and suitable AMD64 worker capacity. Publish the separately verified AMD64 image, not the local Arm64 image.

Registry publication and OKE apply are user-run commands under the current OCI skill boundary. No resources in OCI were created, updated, or deleted by this local stage. KServe follows the verified regular Deployment/Service baseline.

Review decisions and remaining limits are retained in `.superpowers/reviewops-container/progress.md`: Arm64 was tested natively and AMD64 under emulation (see `amd64-progress.md`), publisher confirmation is required for digest contents, and this one-replica demo does not establish production availability, capacity, or stronger artifact authenticity. No Git repository exists, so preparation hashes and retained logs provide the local provenance record.
