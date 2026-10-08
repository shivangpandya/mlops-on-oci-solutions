# ReviewOps Container and OKE Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Execution status:** Local Linux Arm64 image built and tested; OKE templates and runbook prepared and reviewed. Publication, real digest rendering, and cloud rollout await OCIR destination and worker architecture verification.

**Goal:** Package the verified MLflow version 2 export as a tested Linux container and prepare traceable deployment to an explicitly selected existing OKE cluster.

**Architecture:** FastAPI still serves the form and performs shared DistilBERT inference. Build only from the verified exported checkpoint, run without upstream downloads, and attach registry/run/revision metadata to the image. Prepare ordinary Kubernetes Deployment/Service manifests first; KServe follows this verified baseline.

**Tech Stack:** Python 3.12, CPU PyTorch/Transformers, FastAPI, Podman-compatible Dockerfile, Kubernetes manifests/Kustomize, OCIR, existing OKE.

**Spec:** [Approved overall design](../../reviewops-mlflow-oke-design.md).

## Global Constraints

- User-selected OCI profile: `axon`. Prefix OCI terminal sessions with `export OCI_CLI_PROFILE=axon`; use `--profile axon` for explicit OCI CLI examples. Do not use DEFAULT.

- Use `artifacts/releases/v2`, numeric registry version 2, run `a42731d0210a442a8b6d687f17d998c9`, and verified checkpoint revision `714eb0fa89d2f80546fda750413ed43d93601a13`.
- Retain strict input limits, checksum verification, `/health`, `/ready`, `/predict`, and the existing UI.
- Do not include MLflow server/database, evaluation data, credentials, caches, virtual environments, or BikeOps artifacts in the runtime image.
- Keep CPU-only serving requirements separate from development/evaluation requirements. Verify actual Python 3.12 Linux wheel compatibility; macOS environment success does not prove Linux success.
- Use non-root execution and port 8000, offline Hugging Face/Transformers settings, read-only root filesystem, and writable `/tmp`.
- Cloud images must match OKE worker architecture and use a verified immutable digest. Podman currently reports Linux Arm64; local native testing does not prove AMD64 compatibility. Test AMD64 separately if that is the selected OKE platform; if emulation is unavailable, report that verification limit explicitly.
- No new cluster, load balancer, or public endpoint in this stage. Use a dedicated namespace and port-forward for first access.
- Do not select a configured OKE context merely because it is current. Destination context, OCIR repository, worker architecture, and access must be identified.
- The OCI code-generator skill restricts OCI commands to generation/user execution. Prepare reviewable publication/deployment commands; do not execute cloud changes without an explicit instruction overriding that boundary.
- No Git repository exists; retain source hashes and execution evidence locally.

## Review Focus

- Exported files are tampered with or metadata does not match checkpoint: refuse build preparation.
- Build context accidentally includes a token, raw data, or local database: allowlist source/artifact inputs and verify contents.
- Linux packages or image architecture are incompatible: fail build/test; never claim deployment readiness.
- Model initialization fails: readiness must stay unavailable, and the pod must receive no Service traffic.
- Destination/image is a placeholder, tag-only reference, or wrong architecture: reject cloud rendering before issuing apply instructions.

## Files and responsibilities

| File | Purpose |
|---|---|
| `prepare_container.py` | Verify the exported release and assemble an allowlisted build context. |
| `requirements-serving.txt` | Pin the direct runtime dependencies without MLflow/evaluation tools. |
| `Dockerfile` | Install Linux CPU dependencies and run the application as non-root. |
| `.dockerignore` | Exclude unintended build inputs. |
| `scripts/smoke_container.py` | Verify live readiness, predictions, validation, and packaged identity. |
| `k8s/base/` | Namespace, Deployment, ClusterIP Service, and Kustomize configuration. |
| `render_oke_release.py` | Validate context/image/platform inputs and render a digest-pinned deployment bundle. |
| `docs/reviewops-oke-runbook.md` | Beginner instructions, architecture diagram, verification, recovery, and rollback. |
| `OCI_REFERENCES.md` | Official sources and separation of user-run read-only/mutating commands. |
| `tests/test_container_release.py` | Preparation and deployment validation failures. |

## Task 1: Verified container packaging

**Interfaces:** `prepare_build_context(release_dir: Path, output_dir: Path) -> dict`, consuming the current `verify_export` interface. Output includes copied application/UI files, checkpoint, release metadata, dependency recipe, and preparation hashes.

- [x] Write tests that a real verified export prepares correctly, a changed file or wrong model identity is rejected, unrelated local files are absent, and existing output is not overwritten silently.
- [x] Run the new tests and observe expected failure before implementing preparation.
- [x] Implement allowlisted preparation, runtime dependency pins, and Dockerfile. Verify all copied release hashes. Resolve base-image digest and available Linux CPU wheels from official registries at implementation time; record the selected versions/platform.
- [x] Build with Podman on the available native platform. Run with non-root UID, read-only filesystem, temporary writable `/tmp`, and a separate loopback port such as 8002. Do not replace the current port-8000 demo.
- [x] Verify readiness, UI, positive/negative labels, blank/type/length/token rejection, no runtime upstream downloads, image labels, and release identity. Record image ID/digest/platform and measured size. Run the complete regression suite.

## Task 2: Validated OKE delivery bundle

**Interfaces:** `render_release(image_digest: str, context: str, platform: str, output_dir: Path) -> dict`. Cloud context/repository inputs come from the user, not ambient defaults. Output contains Kubernetes YAML and provenance; it does not execute apply.

- [x] Write tests rejecting empty/placeholder destinations, tag-only images, invalid digests, unsupported platforms, and inconsistent release metadata.
- [x] Observe expected failures, then implement the renderer and base manifests.
- [x] Use namespace `reviewops`, one replica, initial requests 1 CPU/1 GiB and limits 2 CPU/2 GiB. Include startup/readiness probes on `/ready`, liveness on `/health`, non-root security settings, disabled service-account token automount, no privilege escalation, and dropped capabilities. Use a registry pull-secret reference; include no token values.
- [ ] Cloud handoff pending: render with a real selected image digest when publication evidence is available. Use local Kustomize rendering and YAML validation; distinguish local checks from server-side dry-run and actual rollout. Do not send Kubernetes requests to a cloud context during local validation.
- [x] Run the full tests. Leave deployment blocked on destination/publication evidence rather than inventing it.

## Task 3: Beginner runbook and rollout verification

- [x] Explain what image, registry, node architecture, namespace, Deployment, Service, and probes mean, using the existing beginner guide and an updated diagram.
- [x] Generate destination-specific user-run instructions for authentication, OCIR publication, private image access, server-side validation, apply, rollout status, port-forward, prediction parity, and logs. Keep authentication tokens out of source, logs, and committed manifests.
- [x] Explain recovery by replacing a demo pod and rollback only when a prior verified release exists. One replica can interrupt requests; this is not a zero-downtime availability claim.
- [x] Add measured local image/test evidence and label OKE behavior unverified until actual rollout evidence exists. Document how the presenter checks model/run/image digest linkage.
- [x] Conduct one fresh read-only final review; address material findings with regression tests and rerun the suite. Update README/guide status accurately.

## Cloud destination and execution handoff

An asynchronous question has requested the intended OKE context and OCIR repository. The current kubeconfig context is `context-cef-oke-liftoff-ch4yy43wxtq`, with multiple alternatives configured; it has not been contacted. Podman 5.6.2 reports `arm64 linux`. No cloud discovery, image publication, or deployment has occurred.

Direct implementation remains the agreed execution method. The user approved this plan and the local implementation has been completed. If the cloud destination is not ready, Tasks 1 and local manifest/runbook preparation can proceed once the plan is approved; cloud actions remain a separate, explicit handoff.

## Official references checked

- OKE private registry pulls: https://docs.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengpullingimagesfromocir.htm
- OCIR image naming: https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryconcepts.htm
- OCI publication workflow: https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registryimagespushingpulling.htm
- Kubernetes probes: https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/
- PyTorch CPU installation: https://pytorch.org/get-started/locally/

Self-review: every file has an owning task; preparation output feeds deployment provenance; cloud image digest is deliberately unavailable until publication. All five failure conditions have tests or live smoke checks. KServe and cluster creation are outside this plan.
