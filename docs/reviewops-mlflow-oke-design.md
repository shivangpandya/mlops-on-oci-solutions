# ReviewOps: evaluation, registration, and OKE delivery

Status: evaluation, MLflow registration/export, AMD64 packaging, and the original OKE deployment are verified. The user reports the KServe controller/predictor Ready and predictor smoke passed. The FastAPI delegation release is locally implemented/tested; its cloud rollout is pending. See [the frontend handoff](reviewops-kserve-frontend.md).

Audience companion: [beginner architecture and presentation guide](reviewops-beginner-guide.md), with diagrams, plain-language component explanations, and a demonstration script.

## Purpose and success criteria

Extend the working beginner sentiment demo into a reproducible MLOps release story for KubeCon: acquire a public checkpoint, measure its behavior on public labeled data, register an eligible model, package the exact artifact, and serve it on OCI OKE. No initial training or GPU is required. Keep the existing form and API contract, including explicit rejection of oversized inputs.

Success means a real evaluation report and MLflow run, a reloadable registered model whose predictions match the API, a tested CPU container, and deployment manifests plus an OCI runbook. Cloud rollout requires an identified existing cluster, namespace, and registry destination. A manifest alone is not evidence of deployment.

## Selected approach and alternatives

Use local SQLite-backed MLflow with local artifacts first. This is the smallest reproducible setup for a single learner; it supports tracking and registry without adding a database service. A remote PostgreSQL-backed MLflow server and Object Storage artifacts are later upgrades. Deploying the whole MLflow platform to OKE immediately adds persistence and authentication work before the first release is understood.

Package the verified model into the application image. This makes startup independent of Hugging Face availability and ties rollback to a container digest. Fetching model artifacts from Object Storage at pod startup is a later extension for demonstrating workload identity; it requires additional IAM and bootstrap logic.

## Evaluation data and metrics

Acquire the public `stanfordnlp/imdb` test split at a recorded dataset revision. Select a deterministic sample of 200 eligible reviews, with 100 examples per label and a fixed seed. Eligibility uses the serving constraints: trimmed text between 1 and 2,000 characters and at most 512 tokens. Record total source rows, eligible rows, excluded rows by reason, selected source row IDs, dataset revision, and sample checksum. Download at execution time and preserve attribution; do not claim an unspecified dataset license or redistribute raw reviews in the container.

Report accuracy, macro F1, per-class precision/recall, a confusion matrix, error counts, and warmed single-request latency percentiles measured on the current machine. These results describe this short-review sample; they are not full IMDb test performance, production capacity, or a comparison of OCI with a laptop. The checkpoint was fine-tuned on SST-2; IMDb evaluation does not prove independence from all upstream pretraining material.

Log the checkpoint revision, manifest hash, dataset identity, selection parameters, sample hash, dependency versions, and machine description alongside metrics. Save local reports and a machine-readable release result. An unavailable dataset or malformed labels must fail clearly, without substituting hand-written examples.

## MLflow model and release gate

Extract model loading and prediction into one shared inference module used by FastAPI, evaluation, and a models-from-code MLflow PyFunc wrapper. Preserve checksum checks, CPU inference, label validation, token limits, and readiness behavior. Bundle the verified checkpoint and tokenizer as MLflow artifacts; downloading the registered model must support offline inference.

Create experiment `reviewops-evaluation` and register passing releases under `reviewops-distilbert`. Use a configurable demonstration gate with default accuracy >= 0.80, macro F1 >= 0.80, both labels represented, and zero inference failures on the selected sample. These are teaching thresholds, not validated business acceptance criteria. Failed runs remain inspectable but do not register or move the `candidate` alias. Registration errors must return failure and must not produce a successful release record. Passing runs create a numbered version and set `candidate`; promotion to `champion` is a separate explicit release action.

An export step resolves a numeric registry version, validates its files, and records its run ID/version and artifact hashes before the container build. The running API continues to expose the Hugging Face revision; release metadata also identifies the MLflow version. Serving must not resolve a moving alias on each request.

## Container and Kubernetes delivery

Add a minimal Python 3.12 Linux image using CPU-compatible, pinned serving dependencies, the static UI, application code, and exported checkpoint. Run as a non-root user; omit evaluation data, local MLflow database, caches, credentials, and BikeOps artifacts. Keep evaluation dependencies separate from serving dependencies. Select Linux AMD64 initially; document the need for a matching image platform when using Arm OKE nodes. Verify Linux package availability rather than copying macOS wheels.

Add Kubernetes namespace, Deployment, ClusterIP Service, and a local overlay with configurable image reference. Start with one replica and initial CPU/memory requests of 1 CPU/1 GiB, limits 2 CPU/2 GiB; these are estimates to validate under load. Use startup/readiness probes on `/ready`, liveness on `/health`, non-root security settings, disabled service-account token automount, and a read-only filesystem with explicit writable temporary storage. Use an OCIR pull secret for a private repository. Require an immutable image digest for cloud releases.

First access through `kubectl port-forward`; this avoids adding public exposure and load-balancer cost before the application is verified. The runbook covers image publication, pull-secret creation without committed tokens, rollout verification, prediction parity, and rollback. Live deployment will need the intended kubeconfig context and OCIR repository. A locally installed OCI profile does not identify or authorize a particular cluster destination.

## OCI benefits demonstrated

1. OKE manages Kubernetes infrastructure while the application uses portable Deployment/Service/probe primitives. Show readiness-controlled traffic and recovery from deleting a pod.
2. OCI Container Registry stores the model-bearing image for cluster delivery. Show the image digest linking evaluated artifacts to the deployed release.
3. CPU inference provides a starting point without GPU provisioning. Measure actual pod memory, CPU, latency, and requests per second on the selected OCI node shape before making cost/performance claims.
4. Later, add Object Storage for DVC datasets and MLflow artifacts, and workload identity for private artifact access. These are planned benefits, not implemented in this stage.

## Implementation sequence and verification

Agreed follow-on: after the first regular OKE deployment is verified, introduce KServe `InferenceService` for the DistilBERT predictor. FastAPI will retain the UI and request validation and delegate inference to that endpoint. The user has verified KServe Standard/internal predictor installation and serving. The approved frontend design uses the existing /predict contract, pinned identity checks, bounded HTTP calls, and no local fallback. The [frontend guide](reviewops-kserve-frontend.md) describes the release and pending cloud validation. The [local evaluation and MLflow plan](superpowers/plans/2026-10-05-reviewops-mlflow.md) covers the immediate milestone.

1. Add shared inference and compatibility tests; rerun the existing 14 tests.
2. Add deterministic dataset acquisition and evaluation; test filtering, label integrity, metric computation, and repeatability.
3. Add MLflow logging, offline model reload, registration, and gate tests covering both passing and failing runs.
4. Run the real public-data evaluation and inspect recorded metrics, artifacts, registry version, and prediction parity.
5. Export the registered artifact; build and smoke-test the container using an available local runtime. Podman is installed; runtime readiness and build support remain unverified.
6. Validate Kubernetes manifests locally where tooling permits; document unverified server-side or cloud behavior.
7. Deploy to the identified OKE destination and verify rollout, predictions, recovery, and rollback, subject to the execution boundary below.

## Execution boundaries and local facts

The current app runs on port 8000. OCI CLI 3.80.0, kubectl, and Podman are installed. No Docker executable was found. OCI DEFAULT configuration is present in us-ashburn-1; no live OCI discovery has been performed. The workspace is not a Git checkout, so there is no commit identity to record yet; hash source files instead.

The OCI code-generator skill permits local inspection and generation but prohibits OCI-affecting execution. Its generated cloud commands are user-run unless the user supplies an instruction overriding that boundary. No OCI resources or state will be changed in the design stage.

## References checked

- MLflow database-backed registry and alias workflow: https://www.mlflow.org/docs/latest/ml/model-registry/workflow/
- MLflow models-from-code and artifact packaging: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html
- Public IMDb dataset card: https://huggingface.co/datasets/stanfordnlp/imdb
- OKE private image pulls: https://docs.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengpullingimagesfromocir.htm
- OCI Registry image publishing: https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registryimagespushingpulling.htm

MLflow 3.16.1 is pinned and its custom PyFunc packaging/reload was verified with Transformers 5.18.0 and PyTorch 2.14.1. This does not claim compatibility for every built-in MLflow Transformers flavor. Public-data results are linked above; image building and cloud rollout remain unverified future stages.
