# ReviewOps Local Evaluation and MLflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Execution status:** Local milestone implemented and independently reviewed; cloud milestones remain separate. See [implementation notes](../../reviewops-implementation-notes.md).

**Goal:** Evaluate the existing pretrained sentiment model on public labeled reviews and create a traceable, reloadable MLflow candidate when it passes a demonstration gate.

Oka
**Tech Stack:** Python 3.12, PyTorch, Transformers, FastAPI, Hugging Face dataset acquisition, scikit-learn metrics, MLflow, SQLite, unittest.

**Spec:** [ReviewOps design](../../reviewops-mlflow-oke-design.md). This plan covers the local evaluation/registration/export stage. Container delivery, OKE, KServe, DVC, and Kubeflow Pipelines are subsequent milestones.

## Global Constraints

- Preserve `POST /predict`, `/health`, `/ready`, and the existing browser interface.
- Trim text; reject non-string/blank inputs, inputs above 2,000 characters, and tokenized inputs above 512 tokens. Never silently truncate.
- Use CPU, local model files, Safetensors, and no remote custom code. Verify manifest identity and hashes before loading.
- Preserve model identity `distilbert/distilbert-base-uncased-finetuned-sst-2-english` and its pinned upstream revision.
- Select a reproducible IMDb test sample of 200 eligible rows, 100 per class, using seed 42. Record exclusions, source revision, row IDs, and hashes.
- Default gate: accuracy >= 0.80, macro F1 >= 0.80, both labels represented, and zero inference failures. These are teaching thresholds.
- Use experiment `reviewops-evaluation`, registered model `reviewops-distilbert`, and alias `candidate`. Do not promote `champion` automatically.
- No live OCI operations or deployment in this milestone. No invented evaluation results.
- The workspace has no Git repository. Record source hashes; omit commit commands rather than inventing a commit identity.
- Verify MLflow compatibility with installed Transformers 5.18.0 and PyTorch 2.14.1 before choosing a pinned dependency version.

## Review Focus

- Corrupted checkpoint: no prediction and no successful export (Tasks 1, 5).
- Dataset has invalid labels or too few eligible examples: fail, rather than changing the sampling promise (Task 2).
- Inference failure or non-finite result: log a failed run and never register it (Tasks 1, 3, 4).
- Registry write failure: return failure and never publish a successful release record (Task 4).
- Reloaded model has different validation/preprocessing: reject or predict identically across API and PyFunc (Tasks 1, 4).

## Files and interfaces

| File | Responsibility |
|---|---|
| `review_inference.py` | Verified model loading, input checks, and a single prediction implementation. |
| `review_app.py` | HTTP/UI behavior, delegating model execution to shared inference. |
| `acquire_review_data.py` | Pinned public-data acquisition and deterministic sample creation. |
| `evaluate_reviews.py` | Metrics, timing, release decision, and MLflow run orchestration. |
| `review_mlflow_model.py` | Models-from-code PyFunc wrapper consuming bundled checkpoint files. |
| `export_review_model.py` | Export a numeric registry version and write verified release metadata. |
| `requirements-mlops.txt` | Pinned evaluation/MLflow dependencies in addition to current requirements. |
| `tests/test_review_inference.py`, `tests/test_review_data.py`, `tests/test_review_evaluation.py`, `tests/test_review_registry.py` | Unit and real-artifact integration checks. |
| `README.md`, `.gitignore` | Run instructions and exclusion of local database/cache/artifact outputs. |

### Task 1: Shared inference without changing API behavior

**Interfaces:** `SentimentModel(model_dir: Path)`, `SentimentModel.predict(text: str) -> dict`; `InvalidReviewError(ValueError)` for invalid input, `ModelUnavailableError(RuntimeError)` for loading or inference failure. Prediction fields match the current API response.

- [x] Write tests for real positive/negative predictions, finite score, pinned identity, blank and oversized inputs, token overflow, and corrupted manifest/file. Assert that valid prediction labels/scores match the existing API and invalid text retains HTTP 422 behavior.
- [x] Run `.venv/bin/python -m unittest discover -s tests -p test_review_inference.py -v`; verify failure because the shared module is absent.
- [x] Extract verified loading and inference into `review_inference.py`, preserving all checks and two CPU threads. Delegate from `review_app.py`; translate validation errors to 422 and unavailable inference to 503.
- [x] Run the new tests and the full existing suite; require all passes before continuing.

### Task 2: Reproducible public evaluation sample

**Interfaces:** `select_sample(rows: list[dict], model: SentimentModel, seed: int = 42) -> tuple[list[dict], dict]`. Selected rows include `source_row_id`, `text`, and normalized `label`. `acquire_review_data.py` CLI writes `data/reviews/imdb_sample.jsonl` and `data/reviews/imdb_sample_manifest.json`.

- [x] Write fixture tests asserting identical row IDs/order/hash with the same input and seed, exactly 100 rows per label, and correct exclusion counts for character/token overflow. Assert invalid labels and insufficient eligible rows raise clear errors; do not silently resample a different split.
- [x] Run `.venv/bin/python -m unittest discover -s tests -p test_review_data.py -v`; verify expected missing-module failure.
- [x] Implement pinned acquisition of `stanfordnlp/imdb` test data, record revision and source file hashes, and apply deterministic eligibility/sampling. Save attribution and acquisition metadata. Subsequent acquisition reuses the recorded revision; failures must not fall back to invented examples.
- [x] Pin acquisition dependencies in `requirements-mlops.txt`; run the tests and dependency consistency check. Keep ordinary unit tests independent of network access.

### Task 3: Metrics and explicit gate

**Interfaces:** `compute_metrics(labels: list[str], predictions: list[str]) -> dict`; `passes_gate(metrics: dict, labels_present: set[str], failure_count: int, min_accuracy: float = 0.80, min_macro_f1: float = 0.80) -> bool`. `evaluate_reviews.py` accepts dataset, manifest, model directory, and threshold options.

- [x] Write tests with known correct/wrong label pairs to verify accuracy, per-label precision/recall, macro F1, and confusion-matrix order `NEGATIVE`, `POSITIVE`. Assert exact-threshold pass; lower metrics, absent class, inference failure, non-finite values, and empty input fail.
- [x] Run `.venv/bin/python -m unittest discover -s tests -p test_review_evaluation.py -v`; verify expected failure before implementation.
- [x] Implement metrics and gate; check sample/manifest hashes before inference. Warm the model before timing and report p50/p95 in milliseconds with machine/thread/sample details. Save `artifacts/review_evaluation/report.json` with metrics, coverage, exclusions, failure details, and explicit gate decision.
- [x] Run the tests. Failed evaluation must return nonzero and preserve its report; no measured benchmark is implied by synthetic timing tests.

### Task 4: Real MLflow logging and registration

**Interfaces:** `ReviewPythonModel` in `review_mlflow_model.py` loads the `checkpoint` artifact through `SentimentModel`; `predict(context, model_input, params=None)` accepts a DataFrame with strict string column `text` and returns prediction rows. `evaluate_reviews.py` adds `--tracking-uri`, defaulting to an absolute workspace-local SQLite URI.

- [x] Write integration tests using a temporary SQLite store and the real local model: log/reload offline and compare label, score, and model identity with shared inference; validate rejected input consistently; passing run creates a version and updates `candidate`; failed-gate run creates no version; simulated registry failure creates no success record. Timing values need not match.
- [x] Verify tests fail before adding logging/registration. Check a compatible MLflow release against official package metadata and docs, install and pin it, then run `pip check`.
- [x] Implement model-from-code packaging with input/output signature and bundled tokenizer/checkpoint. Log metrics, checkpoint and dataset identity, source/dependency hashes, reports, and selection parameters. Register only after a passing gate; record the numeric version and run ID after registration/alias assignment succeeds.
- [x] Run real SQLite integration tests and full suite. Keep loaded model predictions offline. Start MLflow UI bound to loopback; do not expose an unauthenticated server publicly.

### Task 5: Export and demonstrate the actual milestone

**Interfaces:** `export_registered_model(tracking_uri: str, version: int, output_dir: Path) -> dict` in `export_review_model.py`, fixed registered name `reviewops-distilbert`. Result records numeric version, run ID, Hugging Face revision, and exported hashes for later container delivery.

- [x] Write tests asserting export/reload prediction parity, numeric-version requirement, missing-version failure, and rejection of corrupted exported files. Never resolve a moving alias during runtime inference.
- [x] Run `.venv/bin/python -m unittest discover -s tests -p test_review_registry.py -v`; verify missing export implementation failure.
- [x] Implement export into a temporary directory, verify checkpoint contents, and publish destination/release metadata only on success. Refuse to overwrite an existing destination by default.
- [x] Acquire the real public data and run real evaluation. Inspect the report and MLflow run; if the gate passes, export its numeric version and verify predictions. If it fails, deliver the failed-run evidence and explain the result rather than weakening thresholds to force a release.
- [x] Update README with tested acquisition/evaluation/UI/export commands, sample limitations, artifact backup scope, and result links. Add generated dataset/cache/database/model outputs to `.gitignore`. Rerun full tests and `pip check`; report actual results and any remaining blockers.

## Self-review and milestone exit

Each task has a red/green verification cycle and covers the five review risks. Numeric model versions, dataset identities, and prediction interfaces agree across tasks. The output is a functioning local lifecycle, not cloud deployment. Preserve the current service while testing a refactor on a separate port; restart the default service only after regression checks pass.

Expected audience walkthrough: submit a review, inspect a real evaluation report, open its MLflow run, show a numbered candidate if the gate passed, and explain the exported artifact that the next milestone will package.

## Following milestones

1. Build and verify a CPU Linux container, publish to OCIR, and deploy a regular Kubernetes Deployment/Service on the identified OKE cluster.
2. Introduce KServe `InferenceService` as the model-serving layer; FastAPI keeps the UI and validation and calls the predictor. Select and verify deployment mode, networking, controller dependencies, and predictor protocol before implementation.
3. Add DVC plus Object Storage for reproducible data/artifact storage, then Kubeflow Pipelines for workflow automation.

KServe has been agreed as a later stage; it is not installed or used by the current application. Document and verify each stage before presenting it as completed.
