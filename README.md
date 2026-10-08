# ReviewOps: pretrained sentiment analysis for learning MLOps on OCI OKE

Enter an English movie review and get positive or negative sentiment from a public pretrained DistilBERT model. FastAPI and the browser form run locally on CPU, with no initial training. Public-data evaluation and local MLflow tracking, registration, and export are implemented. The Linux AMD64 CPU container for OKE is built and tested under local emulation; an Arm64 image remains for local testing. The original OCI deployment and KServe predictor have been verified in the user’s terminal. The user also verified frontend delegation through the public Load Balancer with `inference_backend: kserve`. DVC and Kubeflow are later milestones.

## Run

Python 3.12 was used. From this directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-mlops.txt
.venv/bin/python download_model.py
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The environment and checkpoint are already present. Open http://127.0.0.1:8000 for the form or http://127.0.0.1:8000/docs for interactive API documentation.

The downloader pins an exact Hugging Face commit in `artifacts/review_model/manifest.json`; later downloads reuse it. Startup checks model/tokenizer file hashes and performs a readiness prediction. Serving uses local files, Safetensors weights, and disables remote custom code. Review text is not sent to a hosted inference API or saved by this application.

## Try the API

```bash
curl http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"The movie was wonderful. I loved it."}'
```

The response includes `label`, `score`, `model_id`, `model_version`, `token_count`, and server `inference_ms`. Model version is `hf:<commit>`. Inference time includes tokenization and model execution, not network time; individual calls are not performance benchmarks.

Text is trimmed and must contain 1–2,000 characters. Text above 512 tokens, non-string inputs, and unknown fields return HTTP 422. Long text is rejected rather than silently truncated. `/health` checks liveness; `/ready` and `/predict` return HTTP 503 if the checkpoint cannot load.

Set `REVIEWOPS_MODEL_DIR` to a local checkpoint directory with its manifest and restart to change the artifact location. The application supports this specific model and label mapping.

## Model and limitations

Model: https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

The model card identifies Hugging Face as developer, SST-2 as fine-tuning data, and Apache 2.0 as the license. The original model card is retained with the downloaded checkpoint. Attribute the model and retain applicable license notices when redistributing it.

The score is not measured accuracy or a calibrated probability guarantee. This binary model has no neutral label. Sarcasm, mixed sentiment, other languages, and domains outside English movie reviews can produce misleading results. This demo does not detect language or measure people's feelings.

## Verify

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

ReviewOps tests use the real pretrained model and local SQLite registries, covering API validation, sampling, evaluation gates, startup/warm-up failures, registry reload, and verified export. Full-suite verification requires the MLOps dependencies installed above. BikeOps tests remain in the suite.

## Evaluate and register with MLflow

Install the additional local MLOps dependencies:

```bash
.venv/bin/python -m pip install -r requirements-mlops.txt
.venv/bin/python acquire_review_data.py
.venv/bin/python evaluate_reviews.py
```

Acquisition uses the public IMDb test split at a recorded commit, then selects 100 eligible reviews per label using seed 42. It filters to the same character/token limits used by serving and records source hashes, selected row IDs, and exclusions. Later acquisition reuses the recorded dataset revision. Public-data provenance and the upstream card live under `data/reviews/`; raw reviews are not packaged for deployment.

Evaluation writes `artifacts/review_evaluation/report.json` and logs a real run to local MLflow. The demonstration gate requires accuracy and macro F1 >= 0.80, both labels, and no inference failures. A passing run registers a numbered `reviewops-distilbert` version and updates `candidate`; a failing run remains visible, returns exit code 1, and does not register. These teaching thresholds are not business acceptance criteria.

The completed first public-data run measured **92% accuracy** and **0.920 macro F1** on 200 selected short reviews, with 16 mistakes and no inference failures. The final verified candidate is version **2**; it uses the same pretrained weights as the initial version 1 run. See the [evaluation results](docs/reviewops-evaluation-results.md).

Open the local dashboard at http://127.0.0.1:5000. To start it again, run this from the project directory:

```bash
.venv/bin/mlflow server --backend-store-uri "sqlite:///$PWD/artifacts/mlflow/reviewops.db" \
  --host 127.0.0.1 --port 5000 --workers 1 --no-serve-artifacts
```

This server is a local learning setup. Metadata is stored in SQLite; model packages and reports are in the artifact directory recorded by the experiment. Back up both. Keep the dashboard bound to loopback; it is not a public authenticated service.

Export a numeric version from the registry:

```bash
.venv/bin/python export_review_model.py --version 2 --output-dir artifacts/releases/v2
```

The verified version 2 export already exists, so rerunning that command intentionally refuses to overwrite it. Choose a new destination when repeating verification. Exported `checkpoint/` files can be served through `REVIEWOPS_MODEL_DIR`; `release.json` records the registry version, upstream revision, run ID, and file hashes. The export and registered model both support offline inference. Only load model code from a trusted registry: models-from-code executes Python when loaded.

A stricter teaching gate can demonstrate refusal to release without changing the model:

```bash
.venv/bin/python evaluate_reviews.py --min-accuracy 1 --min-macro-f1 1 \
  --output-dir artifacts/review_evaluation_failed
```

Expect exit code 1 for this sample and unchanged candidate version. This failure is an explicitly stricter evaluation requirement, not a degraded model.

## Local container and OKE delivery

The OKE image is `localhost/reviewops:mlflow-v2-amd64`. Open http://127.0.0.1:8004 for the emulated AMD64 demo. The Arm64 local test image remains at http://127.0.0.1:8002; the original application is on port 8000.

Both AMD64 and Arm64 images have been tested. The AMD64 image for OKE runs as UID 10001 with a read-only root filesystem and writable `/tmp`, and contains CPU-only PyTorch without MLflow. Real predictions passed with networking disabled. Positive/negative predictions and invalid-input/readiness behavior were checked live. The AMD64 image is approximately 1.53 GB uncompressed (Arm64: 1.43 GB); this is not a memory requirement.

See the [beginner OKE runbook](docs/reviewops-oke-runbook.md) for build commands, architecture explanations, cloud prerequisites, publication, probes, recovery, and rollback. See the [container verification results](docs/reviewops-container-results.md) for the tested image identity and checks. Local image/serving test evidence lives in `artifacts/container/`. The [OCI references](OCI_REFERENCES.md) record official sources and execution boundaries.

The intended OKE context is `context-cef-oke-liftoff-ch4yy43wxtq`; use `export OCI_CLI_PROFILE=axon`. Original image publication and OKE Deployment/Service verification were completed by the user; the user subsequently reported a passing public-LB frontend-to-KServe smoke test. Confirmed OCIR repository: `iad.ocir.io/idtfszw3ugfy/reviewops`; the user reports Ready AMD64 workers. Do not apply `k8s/base/` directly: it contains an image placeholder. Generate a bundle with `render_oke_release.py` only after the real published digest is available. The renderer checks syntax/release metadata but cannot independently confirm the contents of an arbitrary supplied registry digest.

## Next learning milestones

For a first-time audience, see the [beginner architecture and presentation guide](docs/reviewops-beginner-guide.md). It includes diagrams, component explanations, OCI demo evidence, and a speaking walkthrough. The [next-stage design](docs/reviewops-mlflow-oke-design.md) records the baseline implementation and subsequent serving design. Frontend-to-KServe serving is verified by the user; later data/pipeline stages remain pending.

1. Completed by the user: publish the original AMD64 image and verify its OKE Deployment and public LB.
2. Completed: connect FastAPI to KServe; user-reported public-LB smoke checks passed.
3. Version evaluation data with DVC; store data/artifacts in OCI Object Storage.
4. Automate acquisition, evaluation, and release checks with Kubeflow Pipelines.
5. Demonstrate OCI workload identity, private artifact access, compute sizing, scaling, and rollback with measurements.

A pretrained model starts at evaluation and serving. Fine-tuning is a later milestone; this demo does not reproduce upstream training.

## BikeOps reference

The bike rental project is preserved in `train.py`, `bike_app.py`, `bike_static/`, `data/raw/`, and its original artifacts. See `BIKEOPS.md` and `REPORT.md`. Run its server using:

```bash
.venv/bin/python -m uvicorn bike_app:app --host 127.0.0.1 --port 8001
```

## FastAPI-to-KServe frontend release

The original app was deployed to OKE and its smoke test passed in the user's terminal. The public LB address reported is http://129.80.162.140. KServe Standard controller and internal `reviewops-sentiment` predictor are now Ready, and the user reports predictor smoke success.

The new FastAPI release supports `REVIEWOPS_PREDICTOR_URL` and forwards predictions without loading model weights in the frontend process. Its AMD64 image tag is `localhost/reviewops:frontend-v1-amd64`; the user reported a passing OKE public-LB smoke test with the KServe backend marker. The first frontend image retains the shared model/package recipe; a smaller dedicated image is a future refinement.

See [the beginner KServe/frontend guide](docs/reviewops-kserve-frontend.md) for the updated architecture, component explanations, local evidence, push, atomic image/config rollout, smoke verification, and rollback. Local two-container demo: http://127.0.0.1:8007. Use the smoke test's `--expect-backend kserve` flag to verify delegation after the cloud rollout.
