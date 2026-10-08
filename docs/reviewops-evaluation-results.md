# ReviewOps: public-data evaluation and release evidence

This is a measured local evaluation, not upstream training or an OKE benchmark. The full evidence is in `artifacts/review_evaluation/report.json`, with a corresponding MLflow run.

## What we tested

| Item | Recorded value |
|---|---|
| Model | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| Model revision | `714eb0fa89d2f80546fda750413ed43d93601a13` |
| Public data | `stanfordnlp/imdb`, test split |
| Dataset revision | `e6281661ce1c48d982bc483cf8a173c1bbeb5d31` |
| Source rows | 25,000 |
| Eligible rows | 20,890 |
| Excluded by character length | 4,091 |
| Excluded by token length | 19 |
| Selected sample | 200 reviews; 100 per class; seed 42 |
| Sample SHA-256 | `3ebec9518168acdc87bfe5a5f9f46ace5775d6b874816f8894c04a57e70a816f` |

Eligibility requires trimmed text of 1–2,000 characters and no more than 512 tokens. Exclusions are sequential: character failures are counted first. Results apply to this eligible short-review sample, not the entire IMDb test split. The model was fine-tuned on SST-2; this evaluation does not prove independence from all upstream pretraining data.

## Results

| Measure | Result |
|---|---|
| Accuracy | 92.0% (184/200 correct) |
| Macro F1 | 0.919992 |
| Wrong predictions | 16 |
| Inference failures | 0 |
| Positive precision / recall | 0.928571 / 0.910000 |
| Negative precision / recall | 0.911765 / 0.930000 |

Read the confusion matrix by row: the true label is on the left; the predicted label is across the top.

| True label | Predicted negative | Predicted positive |
|---|---:|---:|
| Negative | 93 | 7 |
| Positive | 9 | 91 |

**Audience explanation:** “Of 100 negative reviews, the model correctly identified 93 and called 7 positive. Of 100 positive reviews, it correctly identified 91 and called 9 negative. That gives 184 correct answers out of 200.”

Warmed local inference measured p50 26.480 ms and p95 53.273 ms, including tokenization. Development checks were running during this measurement; it is not an isolated capacity benchmark, end-to-end browser latency, or a measurement of OCI performance. Machine details are retained in the JSON report.

## Release evidence

The configurable teaching gate required accuracy >= 0.80, macro F1 >= 0.80, both labels present, and no inference failures. The run passed.

- Experiment: `reviewops-evaluation`.
- MLflow run ID: `a42731d0210a442a8b6d687f17d998c9`.
- Registered model: `reviewops-distilbert`.
- Numeric version: `2`; alias: `candidate`.
- Export: `artifacts/releases/v2/checkpoint/`, with registry/run/revision/checksums in `artifacts/releases/v2/release.json`.

Registration is not deployment. The running local API has the same verified upstream checkpoint; OKE and KServe are future stages. A stricter gate requiring perfect accuracy/F1 is useful for showing a genuine failed release decision on the same sample.

For the presentation, show the browser prediction, this confusion matrix, the MLflow run, the numbered candidate, and the export metadata. Explain that a review’s model score is different from measured dataset accuracy.

See the [beginner guide](reviewops-beginner-guide.md) and [run instructions](../README.md).

The stricter demonstration was also run: thresholds of 1.0 accuracy and macro F1 failed on the same sample. Run `42243ccd2059402c80563c9fe18cdf2c` records the failed gate. No model version was added, and `candidate` remained version 2.

The initial evaluation registered version 1 (run `4fae5375888c4182a86d0f191c700ccb`). After review fixes, the final evaluator registered version 2 with identical pretrained weights and the same accuracy/F1. This is a new recorded evaluation/package version, not additional training. The latest report includes requirements-file hashes and an installed-environment snapshot.
