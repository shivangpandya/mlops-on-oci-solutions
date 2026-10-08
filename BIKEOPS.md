# BikeOps: public-data MLOps learning project

Predict hourly bike rentals from calendar and weather information. Local training and the FastAPI browser application are implemented. OCI deployment, MLflow, DVC, and Kubeflow are later milestones.

## Run

Python 3.12 was used for the recorded run. From this directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python train.py
```

The environment is already created in this workspace. To repeat training now, run `.venv/bin/python train.py`.

## Open the prediction application

```bash
.venv/bin/python -m uvicorn bike_app:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001 for the browser form or http://127.0.0.1:8001/docs for interactive API documentation. The browser form is served by FastAPI itself, so one server is enough.

Enter a date, hour, weather, temperature in °C, humidity in %, wind speed in km/h, and whether the date is a holiday. The API derives month, weekday, season, and working-day status. It converts temperature using `/41`, humidity using `/100`, and wind speed using `/67`, matching the downloaded CSV's original readme. Season boundaries match the actual CSV transitions: March 21, June 21, September 23, and December 21.

Example prediction request:

```bash
curl http://127.0.0.1:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"date":"2012-10-01","hour":8,"holiday":false,"temperature_c":20,"humidity_pct":60,"windspeed_kmh":10,"weather":"clear"}'
```

Weather values: `clear`, `mist`, `light_precipitation`, or `severe_weather`. The response contains the raw predicted hourly rentals, a SHA-256 model version, and the normalized model inputs. The UI rounds rentals to the nearest integer.

The supported temperature scale is 0–41 °C, humidity 0–100%, wind speed 0–67 km/h, and hour 0–23. Inputs outside those ranges and unknown fields are rejected with HTTP 422. Severe-weather observations are sparse in the training data; extreme scenarios should be interpreted cautiously.

The server loads the saved model once at startup; it never trains during prediction. `/health` checks liveness; `/ready` returns HTTP 503 when the model cannot load, as does `/predict`. Set `BIKEOPS_MODEL_PATH` to serve a different trusted artifact with the same feature schema, then restart. The default is `artifacts/model.joblib`. The SHA-256 version identifies the loaded bytes.

The local app is a historical learning demo: selecting a present-day date creates a hypothetical scenario and does not fetch current weather. The date is interpreted as the local Washington, D.C. calendar.

## What training does

1. Read the public `data/raw/hour.csv` dataset.
2. Check missing values, duplicate hours, and rental-count consistency.
3. Train a constant-mean baseline and two Random Forest candidates on January 2011–June 2012.
4. Choose the model with the lowest validation error on July–September 2012.
5. Measure final performance on October–December 2012 without fitting to those records.
6. Save the selected model with its preprocessing and verify saved/reloaded predictions match.

`cnt` is the answer being predicted. `casual` and `registered` sum to that answer and must never be model inputs. Features use an explicit allowlist. Year is omitted so the API will not require the dataset-specific 2011/2012 flag.

## Outputs

- `REPORT.md`: readable results and limitations.
- `artifacts/metrics.json`: metrics, date ranges, package versions, and dataset SHA-256.
- `artifacts/model.joblib`: saved model and preprocessing. Load only artifacts you trust.
- `artifacts/sample_predictions.csv`: five held-out examples.

Training inputs `temp`, `hum`, and `windspeed` use normalized units; the API converts friendly units to those same values. This is a historical regression exercise using observed weather, not a validated future-weather forecast. It predicts system-wide rentals, not available bikes at a specific station.

## Public dataset and attribution

Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Download: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip

Fanaee-T, H. (2013). *Bike Sharing*. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.

Dataset license: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. The original CSV and dataset readme are included unchanged. Our preprocessing and models are derived work; attribution does not imply endorsement. Observations cover Capital Bikeshare in 2011–2012.

## Next learning milestones

1. Track training runs and artifacts in MLflow; version this CSV with DVC.
2. Publish container images to OCI Container Registry and deploy on OKE.
3. Use Object Storage for datasets/models and demonstrate workload identity for native SDK artifact access.
4. Run the same training stages in standalone Kubeflow Pipelines.
5. Add monitoring, model release/rollback, and measured OCI compute results for the KubeCon demo.

The training code is independent of OCI. The cloud milestones add deployment, artifact storage, identity, networking, and operations.
