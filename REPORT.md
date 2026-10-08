# BikeOps: first public-data training run

Predict total hourly rentals across Capital Bikeshare using calendar and weather inputs.

## Data and evaluation

- Real public data: 17,379 hourly records from 2011–2012.
- Training precedes July 2012; validation is July–September; final test is October–December.
- Model selection uses validation MAE. Final test data is not used to fit or select models.
- Target components `casual` and `registered` are excluded.

| Model | Validation MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| mean_baseline | 197.83 | 156.57 | 208.06 | -0.065 |
| random_forest_leaf_3 | 95.76 | 87.22 | 127.44 | 0.600 |

The selected model reduces test MAE by **44.3%** versus the constant-mean baseline.
MAE is the average absolute error in rentals per hour. This is a simple baseline comparison, not a claim of state-of-the-art performance.

## Example predictions

| Historical hour | Actual rentals | Predicted rentals |
|---|---:|---:|
| 2012-10-01 00:00:00 | 45 | 37.2 |
| 2012-10-01 01:00:00 | 18 | 17.2 |
| 2012-10-01 02:00:00 | 12 | 14.6 |
| 2012-10-01 03:00:00 | 7 | 11.2 |
| 2012-10-01 04:00:00 | 10 | 11.3 |

## Limitations

- Historical system-wide demand, not station availability.
- Uses observed weather: not a validated future-weather forecast.
- Missing hours are not filled; no live data or OCI deployment yet.
- Selected artifact remains fitted on training partition only.

## Attribution

Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894. CC BY 4.0.
