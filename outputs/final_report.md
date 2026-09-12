# Pipeline Run Report

- Raw readings: 29997 rows -> pivoted to 861 site/date rows -> 829 rows with a usable target.

- 10-fold cross-validation.

| Model             |       R2 |     RMSE |      MAE |
|:------------------|---------:|---------:|---------:|
| Gradient Boosting | 0.97574  | 0.257772 | 0.163128 |
| Random Forest     | 0.961171 | 0.326743 | 0.183385 |
| Linear Regression | 0.32232  | 1.22037  | 0.746849 |


- Selected model: **Gradient Boosting**

- Standard 80/20 holdout split.
- Holdout metrics: `{'mae': 0.17812569202301604, 'rmse': 0.2768643373138808, 'r2': 0.9721450747372823}`

- Explainability: SHAP TreeExplainer succeeded.

- Anomaly threshold (std, x2.0): 0.4044

- Residual error metrics: `{'count': 829.0, 'mean_error': 2.220446049250313e-16, 'mae': 0.13360725787071814, 'rmse': 0.2020903122688723, 'max_absolute_error': 1.004046283491772}`

- Flagged anomalies: 53 / 829
