# Before Suzy Misses a Sale

A small decision-science project about terminal reliability: can we detect emerging failure patterns early enough to intervene before a seller loses a sale?

The project uses synthetic terminal telemetry and public Square Terminal Sandbox behavior as the framing. No Block or Square internal data are used.

## Question

A model can rank terminals by failure risk. The harder question is operational: **when is the predicted risk high enough to justify an intervention?**

Too low a threshold and healthy terminals get unnecessary support touches. Too high a threshold and real failures reach sellers first.

## What I built

- 4,000 synthetic terminals observed over 30 days (120,000 terminal-days)
- A 24-hour material-failure outcome
- Snapshot and longitudinal feature sets
- Logistic regression and XGBoost models
- Grouped holdout validation so the same terminal never appears in train and test
- ROC and precision-recall evaluation for a low-prevalence outcome
- A threshold simulator translating probabilities into operational consequences
- A 7 / 14 / 30 day telemetry view to inspect deterioration before failure
- A Snowflake-compatible SQL feature-engineering example using window functions and leakage-safe trailing history

## Important threshold interpretation

The intervention slider **does not change the evaluation cohort**.

The evaluation set is:

`1,000 held-out terminals × 24 scored days per terminal = 24,000 prediction observations`

The simulation spans 30 days, but days 0–5 are removed before modelling so trailing-history features have prior context. One held-out terminal on one eligible scored day is one prediction observation.

The fixed evaluation set contains **640 failure-positive observations**, giving a baseline positive rate of **2.67%**.

Changing the risk threshold only changes how the model converts predicted risk into an action. Therefore failures detected, failures missed, healthy observations flagged, interventions, precision, recall and false-positive rate are threshold-dependent while the evaluation set remains fixed.

## Intervention impact is a scenario, not a causal model output

The dashboard no longer treats “failures prevented” as something learned by the classifier.

Instead, it exposes a separate **assumed intervention-effectiveness** slider. The displayed scenario output is:

`detected failure-positive observations × assumed intervention effectiveness`

This is explicitly a sensitivity analysis. The risk model estimates who is likely to fail; it does **not** estimate the causal effect of support or device intervention.

## Results

| Model | Features | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| Logistic regression | snapshot | 0.912 | 0.611 |
| XGBoost | snapshot | 0.912 | 0.597 |
| Logistic regression | snapshot + history | 0.923 | 0.719 |
| XGBoost | snapshot + history | 0.920 | 0.786 |

The main result is not that one algorithm "wins." The useful result is that longitudinal telemetry materially improves retrieval of rare failures, and that the intervention threshold changes the operational trade-off.

At a 25% intervention threshold in the synthetic holdout:

- 500 of 640 failure-positive observations are detected
- 140 failure-positive observations are missed
- 200 healthy prediction observations are flagged
- 700 proactive interventions are triggered
- precision is 71.4%
- recall is 78.1%

Those trade-offs are the point of the project.

## SQL artifact

[`sql/build_terminal_risk_features.sql`](sql/build_terminal_risk_features.sql) shows how the same modelling table could be assembled in a warehouse from daily telemetry and event tables.

It demonstrates:

- daily deduplication with `ROW_NUMBER()` / `QUALIFY`
- 7 / 14 / 30 day window features
- current-vs-history deltas
- next-24-hour label construction
- explicit prevention of future leakage by ending history windows at `1 PRECEDING`
- preservation of `terminal_id` for grouped train / test validation

The table names are illustrative. The SQL is not based on Block or Square internal schemas.

## Demo

The static demo lives in [`docs/`](docs/) and is designed for GitHub Pages. It has two views:

1. **Intervention threshold** — move the threshold and watch detection, false positives and intervention volume change. A separate scenario slider shows how assumed intervention effectiveness changes the hypothetical impact estimate.
2. **Telemetry history** — switch between 7, 14 and 30 day views for a representative synthetic terminal approaching failure.

## Reproduce the analysis

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

The script regenerates the synthetic data, trains the models and prints the evaluation table plus threshold metrics.

## Notes on the synthetic data

The telemetry schema is intentionally generic. It includes signals such as Wi-Fi strength, reconnect events, checkout latency, timeout rate, transaction volume and charging interruptions. These fields are simulated for the purpose of demonstrating modelling and decision support; they are not presented as Square's internal production telemetry schema.

Square's public Terminal Sandbox documents simulated checkout states, including successful checkouts, cancellations, timeouts and offline-device behavior:

- https://developer.squareup.com/docs/devtools/sandbox/testing
- https://developer.squareup.com/docs/devtools/sandbox/overview

## Why this project

Terminal reliability is a classification problem only until someone needs to decide what to do with the probability.

The goal here is to connect model performance to an operational decision: protect the seller without creating unnecessary intervention burden.
