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

## Important threshold interpretation

The intervention slider **does not change the evaluation cohort**. The synthetic holdout stays fixed at **24,000 terminal-days**, containing **640 observed failures** for a baseline failure prevalence of **2.67%**.

Changing the threshold only changes how the model converts predicted risk into an action. Therefore the following are threshold-dependent:

- failures detected
- failures missed
- healthy terminal-days flagged
- healthy terminal-days not flagged
- total interventions
- precision
- recall
- false-positive rate
- estimated failures prevented under the stated intervention assumption

This distinction is now made explicit in the live dashboard so fixed cohort statistics are not mistaken for slider-driven model outputs.

## Results

| Model | Features | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| Logistic regression | snapshot | 0.912 | 0.611 |
| XGBoost | snapshot | 0.912 | 0.597 |
| Logistic regression | snapshot + history | 0.923 | 0.719 |
| XGBoost | snapshot + history | 0.920 | 0.786 |

The main result is not that one algorithm "wins." The useful result is that longitudinal telemetry materially improves retrieval of rare failures, and that the intervention threshold changes the operational outcome.

At a 25% intervention threshold in the synthetic holdout:

- 500 of 640 impending failures are detected
- 140 failures are missed
- 200 healthy terminal-days are flagged
- 700 proactive interventions are triggered
- precision is 71.4%
- recall is 78.1%

Those trade-offs are the point of the project.

## Demo

The static demo lives in [`docs/`](docs/) and is designed for GitHub Pages. It has two views:

1. **Intervention threshold** — move the threshold and watch failures detected, missed failures, healthy terminal-days flagged, intervention volume and estimated support impact update while the holdout cohort stays fixed.
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

The telemetry schema is intentionally generic. It includes signals such as Wi-Fi strength, reconnect events, checkout latency, timeout rate, transaction volume and charging interruptions. These fields are simulated for the purpose of demonstrating modeling and decision support; they are not presented as Square's internal production telemetry schema.

Square's public Terminal Sandbox documents simulated checkout states, including successful checkouts, cancellations, timeouts and offline-device behavior:

- https://developer.squareup.com/docs/devtools/sandbox/testing
- https://developer.squareup.com/docs/devtools/sandbox/overview

## Why this project

Terminal reliability is a classification problem only until someone needs to decide what to do with the probability.

The goal here is to connect model performance to an operational decision: protect the seller without creating unnecessary intervention burden.
