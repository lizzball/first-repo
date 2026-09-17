from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from simulate import add_history_features, simulate_terminal_fleet


SNAPSHOT_FEATURES = [
    "firmware",
    "seller_type",
    "wifi_rssi",
    "reconnects",
    "latency_s",
    "timeout_rate",
    "transactions",
    "battery_risk",
    "charge_interrupts",
    "weekend",
]


def prepare_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train[features], test[features]], axis=0)
    combined = pd.get_dummies(
        combined,
        columns=["firmware", "seller_type"],
        dtype=float,
    )
    combined = combined.replace([np.inf, -np.inf], np.nan)
    medians = combined.iloc[: len(train)].median(numeric_only=True)
    combined = combined.fillna(medians).fillna(0)

    return (
        combined.iloc[: len(train)].copy(),
        combined.iloc[len(train) :].copy(),
    )


def score_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float, np.ndarray]:
    probability = model.predict_proba(x_test)[:, 1]
    return (
        roc_auc_score(y_test, probability),
        average_precision_score(y_test, probability),
        probability,
    )


def threshold_table(y_true: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.05, 0.65, 0.05):
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        false_positive_rate = fp / (fp + tn) if fp + tn else 0

        # Explicit assumptions for the decision layer, not learned parameters.
        prevented = round(tp * 0.72)
        support_contacts_avoided = round(prevented * 0.60)

        rows.append(
            {
                "threshold": threshold,
                "failures_detected": tp,
                "healthy_terminals_flagged": fp,
                "precision": precision,
                "recall": recall,
                "false_positive_rate": false_positive_rate,
                "interventions": tp + fp,
                "estimated_failures_prevented": prevented,
                "estimated_support_contacts_avoided": support_contacts_avoided,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df = add_history_features(simulate_terminal_fleet())
    df = df[df["day"] >= 6].copy()

    history_features = SNAPSHOT_FEATURES + [
        column
        for column in df.columns
        if any(
            token in column
            for token in [
                "_7d_mean",
                "_14d_mean",
                "_30d_mean",
                "wifi_delta_",
                "timeout_delta_",
            ]
        )
    ]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_index, test_index = next(
        splitter.split(df, groups=df["terminal_id"].to_numpy())
    )
    train = df.iloc[train_index].copy()
    test = df.iloc[test_index].copy()
    y_train = train["failure_next_24h"]
    y_test = test["failure_next_24h"]

    results = []
    history_probability = None

    for label, features in [
        ("snapshot", SNAPSHOT_FEATURES),
        ("snapshot + history", history_features),
    ]:
        x_train, x_test = prepare_matrix(train, test, features)

        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2_000),
        )
        logistic.fit(x_train, y_train)
        roc_auc, pr_auc, _ = score_model(logistic, x_test, y_test)
        results.append(
            {
                "model": "Logistic regression",
                "features": label,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

        xgboost = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.90,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
            reg_lambda=2,
            min_child_weight=3,
        )
        xgboost.fit(x_train, y_train)
        roc_auc, pr_auc, probability = score_model(xgboost, x_test, y_test)
        results.append(
            {
                "model": "XGBoost",
                "features": label,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

        if label == "snapshot + history":
            history_probability = probability

    print("\nModel evaluation")
    print(pd.DataFrame(results).round(3).to_string(index=False))
    print(f"\nHoldout terminal-days: {len(test):,}")
    print(f"Holdout failure prevalence: {y_test.mean():.2%}")

    if history_probability is not None:
        print("\nIntervention thresholds — XGBoost with history")
        print(
            threshold_table(y_test.to_numpy(), history_probability)
            .round(4)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
