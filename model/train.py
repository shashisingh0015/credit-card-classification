"""Fit every registered model and persist it.

Run with:  python -m model.train

Writes one joblib file per model into model/artifacts/. Those artifacts are
committed so the Streamlit app can load fitted pipelines instantly instead of
retraining on every cold start -- which also means the app needs no training data,
keeping the deployed footprint to the test set only.

The metrics printed here are a sanity check, not the deliverable. The full six-metric
comparison table is M4's job (model/evaluate.py).
"""

from __future__ import annotations

import time

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from . import config
from .models import MODEL_BUILDERS, artifact_path


def load_splits() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if not config.TRAIN_CSV.exists():
        raise SystemExit(
            f"{config.TRAIN_CSV} not found -- run `python -m model.data_prep` first"
        )
    train = pd.read_csv(config.TRAIN_CSV)
    test = pd.read_csv(config.TEST_CSV)
    return (
        train[config.FEATURES],
        train[config.TARGET],
        test[config.FEATURES],
        test[config.TARGET],
    )


def main() -> None:
    X_train, y_train, X_test, y_test = load_splits()
    print(f"train {X_train.shape}   test {X_test.shape}")
    print(f"positive rate: train {y_train.mean():.2%}  test {y_test.mean():.2%}")

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = 1 - y_test.mean()
    print(f"\nall-negative baseline accuracy on test: {baseline:.2%}")
    print(
        "Any model near that number has learned nothing -- watch MCC, not accuracy.\n"
    )

    header = f"{'model':<24} {'acc':>7} {'auc':>7} {'f1':>7} {'mcc':>7} {'fit s':>7}"
    print(header)
    print("-" * len(header))

    rows = []
    for name, builder in MODEL_BUILDERS.items():
        pipe = builder()

        started = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        y_pred = pipe.predict(X_test)
        # AUC needs scores, not hard labels -- it measures ranking quality across
        # every possible threshold, so a 0/1 vector would throw the information away.
        y_score = pipe.predict_proba(X_test)[:, 1]

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_score),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_test, y_pred),
            "fit_seconds": fit_seconds,
        }
        rows.append(row)
        print(
            f"{name:<24} {row['accuracy']:>7.4f} {row['auc']:>7.4f} "
            f"{row['f1']:>7.4f} {row['mcc']:>7.4f} {fit_seconds:>7.2f}"
        )

        path = artifact_path(name)
        # compress=3 trades a little load time for a much smaller artifact, which
        # matters because these are committed to git.
        joblib.dump(pipe, path, compress=3)

    print("\nArtifacts written:")
    for name in MODEL_BUILDERS:
        path = artifact_path(name)
        print(
            f"  {path.relative_to(config.PROJECT_ROOT)}  "
            f"({path.stat().st_size / 1024:,.0f} KB)"
        )

    best = max(rows, key=lambda r: r["mcc"])
    print(f"\nBest MCC so far: {best['model']} ({best['mcc']:.4f})")


if __name__ == "__main__":
    main()
