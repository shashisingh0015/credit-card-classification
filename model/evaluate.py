"""The six-metric x six-model comparison table -- the M4 deliverable, worth 5 of
the assignment's 15 marks.

Loads the committed `model/artifacts/*.joblib` pipelines rather than refitting.
Two reasons that matters, not just speed:

* **These are the exact artifacts `app.py` loads in M5.** Refitting here would
  score a *different* model than the one actually served, which could silently
  diverge if training were ever non-deterministic.
* `python -m model.train` already re-splits and re-fits; this module's only job is
  scoring what is already on disk against the committed test set.

Run with:  python -m model.evaluate

Writes reports/comparison_table.md and reports/comparison_table.csv. Both are
fully regenerable from the committed artifacts and test_data.csv, which is what
lets the table in README.md be reproduced rather than merely asserted.
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config
from .models import MODEL_BUILDERS, artifact_path

REPORT_DIR = config.PROJECT_ROOT / "reports"
MD_PATH = REPORT_DIR / "comparison_table.md"
CSV_PATH = REPORT_DIR / "comparison_table.csv"

# Column order doubles as report column order. Accuracy first because a reader
# expects it there, then precision/recall/f1 (the threshold-dependent family, in
# the order that builds up to F1), then the two ranking/reliability metrics this
# project treats as headline numbers -- see CLAUDE.md.
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "auc", "mcc"]


def load_test_split() -> tuple[pd.DataFrame, pd.Series]:
    if not config.TEST_CSV.exists():
        raise SystemExit(
            f"{config.TEST_CSV} not found -- run `python -m model.data_prep` first"
        )
    test = pd.read_csv(config.TEST_CSV)
    return test[config.FEATURES], test[config.TARGET]


def score_model(name: str, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    path = artifact_path(name)
    if not path.exists():
        raise SystemExit(f"{path} missing -- run `python -m model.train` first")
    pipe = joblib.load(path)

    y_pred = pipe.predict(X_test)
    # AUC needs scores, not hard labels -- it measures ranking quality across
    # every threshold, so a 0/1 vector would throw the information away.
    y_score = pipe.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        # zero_division=0 rather than the sklearn default (which warns and
        # returns 0 anyway): a model that predicts no positives at all is a real,
        # sometimes-informative outcome here, not a code error to surface as a
        # warning on every run.
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_score),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }


def to_markdown(rows: list[dict]) -> str:
    header = "| model | " + " | ".join(METRIC_COLUMNS) + " |"
    sep = "|---|" + "---|" * len(METRIC_COLUMNS)
    lines = [header, sep]
    for row in rows:
        cells = " | ".join(f"{row[m]:.4f}" for m in METRIC_COLUMNS)
        lines.append(f"| {row['model']} | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    X_test, y_test = load_test_split()
    rows = [score_model(name, X_test, y_test) for name in MODEL_BUILDERS]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    markdown = to_markdown(rows)
    MD_PATH.write_text(markdown, encoding="utf-8")

    frame = pd.DataFrame(rows).set_index("model")
    frame.to_csv(CSV_PATH)

    print(markdown)

    best_mcc = max(rows, key=lambda r: r["mcc"])
    best_auc = max(rows, key=lambda r: r["auc"])
    print(f"Best MCC: {best_mcc['model']} ({best_mcc['mcc']:.4f})")
    print(f"Best AUC: {best_auc['model']} ({best_auc['auc']:.4f})")
    if best_mcc["model"] != best_auc["model"]:
        print(
            "MCC and AUC disagree on the winner -- worth a sentence in the "
            "README observations rather than picking one silently."
        )

    print(
        f"\nWritten: {MD_PATH.relative_to(config.PROJECT_ROOT)}, "
        f"{CSV_PATH.relative_to(config.PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
