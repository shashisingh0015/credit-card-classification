"""Contract tests for the M4 comparison table.

Guards the invariant that matters most for a 5-mark deliverable: the table
actually has all 6 models and all 6 metrics, and every number is in its valid
range. Requires `python -m model.train` to have written the artifacts first.
"""

from __future__ import annotations

import pytest

from model import config
from model.evaluate import METRIC_COLUMNS, load_test_split, score_model
from model.models import MODEL_BUILDERS, artifact_path


@pytest.fixture(scope="module")
def rows():
    for name in MODEL_BUILDERS:
        if not artifact_path(name).exists():
            pytest.skip("run `python -m model.train` first")
    if not config.TEST_CSV.exists():
        pytest.skip("run `python -m model.data_prep` first")
    X_test, y_test = load_test_split()
    return [score_model(name, X_test, y_test) for name in MODEL_BUILDERS]


def test_all_six_models_scored(rows):
    assert {r["model"] for r in rows} == set(MODEL_BUILDERS)
    assert len(rows) == 6


def test_all_six_metrics_present(rows):
    assert len(METRIC_COLUMNS) == 6
    for row in rows:
        assert set(METRIC_COLUMNS) <= set(row)


def test_metrics_in_valid_range(rows):
    for row in rows:
        for metric in ("accuracy", "precision", "recall", "f1", "auc"):
            assert 0.0 <= row[metric] <= 1.0, f"{row['model']}.{metric}"
        assert -1.0 <= row["mcc"] <= 1.0, row["model"]


def test_no_model_matches_the_do_nothing_baseline_on_auc(rows):
    # AUC 0.5 means the model ranks no better than a coin flip. Every model here
    # is expected to have learned something real from the data.
    for row in rows:
        assert row["auc"] > 0.55, row["model"]
