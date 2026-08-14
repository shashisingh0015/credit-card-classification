"""Contract tests for the data layer.

These guard the invariants the rest of the project assumes. They become the CI
gate in M6, so a change that silently breaks the split or the feature contract
fails the build instead of quietly corrupting the metrics table.

Requires `python -m model.data_prep` to have been run first.
"""

from __future__ import annotations

import pandas as pd
import pytest

from model import config
from model.preprocess import build_preprocessor

# Stratification should hold almost exactly; allow half a percentage point of
# slack for the rounding inherent in splitting ~30k rows.
RATE_TOLERANCE_PP = 0.005


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    if not config.TRAIN_CSV.exists():
        pytest.skip("run `python -m model.data_prep` first")
    return pd.read_csv(config.TRAIN_CSV)


@pytest.fixture(scope="module")
def test() -> pd.DataFrame:
    if not config.TEST_CSV.exists():
        pytest.skip("run `python -m model.data_prep` first")
    return pd.read_csv(config.TEST_CSV)


def test_test_csv_is_committed_at_the_mandated_path():
    # The assignment specifies test_data.csv at the repo root by name.
    assert config.TEST_CSV.exists()
    assert config.TEST_CSV.name == "test_data.csv"
    assert config.TEST_CSV.parent == config.PROJECT_ROOT


def test_test_csv_stays_small_enough_for_streamlit_free_tier(test):
    # The assignment explicitly warns to upload test data only. Keep it well
    # under a megabyte so the hosted app stays responsive.
    assert config.TEST_CSV.stat().st_size < 1_000_000
    assert len(test) < 10_000


def test_both_splits_carry_the_target(train, test):
    # Without ground truth the app cannot compute any of the six metrics.
    assert config.TARGET in train.columns
    assert config.TARGET in test.columns


def test_column_contract_and_order(train, test):
    expected = config.FEATURES + [config.TARGET]
    assert list(train.columns) == expected
    assert list(test.columns) == expected
    # 23 features, comfortably above the assignment's minimum of 12.
    assert len(config.FEATURES) == 23


def test_no_nulls(train, test):
    assert int(train.isna().sum().sum()) == 0
    assert int(test.isna().sum().sum()) == 0


def test_stratification_preserved(train, test):
    train_rate = train[config.TARGET].mean()
    test_rate = test[config.TARGET].mean()
    combined = pd.concat([train, test])[config.TARGET].mean()

    assert abs(train_rate - combined) < RATE_TOLERANCE_PP
    assert abs(test_rate - combined) < RATE_TOLERANCE_PP
    # Sanity-check we are still looking at the dataset we think we are.
    assert 0.21 < combined < 0.23


def test_split_proportions(train, test):
    ratio = len(test) / (len(train) + len(test))
    assert abs(ratio - config.TEST_SIZE) < 0.01


def test_no_row_appears_in_both_splits(train, test):
    # Deduplicating before splitting means a test row cannot also be a train row.
    # If this fails, a model can memorise a test answer and the metrics inflate.
    overlap = pd.merge(train, test, how="inner")
    assert len(overlap) == 0


def test_undocumented_category_codes_were_folded(train, test):
    for df in (train, test):
        assert set(df["EDUCATION"].unique()) <= config.VALID_EDUCATION
        assert set(df["MARRIAGE"].unique()) <= config.VALID_MARRIAGE


def test_ordinal_pay_columns_are_within_documented_range(train):
    # -2..9 once the undocumented -2 and 0 codes are accounted for.
    for col in config.ORDINAL:
        assert train[col].between(-2, 9).all()


def test_preprocessor_emits_the_expected_width(train):
    X = train[config.FEATURES]
    out = build_preprocessor().fit_transform(X)
    assert out.shape == (len(train), config.N_FEATURES_OUT)


def test_preprocessor_scales_numeric_columns(train):
    # Confirms the scaler actually engaged: standardised columns have ~zero mean
    # and unit variance, unlike the raw values where LIMIT_BAL averages ~167,000.
    X = train[config.FEATURES]
    out = build_preprocessor().fit_transform(X)
    n_scaled = len(config.NUMERIC) + len(config.ORDINAL)
    scaled = out[:, :n_scaled]
    assert abs(scaled.mean()) < 1e-9
    assert abs(scaled.std() - 1.0) < 1e-6


def test_preprocessor_tolerates_unseen_categories(train):
    # The Streamlit app accepts arbitrary uploads, so an unexpected category
    # must encode as all-zeros rather than raising.
    X = train[config.FEATURES]
    pre = build_preprocessor().fit(X)

    unseen = X.head(5).copy()
    unseen.loc[:, "EDUCATION"] = 99
    out = pre.transform(unseen)
    assert out.shape == (5, config.N_FEATURES_OUT)
