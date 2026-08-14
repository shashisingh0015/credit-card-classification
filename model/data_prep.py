"""Fetch, clean and split the UCI Credit Card Default dataset.

Run with:  python -m model.data_prep

Writes two files:
  data/train.csv   -- git-ignored, regenerated deterministically from UCI
  test_data.csv    -- repo root, committed (the assignment mandates this path)

Both include the target column. The Streamlit app has to *display metrics* on the
uploaded file, and you cannot compute Accuracy or MCC without ground truth, so
stripping the label from the test CSV would break the app's whole purpose.

The script is idempotent: the same seed and the same cleaning rules mean rerunning
it reproduces byte-identical output.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw() -> pd.DataFrame:
    """Download the dataset from UCI and give the columns their real names."""
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=config.UCI_DATASET_ID)
    # ucimlrepo hands back features and targets separately; we want one frame.
    df = pd.concat([repo.data.features, repo.data.targets], axis=1)
    return df.rename(columns=config.RENAME_MAP)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fold undocumented category codes and drop duplicate rows.

    Two distinct problems are handled here.

    Undocumented categories: EDUCATION contains 0, 5 and 6 and MARRIAGE contains
    0, none of which appear in the UCI documentation. Together that is ~1.3% of
    rows. Rather than drop them we fold each into that column's documented
    "others" bucket, which is what the code is for.

    Duplicates: 56 rows are exact duplicates. Left in place, the same row can land
    in both the train and test split, letting a model memorise a test answer and
    biasing the reported metrics upward. The count is trivial but the fix is free,
    so we deduplicate *before* splitting.
    """
    out = df.copy()

    out["EDUCATION"] = out["EDUCATION"].replace(
        config.EDUCATION_UNDOCUMENTED, config.EDUCATION_OTHER
    )
    out["MARRIAGE"] = out["MARRIAGE"].replace(
        config.MARRIAGE_UNDOCUMENTED, config.MARRIAGE_OTHER
    )

    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        print(f"  dropped {dropped} exact duplicate rows ({before} -> {len(out)})")

    # Column order is part of the contract the app relies on when validating
    # uploads, so pin it explicitly instead of inheriting UCI's ordering.
    return out[config.FEATURES + [config.TARGET]]


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split.

    Stratifying matters because the positive class is only ~22%. On an
    unstratified split the test-set positive rate drifts by a percentage point or
    more between seeds, which quietly moves every threshold-dependent metric and
    makes the README table irreproducible.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[config.TARGET],
    )
    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _summarise(name: str, df: pd.DataFrame) -> None:
    rate = df[config.TARGET].mean()
    print(
        f"  {name:<6} {len(df):>6,} rows  "
        f"{int(df[config.TARGET].sum()):>5,} positive  ({rate:.2%})"
    )


def main() -> None:
    print("Fetching UCI dataset id=%d ..." % config.UCI_DATASET_ID)
    raw = load_raw()
    print(f"  raw shape: {raw.shape}")

    print("Cleaning ...")
    df = clean(raw)

    nulls = int(df.isna().sum().sum())
    print(f"  nulls: {nulls}")
    print(f"  EDUCATION values: {sorted(df['EDUCATION'].unique())}")
    print(f"  MARRIAGE  values: {sorted(df['MARRIAGE'].unique())}")

    print("Splitting (stratified 80/20) ...")
    train_df, test_df = split(df)

    print("Class balance:")
    _summarise("all", df)
    _summarise("train", train_df)
    _summarise("test", test_df)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.TRAIN_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)

    print("Wrote:")
    for path in (config.TRAIN_CSV, config.TEST_CSV):
        print(f"  {path.relative_to(config.PROJECT_ROOT)}  "
              f"({path.stat().st_size / 1024:,.0f} KB)")

    # The number every model in M4 must beat. A classifier that always predicts
    # "no default" scores this on accuracy while being completely useless --
    # its recall, F1 and MCC are all 0 and its AUC is 0.5.
    majority = 1 - df[config.TARGET].mean()
    print(f"\nMajority-class baseline accuracy: {majority:.4%}")
    print("  (same model scores Recall 0.0, F1 0.0, MCC 0.0, AUC 0.5)")


if __name__ == "__main__":
    main()
