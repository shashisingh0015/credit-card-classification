"""The shared preprocessing contract for all six models.

Why this is its own module: every model in M2/M3 wraps *this same*
ColumnTransformer inside its own Pipeline. That gives us two things.

1. A fair comparison. All six models see identically prepared features, so
   differences in the metrics table reflect the algorithms, not accidental
   differences in preprocessing.

2. Leakage becomes structurally impossible. Because the scaler lives inside the
   Pipeline, `fit` only ever sees training rows -- including inside each
   cross-validation fold. Scaling the full dataset up front is the classic
   version of this mistake: the test set's mean and variance bleed into training
   and the reported scores come out optimistically biased.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    """Return the ColumnTransformer shared by every model pipeline.

    Emits ``config.N_FEATURES_OUT`` (29) columns:
      - 14 NUMERIC  -> StandardScaler
      -  6 ORDINAL  -> StandardScaler (kept as ordered integers; see config)
      -  3 NOMINAL  -> OneHotEncoder  -> 9 columns

    Scaling is applied uniformly rather than per-model. kNN and Logistic
    Regression genuinely require it -- unscaled, LIMIT_BAL (up to 1,000,000)
    would dominate every distance calculation while AGE (21..79) contributed
    almost nothing. The tree-based models are invariant to any monotonic
    rescaling, so applying it to them changes their splits not at all, and
    keeping one uniform preprocessor is simpler than special-casing three
    models.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC),
            ("ord", StandardScaler(), config.ORDINAL),
            (
                "nom",
                # handle_unknown="ignore" so an uploaded CSV containing an
                # unexpected category encodes as all-zeros instead of raising.
                # The Streamlit app accepts arbitrary user files, so this is a
                # real path, not defensive padding.
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.NOMINAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
