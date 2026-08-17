"""Streamlit app -- M5. CSV upload, model dropdown, metrics panel, confusion
matrix and classification report, all served from the artifacts M2/M3 already
committed.

Serving vs training skew is the concept this milestone is really about: an
uploaded CSV must go through the *exact* preprocessing each model was trained
with, or the metrics reported here would not mean what they claim to. Because
`build_preprocessor()` lives inside every pipeline (see `model/preprocess.py`),
that is structurally guaranteed rather than something this file has to get
right on its own -- `pipe.predict` re-applies the fitted scaler and encoder
before the estimator ever sees the data.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model import config, viz
from model.evaluate import METRIC_COLUMNS, score_model
from model.models import MODEL_BUILDERS, artifact_path

st.set_page_config(
    page_title="Credit Default Classifier",
    page_icon=":bar_chart:",
    layout="wide",
)

viz.apply_style()

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "auc": "AUC-ROC",
    "mcc": "MCC",
}


# ----------------------------------------------------------------------------
# Caching. @st.cache_resource for the fitted pipeline objects -- they are not
# serializable data, and reloading a joblib file on every widget interaction
# would make the app noticeably laggy. @st.cache_data for the uploaded
# dataframe, keyed on the file's bytes, so re-selecting the same model doesn't
# re-parse the CSV.
# ----------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_pipeline(model_name: str):
    path = artifact_path(model_name)
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_bundled_test_set() -> pd.DataFrame:
    return pd.read_csv(config.TEST_CSV)


@st.cache_data(show_spinner=False)
def parse_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def validate_schema(df: pd.DataFrame) -> tuple[list[str], bool]:
    """Return (missing feature columns, whether the target column is present).

    The pipelines tolerate *extra* columns and unseen categorical values fine
    (`OneHotEncoder(handle_unknown="ignore")` in `preprocess.py` sees to that),
    but a missing feature column is fatal -- the ColumnTransformer selects
    columns by name and raises. Checking here turns that into a readable
    message instead of a stack trace.
    """
    missing = [c for c in config.FEATURES if c not in df.columns]
    has_target = config.TARGET in df.columns
    return missing, has_target


def render_confusion_matrix(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap=viz.CMAP_SEQ)

    labels = [viz.CLASS_LABELS[0], viz.CLASS_LABELS[1]]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix", loc="left")

    # Text colour flips against the cell's own fill so counts stay legible from
    # the palest cell (usually the minority-class corner) to the darkest.
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            colour = viz.SURFACE if cm[i, j] > threshold else viz.INK_PRIMARY
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=colour)

    fig.tight_layout()
    st.pyplot(fig, width="content")


def render_classification_report(y_true, y_pred) -> None:
    report = classification_report(
        y_true,
        y_pred,
        target_names=[viz.CLASS_LABELS[0], viz.CLASS_LABELS[1]],
        output_dict=True,
        zero_division=0,
    )
    frame = pd.DataFrame(report).transpose()
    frame["support"] = frame["support"].astype(int)
    st.dataframe(frame.style.format(precision=3), width="stretch")


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

st.sidebar.header("Configuration")

model_name = st.sidebar.selectbox(
    "Model",
    options=list(MODEL_BUILDERS),
    help="All six were trained on the same 23,972-row split with the same "
    "preprocessing -- see model/models.py for what each one's docstring "
    "already found.",
)

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV to score",
    type="csv",
    help=f"Must contain the {len(config.FEATURES)} feature columns "
    f"(see below). Include `{config.TARGET}` too if you want metrics, "
    "a confusion matrix and a classification report -- without it you'll "
    "get predictions only.",
)

with st.sidebar.expander("Required columns"):
    st.code(", ".join(config.FEATURES), language=None)

st.sidebar.caption(
    "No file uploaded yet? The panel below defaults to the committed "
    f"`{config.TEST_CSV.name}` ({config.TEST_CSV.stat().st_size / 1024:.0f} KB, "
    "6,000 held-out rows the models never trained on)."
)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

st.title("Credit Card Default Classifier")
st.caption(
    "UCI “Default of Credit Card Clients” (Taiwan) — 6 models, "
    "6 metrics, one shared preprocessing pipeline."
)

if uploaded_file is not None:
    df = parse_uploaded_csv(uploaded_file.getvalue())
    source_label = f"uploaded file `{uploaded_file.name}`"
else:
    df = load_bundled_test_set()
    source_label = f"bundled `{config.TEST_CSV.name}`"

missing_features, has_target = validate_schema(df)

if missing_features:
    st.error(
        f"This CSV is missing {len(missing_features)} required feature "
        f"column(s): {', '.join(missing_features)}. Scoring needs all "
        f"{len(config.FEATURES)} columns listed in the sidebar."
    )
    st.stop()

st.write(f"Scoring **{len(df):,} rows** from {source_label} with **{model_name}**.")
if has_target:
    positive_rate = df[config.TARGET].mean()
    st.caption(f"Positive rate in this file: {positive_rate:.2%}")

pipeline = load_pipeline(model_name)
X = df[config.FEATURES]

if has_target:
    y_true = df[config.TARGET]
    metrics = score_model(model_name, X, y_true)
    y_pred = pipeline.predict(X)

    st.subheader("Metrics")
    baseline = 1 - y_true.mean()
    st.caption(
        f"All-negative baseline on this file: {baseline:.2%} accuracy. "
        "MCC and AUC are the metrics to trust here -- accuracy alone can look "
        "high while the model has learned nothing."
    )
    cols = st.columns(len(METRIC_COLUMNS))
    for col, metric in zip(cols, METRIC_COLUMNS):
        col.metric(METRIC_LABELS[metric], f"{metrics[metric]:.4f}")

    left, right = st.columns([1, 1.4])
    with left:
        render_confusion_matrix(y_true, y_pred)
    with right:
        st.subheader("Classification report")
        render_classification_report(y_true, y_pred)
else:
    st.warning(
        f"No `{config.TARGET}` column in this file, so accuracy, precision, "
        "recall, F1, AUC, MCC, the confusion matrix and the classification "
        "report all need ground-truth labels that aren't here. Showing "
        "predictions only."
    )
    y_pred = pipeline.predict(X)
    y_score = pipeline.predict_proba(X)[:, 1]
    st.subheader("Predictions")
    st.dataframe(
        pd.DataFrame(
            {
                "predicted_default": y_pred,
                "predicted_probability": np.round(y_score, 4),
            }
        ),
        width="stretch",
    )
