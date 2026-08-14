"""Single source of truth for dataset paths, column groups and constants.

Every other module imports from here so that no column list is ever duplicated.
If a column is renamed or regrouped, this is the only file that changes.

Dataset: UCI id=350, "Default of Credit Card Clients" (Taiwan, 2005).
30,000 clients, 23 features, binary target.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths. Anchored to the project root so scripts work from any cwd.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "model" / "artifacts"

TRAIN_CSV = DATA_DIR / "train.csv"
# Lives at the repo root and is committed: the assignment mandates this exact path.
TEST_CSV = PROJECT_ROOT / "test_data.csv"

UCI_DATASET_ID = 350

# Fixed everywhere that accepts it, so every number in the README is regenerable.
RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET = "default_next_month"

# --------------------------------------------------------------------------
# ucimlrepo serves this dataset with opaque column names (X1..X23, Y).
# The real names live only in the metadata, so we rename on load.
# --------------------------------------------------------------------------
RENAME_MAP = {
    "X1": "LIMIT_BAL",   # credit limit, NT dollars
    "X2": "SEX",         # 1 = male, 2 = female
    "X3": "EDUCATION",   # 1 = grad school, 2 = university, 3 = high school, 4 = others
    "X4": "MARRIAGE",    # 1 = married, 2 = single, 3 = others
    "X5": "AGE",         # years
    # Repayment status. Note there is no PAY_1 -- that gap is upstream naming,
    # not a mistake here. PAY_0 is the most recent month (September 2005),
    # PAY_6 the oldest (April 2005).
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    # Bill statement amount per month, most recent first.
    **{f"X{11 + i}": f"BILL_AMT{i}" for i in range(1, 7)},
    # Amount actually paid per month, most recent first.
    **{f"X{17 + i}": f"PAY_AMT{i}" for i in range(1, 7)},
    "Y": TARGET,
}

# --------------------------------------------------------------------------
# Column groups -- these drive the ColumnTransformer in preprocess.py.
# --------------------------------------------------------------------------

# Continuous money/age values. Spans ~5 orders of magnitude (LIMIT_BAL reaches
# 1,000,000 while AGE stays in 21..79), which is why scaling is mandatory for
# the distance- and gradient-based models.
NUMERIC = (
    ["LIMIT_BAL", "AGE"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)

# Repayment-status codes, treated as ordinal integers rather than one-hot.
#
# The UCI docs only describe -1 (paid duly) and 1..9 (months of delay), but the
# data also contains -2 and 0 -- and 0 is the single most common value, at
# roughly half of all rows. Actual semantics:
#     -2 = no credit used that month
#     -1 = paid in full
#      0 = revolving credit (paid the minimum, carrying a balance)
#   1..9 = months of payment delay
#
# That sequence is monotonic in risk, and the data agrees: as raw integers PAY_0
# correlates 0.325 with the target and PAY_2..PAY_6 take the next five places,
# making these the six strongest predictors in the dataset. One-hot encoding
# would throw that ordering away and add ~66 columns, which would hurt kNN
# through the curse of dimensionality.
ORDINAL = ["PAY_0"] + [f"PAY_{i}" for i in range(2, 7)]

# Genuinely unordered. EDUCATION looks ordinal across 1..3 (grad school >
# university > high school), but 4 = "others" breaks the ranking, so treating it
# as a number would invent an ordering that does not exist.
NOMINAL = ["SEX", "EDUCATION", "MARRIAGE"]

FEATURES = NUMERIC + ORDINAL + NOMINAL

# --------------------------------------------------------------------------
# Undocumented category codes, folded into the documented "others" bucket.
# EDUCATION {0, 5, 6} covers 345 rows (1.15%); MARRIAGE {0} covers 54 (0.18%).
# --------------------------------------------------------------------------
EDUCATION_OTHER = 4
MARRIAGE_OTHER = 3
EDUCATION_UNDOCUMENTED = [0, 5, 6]
MARRIAGE_UNDOCUMENTED = [0]

VALID_EDUCATION = {1, 2, 3, 4}
VALID_MARRIAGE = {1, 2, 3}

# Number of columns build_preprocessor() emits: 14 numeric + 6 ordinal, plus
# 9 one-hot columns (SEX 2 + EDUCATION 4 + MARRIAGE 3).
N_FEATURES_OUT = 29
