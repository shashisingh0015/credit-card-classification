# ML Assignment 2 — Classification Models + Streamlit App

BITS WILP M.Tech (AIML). 15 marks. **Deadline: 18-Aug-2026 23:59.**

Read `PROGRESS.md` at the start of every session to see where we left off.
Read `ROADMAP.md` for the full milestone plan.

## Deliverables (graded)

| Item | Marks |
|---|---|
| Dataset description in README | 1 |
| GitHub repo with all required files | 1 |
| 6 models × 6 metrics (comparison table) | 5 |
| Per-model performance observations | 3 |
| Streamlit: CSV upload | 1 |
| Streamlit: model dropdown | 1 |
| Streamlit: metrics display | 1 |
| Streamlit: confusion matrix / classification report | 1 |
| BITS Virtual Lab screenshot | 1 |

Required repo layout (from the assignment PDF — do not rename):

```
app.py
requirements.txt
README.md
test_data.csv
model/           # training code for all models (*.py or *.ipynb)
```

## Dataset

**UCI "Default of Credit Card Clients" (Taiwan), UCI id=350.**
30,000 instances × 23 features. Binary target. No nulls.
Class balance: 23,364 non-default (77.88%) / 6,636 default (22.12%).

Satisfies the assignment minimums: 23 features (≥12), 30,000 instances (≥500).

**Gotcha:** `ucimlrepo` serves columns as opaque `X1..X23`. Always rename on load:

| Raw | Real name | Raw | Real name |
|---|---|---|---|
| X1 | LIMIT_BAL | X12–X17 | BILL_AMT1..BILL_AMT6 |
| X2 | SEX | X18–X23 | PAY_AMT1..PAY_AMT6 |
| X3 | EDUCATION | Y | default_next_month (target) |
| X4 | MARRIAGE | | |
| X5 | AGE | | |
| X6–X11 | PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 | | |

Note `PAY_0` — there is no `PAY_1`. That is upstream naming, not a bug.

Semantics worth remembering: `PAY_*` are repayment-status codes. UCI documents only
−1 (paid duly) and 1..9 (months late), but the data also holds **−2 (no credit
used) and 0 (revolving)** — and `0` is the most common value at ~49% of rows.
`EDUCATION` {0,5,6} and `MARRIAGE` {0} are likewise undocumented; both are folded
into each column's "others" bucket.

**Measured in M1 — `PAY_0` risk is NOT monotonic in the code.** Default rate by
code: −2 → 13.2%, −1 → 17.0%, 0 → 12.6%, 1 → 34.2%, 2 → 69.3%, 3 → 77.2%,
4 → 69.4%. The real structure is a *threshold* at `PAY_0 >= 1`: not-late 13.8% vs
late 50.5%, a 3.7× risk ratio. Kept ordinal anyway (one tree split captures the
threshold; one-hot would add ~66 columns and hurt kNN), but this predicts
**Logistic Regression will trail the tree models**, since a single coefficient
cannot represent the −2/−1/0 inversion. M4 should confirm or refute that.

## Environment

Windows, PowerShell. Python 3.13.14 in a local venv (`.venv/`, git-ignored).

```powershell
.\.venv\Scripts\python.exe -m model.data_prep   # regenerate train.csv + test_data.csv
.\.venv\Scripts\python.exe -m model.train       # fit all models -> model/artifacts/
.\.venv\Scripts\python.exe -m pytest tests -q   # contract tests (13)
.\.venv\Scripts\ruff.exe check model tests      # lint
.\.venv\Scripts\streamlit.exe run app.py        # run app locally (M5)
```

`/train` is a project slash command (`.claude/commands/train.md`) that wraps the
training run and flags suspicious metrics. A PostToolUse hook
(`.claude/hooks/format_python.py`) runs ruff on any `.py` file after an edit.
Note `jq` is **not** installed on this machine, so hook payloads are parsed with
Python rather than the usual `jq` one-liner.

**Hook gotcha:** that hook runs `ruff check --fix`, which deletes imports that are
unused *at the moment the edit lands*. Adding an import in one edit and its first
use in a later edit means the import is silently stripped in between, and the next
run fails with `NameError`. Add imports in the same edit as their first use.

Always invoke `.\.venv\Scripts\python.exe` explicitly — the system `python` has
none of the ML packages installed.

`model/` is a package, so run its scripts with `-m` (`python -m model.data_prep`),
not by path — the relative imports fail otherwise.

**Do not `pip install jupyter`.** The metapackage's JupyterLab extension paths
exceed the Windows path limit under this OneDrive directory and the install fails
partway. `nbformat`, `nbclient`, `nbconvert` and `ipykernel` are installed instead,
which is enough to execute notebooks headlessly.

Notebook `source` strings do not survive `\n` escapes through a
write/execute/re-read round-trip — they collapse into real newlines and break
f-strings. Use a bare `print()` for blank lines inside notebook code.

## Decisions already made

- **Dataset switched away from `card_fraud.csv`** (PaySim). It has only 10 usable
  features vs the mandated 12, is 493 MB (over GitHub's 100 MB limit), and is
  0.13% positive — so extreme that Accuracy is meaningless and most models
  collapse to the majority class. Credit Card Default avoids all three problems.
- **6 models, not 5.** The PDF says "all the 6 ML models" but lists only 5. We
  implement the 5 named (Logistic Regression, Decision Tree, kNN, Naive Bayes,
  Random Forest) **plus Gradient Boosting** to satisfy either reading.
- **BITS Lab strategy:** develop locally, then clone + run once in the Virtual Lab
  purely to capture the required screenshot.
- **`scikit-learn` is pinned exactly** in requirements.txt. Models are persisted
  with joblib, and unpickling across sklearn minor versions breaks. The training
  env and Streamlit Cloud must match.

## Conventions

- One `sklearn.pipeline.Pipeline` per model, with preprocessing **inside** it.
  This prevents test-set leakage and makes each model a single serializable unit.
- `random_state=42` everywhere that accepts it.
- Stratified splits always — the 22/78 imbalance must be preserved in the test set.
- Report **MCC and AUC as the headline metrics**, not Accuracy. A
  predict-everything-negative baseline scores 77.9% Accuracy here; quoting
  Accuracy alone hides a useless model.
- Never commit `.venv/`, raw downloaded data, or `*.joblib` larger than ~25 MB.

## Anti-plagiarism

The assignment explicitly checks for identical repo structure, variable names, and
copy-paste Streamlit templates across students. Keep naming and UI choices
specific to this project; commit incrementally so history shows real development.
