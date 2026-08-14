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

Semantics worth remembering: `PAY_*` are repayment-status codes (−1 = paid duly,
1..9 = months of delay), so they are **ordinal, not continuous**. `EDUCATION` and
`MARRIAGE` contain undocumented codes (0, 5, 6) that need explicit handling.

## Environment

Windows, PowerShell. Python 3.13.14 in a local venv (`.venv/`, git-ignored).

```powershell
.\.venv\Scripts\python.exe -m pytest        # tests
.\.venv\Scripts\python.exe model\train.py   # retrain all models
.\.venv\Scripts\streamlit.exe run app.py    # run app locally
```

Always invoke `.\.venv\Scripts\python.exe` explicitly — the system `python` has
none of the ML packages installed.

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
