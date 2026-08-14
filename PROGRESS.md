# Progress Log

**How to use this file:** at the start of a session, read it. At the end of a
session, update the checkboxes and append a "Session N" entry. This is the mutable
counterpart to `CLAUDE.md` (which holds only stable facts).

**Resume prompt** — paste this to start any new session:

> Read CLAUDE.md, ROADMAP.md and PROGRESS.md, then continue from the next
> unchecked milestone.

---

## Milestone status

- [x] **M0 — Foundation & session persistence**
- [ ] **M1 — Data layer & EDA**
- [ ] **M2 — Baselines: Logistic Regression, Decision Tree, Naive Bayes**
- [ ] **M3 — kNN, Random Forest, Gradient Boosting**
- [ ] **M4 — Evaluation harness & comparison table**
- [ ] **M5 — Streamlit app**
- [ ] **M6 — CI/CD & deployment**
- [ ] **M7 — Submission package (README + PDF + Lab screenshot)**

## Blocked / needs the user

- [ ] **Create the GitHub repo.** You have an account but no repo yet, and the
      `gh` CLI is not installed on this machine. Either install GitHub CLI
      (`winget install GitHub.cli`) so I can create and push it, or create the
      repo in the browser and give me the URL.
- [ ] **BITS Virtual Lab access** — confirm you can log in, well before M7. This
      is 1 mark and the only step that cannot be done locally.

---

## Session 1 — 2026-08-15

**Done**
- Profiled the originally-supplied `card_fraud.csv` (PaySim, at
  `../archive/card_fraud.csv`): 6,362,620 rows × 11 columns, 493 MB.
  Rejected it — see "Key decisions" below.
- Selected and verified UCI id=350 (Default of Credit Card Clients):
  30,000 × 23, binary, 22.12% positive, 0 nulls. Download confirmed working.
- Recorded the `X1..X23` → semantic column mapping in `CLAUDE.md`.
- Created `.venv/` with scikit-learn 1.9.0, pandas 3.0.5, numpy 2.5.2,
  streamlit 1.61.1, matplotlib 3.11.1, seaborn 0.13.2, joblib, ucimlrepo.
- Wrote `CLAUDE.md`, `ROADMAP.md`, `PROGRESS.md`, `.gitignore`,
  `requirements.txt`. Initialised the git repo.

**Key decisions**
1. **Dropped `card_fraud.csv`.** Three independent blockers: only 10 usable
   features vs the mandated 12; 493 MB exceeds GitHub's 100 MB file limit; and at
   0.13% positive, Accuracy is meaningless and most of the 6 models degenerate to
   predicting the majority class. Replaced with UCI id=350.
2. **6 models, not 5** — the PDF says "6" but lists 5, so we add Gradient Boosting.
3. **Develop locally, run once in BITS Lab** for the screenshot only.
4. **Pin `scikit-learn` exactly** in requirements.txt, because joblib pickles are
   not portable across sklearn minor versions.

**Next session — start here**
M1. Build `model/data_prep.py` and `model/eda.ipynb`. Enter plan mode first so the
preprocessing approach gets reviewed before any code is written.
