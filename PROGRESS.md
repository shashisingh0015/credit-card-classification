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
- [x] **M1 — Data layer & EDA**
- [ ] **M2 — Baselines: Logistic Regression, Decision Tree, Naive Bayes**
- [ ] **M3 — kNN, Random Forest, Gradient Boosting**
- [ ] **M4 — Evaluation harness & comparison table**
- [ ] **M5 — Streamlit app**
- [ ] **M6 — CI/CD & deployment**
- [ ] **M7 — Submission package (README + PDF + Lab screenshot)**

## Blocked / needs the user

- [x] **GitHub repo** — created and connected manually by the user (2026-08-15).
      Remote: https://github.com/shashisingh0015/shashisingh0015-fraud-classification-beta
      `main` tracks `origin/main`; M0 commits pushed.
- [x] **BITS Virtual Lab access** — user confirmed access and will run the final
      app there to capture the M7 screenshot.
- [ ] **Consider renaming the repo** before M6 deploys to Streamlit Cloud. The
      current name says "fraud-classification-beta", but the project is credit-card
      *default* prediction (we dropped the fraud dataset). Renaming after the
      Streamlit app is connected means reconnecting it, so now is cheaper.
      Suggested: `credit-default-classification`. Purely cosmetic — not blocking.

## Notes

- `gh` CLI is not installed; all GitHub operations are done manually by the user
  or via plain `git` over HTTPS.

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

---

## Session 2 — 2026-08-15 (M1)

Planned in **plan mode** first (plan saved at
`~/.claude/plans/elegant-brewing-knuth.md`), then implemented.

**Built**
- `model/config.py` — single source of truth: paths, `RANDOM_STATE=42`, the
  `X1..X23` rename map, and the `NUMERIC`/`ORDINAL`/`NOMINAL` column groups.
- `model/preprocess.py` — `build_preprocessor()` → `ColumnTransformer`
  (StandardScaler on 14 numeric + 6 ordinal, OneHotEncoder on 3 nominal → **29
  output columns**). Every model pipeline in M2+ wraps this same object.
- `model/data_prep.py` — fetch → rename → fold undocumented codes → dedupe →
  stratified 80/20 split.
- `model/viz.py` — shared palette/style. CVD-validated categorical pair
  (blue/orange, worst ΔE 24.7 protan / 33.6 normal), sequential blue ramp,
  diverging blue↔red with a neutral grey midpoint. Reused by the M5 app.
- `model/eda.ipynb` — 22 cells, executed with outputs and 6 figures embedded.
- `reports/figures/*.png` — 6 figures for the README.
- `tests/test_data_prep.py` + root `conftest.py` — **13 tests, all passing.**
  These become the M6 CI gate.

**Data outcome**
30,000 → **29,965** rows after dropping 35 exact duplicates.
Split: **23,972 train / 5,993 test**, positive rate **22.13%** in all three
(stratification verified by test). `test_data.csv` = 536 KB. Zero nulls.

**Findings worth carrying forward**
1. **`PAY_0` risk is not monotonic** — codes −2/−1/0 sit at 13.2/17.0/12.6%, so
   "revolving" is *lower* risk than "paid in full". The real signal is a threshold
   at `PAY_0 >= 1` (13.8% vs 50.5%, 3.7×). Recorded in `CLAUDE.md`. This predicts
   **Logistic Regression will trail the tree models**; M4 tests that.
2. **21 contradictory records** — identical features, opposite labels (56
   feature-duplicates vs 35 full-row). Sets a small irreducible error floor, so
   100% accuracy is unattainable.
3. Feature ranges span ~10⁵ → scaling mandatory for kNN/LogReg, irrelevant to trees.
4. `BILL_AMT*` collinear at 0.80–0.95 → don't over-read LogReg coefficients.

**Corrected mid-session:** the first draft of the notebook claimed the `PAY_0`
default rate "rises monotonically". Rendering the chart disproved it. Title,
narrative and `CLAUDE.md` all now state the threshold structure instead.

**Gotchas hit (both recorded in `CLAUDE.md`)**
- `pip install jupyter` fails here — JupyterLab extension paths exceed the Windows
  path limit under this OneDrive folder. Use nbformat/nbclient/nbconvert/ipykernel.
- `\n` escapes inside notebook cell source collapse to real newlines across a
  write/execute/re-read cycle, breaking f-strings. Use a bare `print()`.

**Next session — start here**
**M2.** Build `model/models.py`: Logistic Regression, Decision Tree and Gaussian
Naive Bayes pipelines on top of `build_preprocessor()`. Decisions already locked:
default class weights (fair 6-way comparison), light hand-picked hyperparameters.
Claude Code features to learn in M2: **custom slash commands**
(`.claude/commands/train.md`) and **hooks** (`.claude/settings.json`).
