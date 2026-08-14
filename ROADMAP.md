# Roadmap

Eight milestones. Each is sized to fit **one working session** and ends in a
committable state, so you can always stop at a milestone boundary without losing
context.

Every milestone has three parts:
- **Build** — what we produce
- **ML concepts** — the theory we cover while building it (assignment goal #1)
- **Claude Code** — the tool feature we learn by using it here (assignment goal #2)

---

## M0 — Foundation & session persistence ✅ (done)

**Build:** venv + dependencies, dataset availability verified, `CLAUDE.md`,
`ROADMAP.md`, `PROGRESS.md`, `.gitignore`, `requirements.txt`, git repo initialised.

**ML concepts:** How to vet a dataset *before* modelling — feature count, instance
count, class balance, nulls, and which columns are unusable (IDs, leaky flags).
This is why we caught that PaySim was the wrong choice in minutes rather than
after writing the models.

**Claude Code:** The **memory hierarchy**. Four levels, each with a different job:

| File | Scope | Loaded | Use for |
|---|---|---|---|
| `~/.claude/CLAUDE.md` | You, everywhere | Every session | Personal style prefs |
| `./CLAUDE.md` | This project | Every session | Project facts, commands, decisions |
| `./PROGRESS.md` | This project | When referenced | Mutable session log |
| `~/.claude/projects/<slug>/memory/` | This project | On recall | Durable cross-session facts |

The distinction that matters: **`CLAUDE.md` is for stable facts** (the dataset
schema, the venv path) — things that stay true. **`PROGRESS.md` is for mutable
state** (what's done, what's next). Mixing them means `CLAUDE.md` churns and stops
being trustworthy.

---

## M1 — Data layer & EDA

**Build:** `model/data_prep.py` — download, rename `X1..X23` → semantic names,
handle undocumented `EDUCATION`/`MARRIAGE` codes, stratified 80/20 split, write
`test_data.csv` (the 6,000-row test set, ~600 KB — safe for GitHub and Streamlit).
Plus `model/eda.ipynb` with distributions, correlation heatmap, class balance.

**ML concepts:**
- **Why stratify.** With 22% positives, a random split can drift the test-set
  positive rate by several points, making metrics unreproducible.
- **Train/test leakage.** Why the scaler must be fit on train only, and why
  putting it inside a `Pipeline` makes that structurally impossible to get wrong.
- **Ordinal vs nominal vs continuous.** `PAY_*` are ordinal codes; `EDUCATION` is
  nominal. Treating a nominal code as a number invents a false ordering.
- **Correlation structure.** `BILL_AMT1..6` are highly collinear — this hurts
  Logistic Regression's coefficient interpretability but barely affects trees.

**Claude Code:** **Plan mode.** Before writing `data_prep.py`, we enter plan mode
so I investigate and propose an approach *without editing files*, you approve it,
and only then do I build. This is the main guard against an agent confidently
writing 200 lines in the wrong direction. Also: **checkpointing** — `Esc Esc`
rewinds the conversation and undoes edits, so a bad refactor costs seconds.

---

## M2 — Baseline models: Logistic Regression, Decision Tree, Naive Bayes

**Build:** `model/models.py` with one `Pipeline` factory per model. Train, persist
to `model/artifacts/*.joblib`.

**ML concepts:**
- **Logistic Regression** — linear decision boundary in log-odds space. The
  sigmoid, why it needs feature scaling (gradient descent on wildly different
  scales converges badly), and what `class_weight='balanced'` actually does to the
  loss. Interpretable coefficients are its main selling point.
- **Decision Tree** — recursive axis-aligned splits. Gini vs entropy, why an
  unconstrained tree hits ~100% train accuracy and overfits hard, and how
  `max_depth` / `min_samples_leaf` trade bias against variance. Needs no scaling.
- **Gaussian Naive Bayes** — the conditional-independence assumption, why it is
  badly violated here (`BILL_AMT*` are strongly correlated), and why the model is
  still often a decent *ranker* (good AUC) despite poorly calibrated
  probabilities. Expect this to be the weakest model — and be able to say why.

**Claude Code:** **Custom slash commands.** We create `.claude/commands/train.md`
so `/train` reruns training and prints the metrics table. Commands are just
markdown prompt templates in `.claude/commands/` — the filename is the command
name, `$ARGUMENTS` interpolates input. Then **hooks**: a `PostToolUse` hook in
`.claude/settings.json` that auto-formats any Python file after I edit it. Hooks
are executed by the harness, not by me, so they fire deterministically.

---

## M3 — Distance & ensemble models: kNN, Random Forest, Gradient Boosting

**Build:** the remaining three pipelines, same interface as M2.

**ML concepts:**
- **kNN** — a lazy learner storing the whole training set; cost is at predict
  time. Scaling is *mandatory* (`LIMIT_BAL` spans ~10⁶, `AGE` ~10¹ — unscaled,
  distance is entirely `LIMIT_BAL`). The **curse of dimensionality**: in 23
  dimensions, distances concentrate and neighbourhoods stop being local. Choosing
  `k` = bias/variance again.
- **Random Forest (bagging)** — bootstrap samples + random feature subsets per
  split, averaged. Decorrelating the trees is what makes the variance reduction
  work. Reduces variance, not bias; hard to overfit by adding trees.
- **Gradient Boosting** — sequential, each tree fitting the previous ensemble's
  residuals. Reduces **bias**; *can* overfit with too many trees. This
  bagging-vs-boosting contrast is the most valuable sentence in your observations
  table.

**Claude Code:** **Subagents.** Hyperparameter exploration is embarrassingly
parallel, so we dispatch several agents at once, each sweeping one model, each
reporting back only its conclusion — keeping my main context clean instead of
filling it with raw sweep logs. This is the core value: subagents have their own
context window, so exploration cost doesn't compound.

---

## M4 — Evaluation harness & the comparison table

**Build:** `model/evaluate.py` producing the 6×6 metrics table as markdown + CSV.

**ML concepts** (this milestone is worth 5 marks — the metrics are the deliverable):
- **Accuracy** — and its trap. The all-negative baseline scores **77.88%** here.
  Any model near that is worthless regardless of how good the number looks.
- **Precision** = TP/(TP+FP) — "when we predict default, how often are we right?"
  Cost of a false positive: wrongly denying credit to a good customer.
- **Recall** = TP/(TP+FN) — "of real defaulters, how many did we catch?" Cost of a
  false negative: an unrecovered loan. Usually the more expensive error, which is
  why the threshold is a business decision, not a statistical one.
- **F1** — harmonic mean of precision and recall. Harmonic, so it punishes
  imbalance between them; ignores true negatives entirely.
- **AUC-ROC** — threshold-*independent* ranking quality: P(random positive scored
  above random negative). Needs `predict_proba`, not `predict`. 0.5 = random.
- **MCC** — correlation between predicted and actual labels, using **all four**
  confusion-matrix cells. This is why it is the most honest single number on
  imbalanced data: unlike F1 it cannot be inflated by ignoring true negatives.
  Range −1 to +1, where 0 = no better than chance.
- **Threshold vs ranking metrics.** Accuracy/Precision/Recall/F1/MCC all depend on
  the 0.5 cutoff; AUC does not. A model can have great AUC and terrible F1 purely
  because 0.5 is the wrong threshold for a 22% base rate.

**Claude Code:** **Skills.** We author `.claude/skills/model-report/SKILL.md` — a
reusable procedure for regenerating the metrics table and observations in a fixed
format. Skills differ from slash commands in that they carry supporting files and
are **model-invoked** (I decide to use one from its `description`), whereas a
command is **user-invoked**. Then **iterative refinement**: I write the harness,
run it, read real numbers, and fix what the numbers reveal — several tight
loops rather than one big speculative implementation.

---

## M5 — Streamlit app

**Build:** `app.py` — CSV upload, model dropdown, metrics panel, confusion matrix
+ classification report. All 4 UI marks live here.

**ML concepts:** Serving vs training skew — the uploaded CSV must go through the
*exact* preprocessing the model was trained on, which is free if preprocessing is
inside the Pipeline. Also: validating uploaded input, and what to do when a user's
CSV has the wrong columns.

**Claude Code:** `@st.cache_resource` vs `@st.cache_data` (models vs dataframes),
and using the **`/run` skill** to actually launch the app and confirm the change
works in the real UI rather than assuming it does.

---

## M6 — CI/CD & deployment

**Build:** `.github/workflows/ci.yml` — on every push: install deps, run tests,
retrain, assert metrics haven't regressed. Then deploy to Streamlit Community
Cloud.

**ML concepts:** Reproducibility — pinned seeds and pinned sklearn mean the table
in your README is regenerable, which is exactly what the anti-plagiarism check
rewards. Plus the **sklearn-version/pickle trap**: this is the single most common
"works locally, 500s on Streamlit Cloud" failure.

**Claude Code:** **GitHub Actions integration**, and the review skills —
`/code-review` on the diff before pushing, `/security-review` before making the
repo public.

---

## M7 — Submission package

**Build:** `README.md` in the mandated a–e structure (problem statement, dataset
description, repo link, comparison table, observations table + overall winner).
BITS Virtual Lab run + screenshot. Assemble the final PDF in the required order:
GitHub link → Streamlit link → screenshot → README content.

**Claude Code:** Wrapping up — `/export` to save the transcript, and a final
`PROGRESS.md` update so the repo tells the whole story.

---

## Suggested session plan

| Session | Milestones | Rough effort |
|---|---|---|
| 1 (done) | M0 | — |
| 2 | M1 | ~1h |
| 3 | M2 + M3 | ~1.5h |
| 4 | M4 | ~1h |
| 5 | M5 | ~1h |
| 6 | M6 + M7 | ~1.5h |

Buffer is deliberate: the deadline is 18-Aug and Streamlit Cloud's first deploy is
the step most likely to surprise us.
