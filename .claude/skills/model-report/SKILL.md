---
name: model-report
description: Regenerate the 6-model x 6-metric comparison table (model/evaluate.py) and write per-model performance observations in the project's fixed format. Use this whenever models.py, an artifact, or the metrics table changes and the README's comparison table or observations need to be brought back in sync.
---

# Model report

Produces the two graded M4/README artifacts together, in the format the rest of
this project already uses (see `PROGRESS.md` session entries for the reference
style): a metrics table and a short, numbers-first observation per model.

## Steps

1. Run `.\.venv\Scripts\python.exe -m model.evaluate` from the project root.
   - If it fails because an artifact is missing, run
     `.\.venv\Scripts\python.exe -m model.train` first, then retry.
   - If it fails because `test_data.csv` is missing, run
     `.\.venv\Scripts\python.exe -m model.data_prep` first, then retry.
2. Read `reports/comparison_table.md` — this is the table, already sorted in
   registry order (simple → complex) and already rounded to 4dp. Do not
   hand-edit it; re-run step 1 instead.
3. For each model, write one observation grounded in the actual numbers just
   produced, not in memorized numbers from a prior run. An observation is not
   complete until it has:
   - The model's own MCC and AUC (the two headline metrics per `CLAUDE.md` —
     accuracy alone is misleading here; the baseline scores 77.87%).
   - A comparison to at least one *other* model in the table, with numbers, not
     just adjectives ("beats X's MCC of 0.35" rather than "performs well").
   - A mechanism, not just an observation — connect the number to *why*, using
     what M1's EDA and the docstrings in `model/models.py` already established
     (e.g. `PAY_0`'s non-monotonic risk breaking linear models; bagging cutting
     variance vs boosting cutting bias; `BILL_AMT*` collinearity violating Naive
     Bayes independence).
4. Flag disagreements explicitly rather than silently picking a winner:
   - If the best-MCC model and best-AUC model differ, say so (evaluate.py's own
     output already flags this — carry it into the write-up).
   - If accuracy and MCC rank models differently, that itself is a finding worth
     a sentence — it is usually the imbalance trap the README should call out.
5. Compare the new table against the last one recorded in `PROGRESS.md`. If any
   metric moved by more than ~0.005 with no code change since, treat that as a
   reproducibility bug, not a footnote — investigate before writing it up.
6. Report back: the markdown table, the per-model observations, and whether
   `reports/comparison_table.csv` / `.md` are new, unchanged, or updated (git
   status is the source of truth for "unchanged").

## Notes

- `model/evaluate.py` scores the **committed artifacts**, not a fresh refit —
  this is the same model `app.py` serves in M5, so the table reflects reality.
- `reports/comparison_table.csv` is intentionally exempted from `.gitignore`'s
  blanket `*.csv` rule (see the `.gitignore` comment) so it can be committed
  alongside `README.md`.
