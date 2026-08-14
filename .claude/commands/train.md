---
description: Retrain all registered models and report the sanity metrics
allowed-tools: Bash(*/.venv/Scripts/python.exe *), Read, Grep
---

Retrain the models and report the result.

1. Run `.\.venv\Scripts\python.exe -m model.train` from the project root.
2. If it fails because `data/train.csv` is missing, run
   `.\.venv\Scripts\python.exe -m model.data_prep` first, then retry.
3. Report the metrics table as-is.
4. Flag anything suspicious rather than glossing over it:
   - any model whose **accuracy is near 77.87%** — that is the all-negative
     baseline, so matching it means nothing was learned
   - any model whose **AUC is below 0.55** — barely better than a coin flip
   - a **large train/test accuracy gap**, which indicates overfitting
   - any convergence warning from Logistic Regression
5. Compare against the previously recorded numbers in `PROGRESS.md`. If a metric
   moved by more than ~0.01, say so explicitly and explain why — a silent
   regression in the graded comparison table is the failure mode this command
   exists to catch.

$ARGUMENTS
