"""Model registry: one Pipeline factory per classifier.

Every factory returns a `Pipeline` whose first step is the *same*
`build_preprocessor()` from `preprocess.py`. Two consequences worth being explicit
about:

* **The comparison is fair.** All models see identically prepared features, so
  differences in the metrics table are attributable to the algorithms rather than
  to accidental differences in preprocessing.
* **Each model is one serializable unit.** `joblib.dump(pipeline)` captures the
  scaler, the encoder and the fitted estimator together, so the Streamlit app
  cannot accidentally apply different preprocessing at serving time than was used
  at training time.

Hyperparameters are hand-picked and fixed rather than grid-searched. The
assignment asks for the six metrics, not for tuned models, and fixed parameters
keep the "why does this model behave this way" reasoning attributable to the
algorithm instead of to a search result.

**No class weighting anywhere.** `class_weight` is supported by Logistic
Regression, Decision Tree and Random Forest but *not* by kNN, GaussianNB or
sklearn's GradientBoosting. Weighting only the three that support it would mean
the comparison table mixed two different training regimes. Everything therefore
trains unweighted, and the resulting modest recall is itself a finding to discuss.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config
from .preprocess import build_preprocessor


def _pipe(name: str, estimator) -> Pipeline:
    """Wrap an estimator behind the shared preprocessor."""
    return Pipeline([("prep", build_preprocessor()), (name, estimator)])


def logistic_regression() -> Pipeline:
    """Linear decision boundary in log-odds space.

    Models log(p / (1-p)) as a linear combination of features, then maps it back
    through the sigmoid to a probability. Because the boundary is a hyperplane, it
    can only express "risk increases steadily with this feature".

    That is exactly the limitation M1 found: `PAY_0` risk is *not* monotonic in the
    code (-2: 13.2%, -1: 17.0%, 0: 12.6%), and no single coefficient can represent
    that inversion.

    **Confirmed.** It trails the Decision Tree on both ranking and classification:
    AUC 0.7185 vs 0.7447, MCC 0.3515 vs 0.3893. The mechanism was predicted from
    the data before either model was fitted.

    `max_iter=2000` because lbfgs on 29 correlated features does not always
    converge inside the default 100 iterations. Scaling (handled by the
    preprocessor) is what keeps this tractable at all -- on raw features, where
    LIMIT_BAL reaches 10^6, the optimiser has to crawl to avoid diverging.
    """
    return _pipe(
        "logreg",
        LogisticRegression(
            max_iter=2000,
            random_state=config.RANDOM_STATE,
        ),
    )


def decision_tree() -> Pipeline:
    """Recursive axis-aligned splits, each chosen to reduce Gini impurity.

    Left unconstrained, a tree keeps splitting until every leaf is pure. Measured
    on this dataset, that is textbook overfitting:

        tree                    train acc   test acc   test MCC   leaves
        unconstrained              0.9995     0.7220     0.2126     3810
        max_depth=5, leaf>=50      0.8235     0.8188     0.3893       26

    The unconstrained tree memorises the training set almost perfectly and then
    scores *below* the 77.87% all-negative baseline on test data, with MCC nearly
    halved. Constraining it costs 18 points of training accuracy and buys back 10
    points of test accuracy -- the bias/variance trade in one table.

    `max_depth=5` caps the tree at 32 possible leaves (it uses 26), and
    `min_samples_leaf=50` ensures every prediction rests on at least 50 clients
    rather than one or two.

    The root split is on `PAY_0`, which is what M1's correlation analysis predicted.

    Trees need no feature scaling: each split is a threshold on a single feature,
    so any monotonic rescaling produces identical splits.
    """
    return _pipe(
        "tree",
        DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=50,
            criterion="gini",
            random_state=config.RANDOM_STATE,
        ),
    )


def gaussian_nb() -> Pipeline:
    """Bayes' rule plus a strong independence assumption.

    Estimates P(default | features) via P(features | default) * P(default), and
    makes the computation tractable by assuming every feature is conditionally
    independent of the others given the class -- hence "naive". Each feature is
    modelled as a Gaussian per class.

    Both assumptions are violated here, knowingly:

    * **Independence fails badly.** M1 measured `BILL_AMT*` correlations of
      0.80-0.95. The model effectively counts that one piece of evidence six
      times, which drives its probabilities toward 0 and 1 -- overconfident and
      poorly calibrated.
    * **Normality fails.** `PAY_AMT*` are strongly right-skewed with a spike at
      zero, nothing like a bell curve.

    **Measured, and the split is striking:**

        accuracy 0.5750    <- *below* the 77.87% do-nothing baseline
        MCC      0.2377    <- weakest of the three
        AUC      0.7300    <- yet BEATS Logistic Regression's 0.7185

    So this is a poor *classifier* but a respectable *ranker*. AUC only cares about
    the order of the scores, whereas accuracy/F1/MCC depend on where the 0.5
    threshold falls. Violating independence pushes the probabilities toward 0 and 1
    and wrecks their calibration, but it largely preserves their ordering.

    Two lessons worth keeping. First, accuracy misleads in *both* directions -- 57.5%
    looks catastrophic, yet this model ranks risk better than the linear one.
    Second, the fix for a good-AUC/bad-F1 model is usually to move the threshold,
    not to abandon the model.

    GaussianNB rather than MultinomialNB: `BILL_AMT*` contain genuine negative
    values (credit balances), and MultinomialNB requires non-negative input.
    """
    return _pipe("gnb", GaussianNB())


def knn() -> Pipeline:
    """Lazy learner: no model is fitted, the training set *is* the model.

    `fit` only memorises the 23,972 training rows; all the work happens at predict
    time, where each test client is compared against every stored one. That inverts
    the usual cost profile -- near-zero training time, the slowest predictions of
    the six, and the largest artifact, since the whole training matrix is pickled.

    **Scaling is not optional here, it is the whole ballgame.** Euclidean distance
    sums squared differences across features, so a feature's influence is
    proportional to its numeric range. Unscaled, `LIMIT_BAL` spans ~10^6 and `AGE`
    ~10^1, meaning a 50-year age gap contributes about 10^-8 as much distance as a
    modest credit-limit gap -- every "neighbour" would be chosen on credit limit
    alone. The `StandardScaler` inside the shared preprocessor is what makes the
    other 28 features count at all.

    **The curse of dimensionality applies even so.** In 29 dimensions the ratio
    between the nearest and farthest points contracts toward 1, so "neighbourhood"
    stops meaning "similar client" and starts meaning "not much further away than
    anyone else". kNN degrades in high dimensions for reasons no amount of tuning
    fixes; this is a good dataset to see it on.

    `n_neighbors=25` is a deliberate bias/variance choice. k=1 gives a perfectly
    memorised, zero-bias, maximum-variance model whose decision boundary chases
    single noisy points -- and M1 found 21 outright contradictory records, which k=1
    would faithfully reproduce. Large k averages over more clients, smoothing the
    boundary and raising bias. 25 also gives `predict_proba` a usable 1/25 = 0.04
    resolution, which matters for AUC: with k=5 only six distinct scores exist and
    the ROC curve degenerates to six points.

    `weights="distance"` lets closer neighbours count for more, which recovers some
    of the locality that plain majority voting throws away at k=25.

    **Measured:** accuracy 0.8098, AUC 0.7412, F1 0.4390, MCC 0.3614. It lands
    between the two linear-ish baselines and the trees -- better than Logistic
    Regression on both headline metrics (MCC 0.3614 vs 0.3515, AUC 0.7412 vs
    0.7185), because a local vote can express the non-monotonic `PAY_0` structure
    that a single linear coefficient cannot; but still short of the Decision Tree
    (MCC 0.3893), which finds the `PAY_0 >= 1` threshold directly instead of
    approximating it through distances in 29 dimensions.

    The cost profile came out exactly inverted as described: **fit 0.03s** -- the
    fastest of the six, since fitting is just storing the array -- but a **2,072 KB
    artifact against Logistic Regression's 2 KB**, a factor of a thousand, and the
    slowest predictions. Same accuracy tier, wildly different engineering cost.
    """
    return _pipe(
        "knn",
        KNeighborsClassifier(
            n_neighbors=25,
            weights="distance",
            n_jobs=-1,
        ),
    )


def random_forest() -> Pipeline:
    """Bagging: many decorrelated deep trees, averaged.

    Two independent sources of randomness, and both are load-bearing:

    1. **Bootstrap sampling** -- each tree trains on a resample of the training set
       drawn with replacement, so no two trees see the same data.
    2. **Random feature subsets** -- at *every split*, only `sqrt(29) ~ 5` features
       are considered as candidates.

    Bagging alone would not help much, because trees grown on resamples of the same
    data are highly correlated, and averaging correlated estimators barely reduces
    variance. The per-split feature subsetting is what breaks that correlation:
    without it, `PAY_0` (the strongest predictor, and the single Decision Tree's
    root split) would be the root of all 300 trees and the ensemble would be 300
    near-copies. Forcing most splits to look elsewhere makes the trees genuinely
    different, and *that* is what makes the averaging work.

    **This reduces variance, not bias.** Each individual tree here is deliberately
    deeper (`max_depth=12`) than the standalone Decision Tree's 5 -- a single tree
    that deep overfits badly, which is fine, because averaging 300 differently-
    overfitted trees cancels the noise while keeping the signal they agree on. That
    is why adding trees essentially cannot overfit a forest: more trees is a better
    Monte Carlo estimate of the same average, not a more flexible model.

    `min_samples_leaf=20` still floors each leaf at 20 clients, and caps the
    artifact size -- these joblib files are committed to git.

    Contrast with `gradient_boosting` below: same base learner, opposite strategy.

    **Measured:** accuracy 0.8193, AUC 0.7725, F1 0.4678, MCC 0.3972 -- the best MCC
    of the six, and the best accuracy. Against the standalone Decision Tree the
    gains are lopsided and instructively so: MCC rises only 0.3893 -> 0.3972, but
    AUC rises 0.7447 -> 0.7725. Averaging 300 trees mostly improves the *ranking* of
    clients by risk, while the count of labels that flip across the 0.5 threshold
    barely moves. That gap is M4's argument for reporting AUC alongside MCC, and the
    hint that 0.5 is the wrong cutoff for a 22% base rate.

    **Fit time 1.35s vs Gradient Boosting's 18.39s**, despite this model having more
    trees (300 vs 200) and deeper ones (12 vs 3). That is the parallel/sequential
    split made concrete: independent trees run on all cores via `n_jobs=-1`, and
    boosting's cannot, at any core count.
    """
    return _pipe(
        "rf",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    )


def gradient_boosting() -> Pipeline:
    """Boosting: many shallow trees, fitted sequentially to the running error.

    The bagging/boosting contrast is the most useful thing in this file, so state it
    plainly:

        Random Forest      parallel     deep trees     averaged      cuts VARIANCE
        Gradient Boosting  sequential   shallow trees  summed        cuts BIAS

    A forest's trees are independent and could be fitted in any order, or all at
    once. Boosting's cannot: tree *n* is fitted to the gradient of the loss with
    respect to the predictions of trees 1..n-1 -- that is, to what the ensemble so
    far is still getting wrong. Each tree is a small correction, added rather than
    averaged.

    `max_depth=3` is the point, not a constraint to relax. Boosting needs *weak*
    learners: a depth-3 tree is barely better than guessing on its own, but 200 of
    them, each cleaning up its predecessors' residuals, compose into a strong model.
    Deep trees would fit the residuals too well, leaving the next tree nothing to
    learn, and the whole sequence would collapse into an overfitted forest.

    **Unlike a forest, this one can overfit by adding trees.** Every additional tree
    increases model capacity, so `n_estimators` is a real bias/variance dial rather
    than a "more is better" knob. `learning_rate=0.05` shrinks each tree's
    contribution, so the ensemble approaches the training data slowly and in small
    steps -- the standard shrinkage/`n_estimators` trade, where a lower rate needs
    more trees but generalises better.

    `subsample=0.8` fits each tree on a random 80% of rows ("stochastic gradient
    boosting"), which adds a little bagging-style variance reduction on top and is
    the one place these two ensembles overlap.

    **Measured:** accuracy 0.8186, AUC 0.7728, F1 0.4685, MCC 0.3958 -- best AUC and
    best F1 of the six, and a statistical dead heat with Random Forest (AUC differs
    by 0.0003, MCC by 0.0014, on 5,993 test rows). Two very different routes to the
    same ceiling, which is itself the finding: on this dataset the limit is the
    signal available in 23 features, not the choice of ensemble.

    Given the tie, **Random Forest is the better engineering choice here** -- 14x
    faster to fit (1.35s vs 18.39s), parallelisable, and far less sensitive to
    `n_estimators`, since overshooting it costs a forest nothing and costs a boosted
    model accuracy. Gradient Boosting's 88 KB artifact against the forest's 4,888 KB
    is the one column where it wins outright: 200 depth-3 stumps are tiny next to
    300 depth-12 trees.
    """
    return _pipe(
        "gb",
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=config.RANDOM_STATE,
        ),
    )


# Registry -- all six models. Keys double as the display names in the comparison
# table and the Streamlit dropdown, so they are written for a reader, not as
# identifiers. Ordered simple -> complex, which is also the order the README's
# comparison table reads in.
MODEL_BUILDERS: dict[str, Callable[[], Pipeline]] = {
    "Logistic Regression": logistic_regression,
    "Decision Tree": decision_tree,
    "Naive Bayes (Gaussian)": gaussian_nb,
    "k-Nearest Neighbours": knn,
    "Random Forest": random_forest,
    "Gradient Boosting": gradient_boosting,
}


def build(name: str) -> Pipeline:
    """Return a fresh, unfitted pipeline by display name."""
    if name not in MODEL_BUILDERS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name]()


def artifact_path(name: str):
    """Filesystem-safe artifact path for a model's display name."""
    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return config.ARTIFACT_DIR / f"{slug}.joblib"
