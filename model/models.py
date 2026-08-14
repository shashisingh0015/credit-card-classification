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

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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
    return _pipe("logreg", LogisticRegression(
        max_iter=2000,
        random_state=config.RANDOM_STATE,
    ))


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
    return _pipe("tree", DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=50,
        criterion="gini",
        random_state=config.RANDOM_STATE,
    ))


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


# Registry. M3 extends this with kNN, Random Forest and Gradient Boosting.
# Keys double as the display names in the comparison table and the Streamlit
# dropdown, so they are written for a reader, not as identifiers.
MODEL_BUILDERS: dict[str, Callable[[], Pipeline]] = {
    "Logistic Regression": logistic_regression,
    "Decision Tree": decision_tree,
    "Naive Bayes (Gaussian)": gaussian_nb,
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
