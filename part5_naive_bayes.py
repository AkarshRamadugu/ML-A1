"""BITS F464 Assignment 1 -- Part 5: flat Gaussian Naive Bayes.

Runs with:  python part5_naive_bayes.py
from the folder containing Frogs_MFCCs.csv.  Uses only numpy/stdlib.

The 22 MFCCs are modelled as conditionally independent Gaussians given the
species.  Everything is computed in log space and vectorised over
(samples, classes, features); no Python loops over samples.
"""

import numpy as np

from common import (
    accuracy,
    derive_family,
    load_data,
    pct,
    species_family_map,
    stratified_split,
)
from part1_decision_tree import DecisionTree


class GaussianNB:
    """Gaussian naive Bayes with variance smoothing."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None          # class means, shape (K, d)
        self.var_ = None            # class variances (+ epsilon), shape (K, d)
        self.epsilon_ = None
        self.n_features_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if y.shape[0] != X.shape[0]:
            raise ValueError("y must have one label per row")

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = self.classes_.size
        self.n_features_ = n_features

        # epsilon = var_smoothing * max_j Var_train(x_j), over ALL train rows.
        self.epsilon_ = float(self.var_smoothing * np.var(X, axis=0).max())

        self.theta_ = np.empty((n_classes, n_features))
        self.var_ = np.empty((n_classes, n_features))
        self.class_prior_ = np.empty(n_classes)

        for k, label in enumerate(self.classes_):
            X_k = X[y == label]
            self.theta_[k] = X_k.mean(axis=0)
            # ddof=0 gives (1/n_c) * sum (x - mu)^2, exactly as specified.
            self.var_[k] = X_k.var(axis=0) + self.epsilon_
            self.class_prior_[k] = X_k.shape[0] / n_samples

        return self

    def _joint_log_likelihood(self, X):
        """log pi_c - 1/2 sum_j [log(2 pi sigma_cj^2) + (xj-mu_cj)^2/sigma_cj^2]."""
        if self.theta_ is None:
            raise RuntimeError("call fit before predict")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # (n, K, d) broadcast: no loop over samples.
        diff = X[:, None, :] - self.theta_[None, :, :]
        log_det = np.sum(np.log(2.0 * np.pi * self.var_), axis=1)   # (K,)
        quad = np.sum(diff * diff / self.var_[None, :, :], axis=2)  # (n, K)
        return np.log(self.class_prior_)[None, :] - 0.5 * (
            log_det[None, :] + quad)

    def predict(self, X):
        scores = self._joint_log_likelihood(X)
        # argmax takes the first maximum -> smallest class label on a tie.
        return self.classes_[np.argmax(scores, axis=1)]


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _brute_force_scores(model, x):
    """Loop implementation of the same formula, for validation only."""
    scores = []
    for k in range(model.classes_.size):
        total = np.log(model.class_prior_[k])
        for j in range(model.n_features_):
            mu = model.theta_[k, j]
            var = model.var_[k, j]
            total -= 0.5 * (np.log(2.0 * np.pi * var)
                            + (x[j] - mu) ** 2 / var)
        scores.append(total)
    return np.array(scores)


def sanity_checks():
    print("Sanity checks")

    # One feature, equal priors: class 0 = {0, 2}, class 1 = {10, 12}.
    X = np.array([[0.0], [2.0], [10.0], [12.0]])
    y = np.array([0, 0, 1, 1])
    model = GaussianNB().fit(X, y)

    assert np.allclose(model.theta_.ravel(), [1.0, 11.0]), model.theta_
    assert np.allclose(model.var_.ravel(), [1.0, 1.0], atol=1e-6), model.var_
    assert np.allclose(model.class_prior_, [0.5, 0.5])
    assert abs(model.epsilon_ - 1e-9 * np.var(X.ravel())) < 1e-20
    pred = model.predict(np.array([[3.0], [9.0]]))
    assert np.array_equal(pred, np.array([0, 1])), pred
    print("  means 1/11, variances 1 (+eps); x=3 -> class 0, x=9 -> class 1"
          "  -> PASS")

    # Vectorised log-likelihood matches a transparent double loop.
    rng = np.random.RandomState(0)
    Xr = rng.normal(size=(40, 5))
    yr = rng.randint(0, 3, size=40)
    model = GaussianNB().fit(Xr, yr)
    scores = model._joint_log_likelihood(Xr[:7])
    for i in range(7):
        assert np.allclose(scores[i], _brute_force_scores(model, Xr[i]))
    print("  vectorised (n, K, d) log-likelihood == brute-force loops  -> PASS")

    # No underflow / infinities in log space.
    assert np.all(np.isfinite(model._joint_log_likelihood(Xr)))
    print("  joint log-likelihood finite for every row  -> PASS")
    print("  All sanity checks passed.\n")


# ---------------------------------------------------------------------------
# Part 5 experiments
# ---------------------------------------------------------------------------


def print_correlation(corr):
    """Print the 22x22 matrix compactly (m1..m22 = MFCCs_1..22)."""
    d = corr.shape[0]
    width = 5
    print("     " + "".join(("%d" % (j + 1)).rjust(width) for j in range(d)))
    for i in range(d):
        row = "m%-4d" % (i + 1)
        row += "".join(("%.2f" % corr[i, j]).rjust(width) for j in range(d))
        print(row)


def main():
    sanity_checks()

    X, species, genus, family, record_id, feature_names = load_data()
    train, test = stratified_split(species)
    mapping = species_family_map(species[train], family[train])

    print("Data: %d rows, %d features; split %d train / %d test"
          % (X.shape[0], X.shape[1], train.size, test.size))
    print("Flat task: %d species classes\n" % np.unique(species).size)

    model = GaussianNB(var_smoothing=1e-9).fit(X[train], species[train])
    pred_train = model.predict(X[train])
    pred_test = model.predict(X[test])

    nb = {
        "train": accuracy(species[train], pred_train),
        "test": accuracy(species[test], pred_test),
        "family": accuracy(family[test], derive_family(pred_test, mapping)),
    }

    print("Gaussian NB (results matrix row)")
    print("  %-10s %10s %10s %12s"
          % ("criterion", "train %", "test %", "family %"))
    print("  %-10s %10s %10s %12s"
          % ("-", pct(nb["train"]), pct(nb["test"]), pct(nb["family"])))
    print("  variance smoothing epsilon = %.6g" % model.epsilon_)
    print()

    print("Correlation of the 22 training MFCCs (m1..m22 = MFCCs_1..22)")
    corr = np.corrcoef(X[train], rowvar=False)
    print_correlation(corr)
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    iu = np.triu_indices(corr.shape[0], k=1)
    pairs = corr[iu]
    print("  off-diagonal |r|: mean %.2f, max %.2f;"
          " %d/%d pairs have |r| >= 0.5, %d have |r| >= 0.8"
          % (np.abs(off).mean(), np.abs(off).max(),
             int((np.abs(pairs) >= 0.5).sum()), pairs.size,
             int((np.abs(pairs) >= 0.8).sum())))
    order = np.argsort(-np.abs(pairs))[:5]
    print("  strongest pairs:")
    for idx in order:
        i, j = iu[0][idx], iu[1][idx]
        print("    %-9s - %-9s r = %+.2f"
              % ("MFCCs_%d" % (i + 1), "MFCCs_%d" % (j + 1), pairs[idx]))
    print("  NB treats these features as independent, so strongly correlated")
    print("  MFCCs count the same evidence several times: the diagonal")
    print("  covariance is misspecified and the class posteriors are")
    print("  overconfident, although the argmax can still be right.")
    print()

    print("Comparison with the flat single tree (test set)")
    print("  %-12s %-9s %10s %10s %12s"
          % ("method", "criterion", "train %", "test %", "family %"))
    flat = {}
    for criterion in ("gini", "entropy"):
        tree = DecisionTree(criterion=criterion,
                            rng=np.random.RandomState(42)).fit(X[train], species[train])
        pred = tree.predict(X[test])
        flat[criterion] = {
            "train": accuracy(species[train], tree.predict(X[train])),
            "test": accuracy(species[test], pred),
            "family": accuracy(family[test], derive_family(pred, mapping)),
        }
        print("  %-12s %-9s %10s %10s %12s"
              % ("SingleTree", criterion, pct(flat[criterion]["train"]),
                 pct(flat[criterion]["test"]), pct(flat[criterion]["family"])))
    print("  %-12s %-9s %10s %10s %12s"
          % ("GaussianNB", "-", pct(nb["train"]), pct(nb["test"]),
             pct(nb["family"])))
    print()
    print("  NB vs flat tree (test species accuracy): gini %+.2f, entropy %+.2f"
          " points"
          % (100.0 * (nb["test"] - flat["gini"]["test"]),
             100.0 * (nb["test"] - flat["entropy"]["test"])))
    print()

    print("Per-species test recall (Gaussian NB)")
    for label in model.classes_:
        mask = species[test] == label
        print("  %-24s n = %4d  recall = %s%%"
              % (label, int(mask.sum()),
                 pct(accuracy(species[test][mask], pred_test[mask]))))


if __name__ == "__main__":
    main()
