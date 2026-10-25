"""BITS F464 Assignment 1 -- Part 2: bagging and multi-class SAMME AdaBoost.

Runs with:  python part2_bagging_boosting.py
from the folder containing Frogs_MFCCs.csv.  Uses only numpy/stdlib plus the
Part 1 DecisionTree.

Bagging : T = 25 unconstrained trees, bootstrap of size N, all 22 features,
          majority vote (ties -> smallest class label).
AdaBoost: T = 50 depth-3 trees, multi-class SAMME, weighted vote.
"""

import math
import time

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


# ---------------------------------------------------------------------------
# Voting helpers
# ---------------------------------------------------------------------------


def majority_vote(predictions, classes):
    """Majority vote over rows of `predictions` (n_models, n_samples).

    Ties are broken towards the smallest class label: `classes` is sorted and
    np.argmax returns the first (smallest) index among equal counts.
    """
    predictions = np.asarray(predictions)
    classes = np.asarray(classes)
    n_models, n_samples = predictions.shape
    class_idx = np.searchsorted(classes, predictions)

    votes = np.zeros((n_samples, classes.size), dtype=np.int64)
    rows = np.arange(n_samples)
    for t in range(n_models):
        votes[rows, class_idx[t]] += 1
    return classes[np.argmax(votes, axis=1)]


def weighted_vote(predictions, weights, classes):
    """Weighted vote (used by AdaBoost): argmax_k sum_t alpha_t 1[h_t = k]."""
    predictions = np.asarray(predictions)
    classes = np.asarray(classes)
    n_models, n_samples = predictions.shape
    class_idx = np.searchsorted(classes, predictions)

    scores = np.zeros((n_samples, classes.size), dtype=np.float64)
    rows = np.arange(n_samples)
    for t in range(n_models):
        scores[rows, class_idx[t]] += weights[t]
    return classes[np.argmax(scores, axis=1)]


# ---------------------------------------------------------------------------
# Bagging
# ---------------------------------------------------------------------------


class BaggingClassifier:
    """Bagged unconstrained trees with majority vote."""

    def __init__(self, n_estimators=25, criterion="gini", max_depth=None,
                 max_features=None, rng=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.rng = np.random.RandomState(42) if rng is None else rng
        self.classes_ = None
        self.trees_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)

        self.trees_ = []
        for _ in range(self.n_estimators):
            boot = self.rng.randint(0, n_samples, size=n_samples)
            tree = DecisionTree(criterion=self.criterion,
                                max_depth=self.max_depth,
                                max_features=self.max_features,
                                rng=self.rng)
            tree.fit(X[boot], y[boot])
            self.trees_.append(tree)
        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees_])
        return majority_vote(predictions, self.classes_)


# ---------------------------------------------------------------------------
# AdaBoost (multi-class SAMME)
# ---------------------------------------------------------------------------


class AdaBoostSAMME:
    """SAMME AdaBoost with depth-limited trees as weak learners."""

    def __init__(self, n_estimators=50, criterion="gini", max_depth=3, rng=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.rng = np.random.RandomState(42) if rng is None else rng
        self.classes_ = None
        self.trees_ = []
        self.alphas_ = []
        self.errors_ = []      # unclipped weighted errors, for reporting
        self.stopped_early_ = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        n_classes = self.classes_.size

        w = np.full(n_samples, 1.0 / n_samples)
        self.trees_, self.alphas_, self.errors_ = [], [], []
        self.stopped_early_ = False

        for _ in range(self.n_estimators):
            tree = DecisionTree(criterion=self.criterion,
                                max_depth=self.max_depth,
                                rng=self.rng)
            tree.fit(X, y, sample_weight=w)
            pred = tree.predict(X)
            wrong = pred != y

            raw_error = float(w[wrong].sum())
            eps = min(max(raw_error, 1e-10), 1.0 - 1e-10)

            # No better than random guessing -> discard and stop boosting.
            if eps >= 1.0 - 1.0 / n_classes:
                self.stopped_early_ = True
                break

            alpha = math.log((1.0 - eps) / eps) + math.log(n_classes - 1)
            self.trees_.append(tree)
            self.alphas_.append(alpha)
            self.errors_.append(raw_error)

            w = w * np.exp(alpha * wrong)
            w /= w.sum()

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = np.zeros((X.shape[0], self.classes_.size), dtype=np.float64)
        rows = np.arange(X.shape[0])
        for tree, alpha in zip(self.trees_, self.alphas_):
            pred_idx = np.searchsorted(self.classes_, tree.predict(X))
            scores[rows, pred_idx] += alpha
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def sanity_checks():
    print("Sanity checks")

    # Majority-vote tie -> smallest class label.
    preds = np.array([[2, 0], [1, 0], [0, 2]])
    voted = majority_vote(preds, np.array([0, 1, 2]))
    assert np.array_equal(voted, np.array([0, 0])), voted
    print("  majority vote [2,1,0] -> class 0 (tie -> smallest)  -> PASS")

    # Weighted-vote tie -> smallest class label.
    scores = weighted_vote(preds, np.array([1.0, 1.0, 1.0]), np.array([0, 1, 2]))
    assert np.array_equal(scores, np.array([0, 0])), scores
    print("  weighted vote tie -> smallest class label  -> PASS")

    # AdaBoost on a simple separable 3-class problem reaches 100% train acc.
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(loc=c * 6.0, scale=0.3, size=(40, 2)) for c in range(3)])
    y = np.repeat(np.arange(3), 40)
    model = AdaBoostSAMME(n_estimators=10, max_depth=2,
                          rng=np.random.RandomState(42)).fit(X, y)
    assert accuracy(y, model.predict(X)) == 1.0
    assert len(model.trees_) == len(model.alphas_) == 10
    print("  SAMME on separable 3-class data: train acc = 100.00%  -> PASS")
    print("  All sanity checks passed.\n")


# ---------------------------------------------------------------------------
# Part 2 experiments
# ---------------------------------------------------------------------------


def run_ensemble(model, X, species, family, train, test, mapping):
    t0 = time.perf_counter()
    model.fit(X[train], species[train])
    fit_time = time.perf_counter() - t0

    pred_train = model.predict(X[train])
    pred_test = model.predict(X[test])

    return {
        "model": model,
        "fit_time": fit_time,
        "train": accuracy(species[train], pred_train),
        "test": accuracy(species[test], pred_test),
        "family": accuracy(family[test], derive_family(pred_test, mapping)),
    }


def main():
    sanity_checks()

    X, species, genus, family, record_id, feature_names = load_data()
    train, test = stratified_split(species)
    mapping = species_family_map(species[train], family[train])

    print("Data: %d rows, %d features; split %d train / %d test"
          % (X.shape[0], X.shape[1], train.size, test.size))
    print("Flat task: %d species classes\n" % np.unique(species).size)

    results = {}
    for criterion in ("gini", "entropy"):
        print("Training %s models ..." % criterion)
        bagging = BaggingClassifier(n_estimators=25, criterion=criterion,
                                    rng=np.random.RandomState(42))
        results[("bagging", criterion)] = run_ensemble(
            bagging, X, species, family, train, test, mapping)

        boosting = AdaBoostSAMME(n_estimators=50, criterion=criterion,
                                 max_depth=3, rng=np.random.RandomState(42))
        results[("adaboost", criterion)] = run_ensemble(
            boosting, X, species, family, train, test, mapping)

        stumps = AdaBoostSAMME(n_estimators=50, criterion=criterion,
                               max_depth=1, rng=np.random.RandomState(42))
        results[("stumps", criterion)] = run_ensemble(
            stumps, X, species, family, train, test, mapping)
    print()

    print("Part 2 results (results matrix rows)")
    print("  %-10s %-8s %10s %10s %12s %8s %8s"
          % ("method", "criterion", "train %", "test %", "family %",
             "rounds", "fit s"))
    for method, label in (("bagging", "Bagging"), ("adaboost", "AdaBoost")):
        for criterion in ("gini", "entropy"):
            r = results[(method, criterion)]
            print("  %-10s %-8s %10s %10s %12s %8d %8.2f"
                  % (label, criterion, pct(r["train"]), pct(r["test"]),
                     pct(r["family"]), len(r["model"].trees_), r["fit_time"]))
    print()

    print("AdaBoost: depth-3 trees vs decision stumps (T=50; stumps not in matrix)")
    print("  %-8s %12s %12s %12s %12s"
          % ("criterion", "d3 train %", "d3 test %", "stump train %", "stump test %"))
    for criterion in ("gini", "entropy"):
        d3 = results[("adaboost", criterion)]
        st = results[("stumps", criterion)]
        print("  %-8s %12s %12s %12s %12s"
              % (criterion, pct(d3["train"]), pct(d3["test"]),
                 pct(st["train"]), pct(st["test"])))
    print("  (A stump is one binary split, so on 10 classes it can only ever")
    print("   isolate one class per round; 50 stumps cannot express the")
    print("   multi-way boundaries that depth-3 trees can.)")
    print()

    # Single-tree baseline from Part 1, for the comparison requested.
    baseline = {}
    for criterion in ("gini", "entropy"):
        tree = DecisionTree(criterion=criterion).fit(X[train], species[train])
        baseline[criterion] = {
            "test": accuracy(species[test], tree.predict(X[test])),
            "family": accuracy(family[test],
                               derive_family(tree.predict(X[test]), mapping)),
        }

    print("Comparison with the single tree of Part 1 (test set)")
    print("  %-10s %-8s %10s %10s %10s %10s"
          % ("method", "criterion", "test %", "vs tree", "family %", "vs tree"))
    for criterion in ("gini", "entropy"):
        b = baseline[criterion]
        print("  %-10s %-8s %10s %10s %10s %10s"
              % ("SingleTree", criterion, pct(b["test"]), "-",
                 pct(b["family"]), "-"))
        for method, label in (("bagging", "Bagging"), ("adaboost", "AdaBoost")):
            r = results[(method, criterion)]
            print("  %-10s %-8s %10s %+10.2f %10s %+10.2f"
                  % (label, criterion, pct(r["test"]),
                     100.0 * (r["test"] - b["test"]),
                     pct(r["family"]),
                     100.0 * (r["family"] - b["family"])))
    print()

    for criterion in ("gini", "entropy"):
        r = results[("adaboost", criterion)]
        model = r["model"]
        if model.errors_:
            detail = ("%d rounds, early stop = %s, first/last eps = %.4f/%.4f"
                      % (len(model.trees_), model.stopped_early_,
                         model.errors_[0], model.errors_[-1]))
        else:
            detail = "0 rounds, stopped immediately"
        print("AdaBoost %-7s: %s" % (criterion, detail))


if __name__ == "__main__":
    main()
