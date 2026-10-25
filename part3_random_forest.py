"""BITS F464 Assignment 1 -- Part 3: random forest on flat trees.

Runs with:  python part3_random_forest.py
from the folder containing Frogs_MFCCs.csv.  Uses only numpy/stdlib plus the
Part 1 DecisionTree.

Random forest = bagging + per-node feature subsetting.  Each tree is trained
on its own bootstrap sample of N rows; at every split m = floor(sqrt(d))
features are drawn without replacement from the d = 22 features, and only
those candidates are scored.  Prediction is a majority vote.
"""

import math
import time

import numpy as np

from common import (
    accuracy,
    derive_family,
    load_data,
    majority_vote,
    pct,
    species_family_map,
    stratified_split,
)
from part1_decision_tree import DecisionTree
from part2_bagging_boosting import BaggingClassifier


# ---------------------------------------------------------------------------
# Random forest
# ---------------------------------------------------------------------------


class RandomForest:
    """Bagged trees with a random feature subset at every node."""

    def __init__(self, n_estimators=50, criterion="gini", max_features=None,
                 max_depth=None, min_samples_split=2, rng=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features      # None -> floor(sqrt(d)) at fit
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = np.random.RandomState(42) if rng is None else rng

        self.classes_ = None
        self.n_features_ = None
        self.max_features_ = None
        self.trees_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)

        if self.max_features is None:
            self.max_features_ = max(1, int(math.floor(math.sqrt(n_features))))
        else:
            self.max_features_ = max(1, min(int(self.max_features), n_features))

        self.trees_ = []
        for _ in range(self.n_estimators):
            boot = self.rng.randint(0, n_samples, size=n_samples)
            # One shared RNG per forest: bootstrap draws and all per-node
            # feature draws come from this same stream.
            tree = DecisionTree(criterion=self.criterion,
                                max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                max_features=self.max_features_,
                                rng=self.rng)
            tree.fit(X[boot], y[boot])
            self.trees_.append(tree)
        return self

    def predict_all_trees(self, X):
        """(n_estimators, n_samples) matrix of individual tree predictions."""
        return np.array([tree.predict(X) for tree in self.trees_])

    def predict(self, X):
        return majority_vote(self.predict_all_trees(X), self.classes_)


def pairwise_agreement(predictions):
    """Mean fraction of samples on which two trees agree, over all pairs."""
    predictions = np.asarray(predictions)
    n_models = predictions.shape[0]
    total = 0.0
    n_pairs = 0
    for i in range(n_models):
        eq = (predictions[i] == predictions[i + 1:])
        total += float(eq.mean(axis=1).sum())
        n_pairs += eq.shape[0]
    return total / n_pairs


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def sanity_checks():
    print("Sanity checks")
    rng = np.random.RandomState(0)

    # Default m = floor(sqrt(d)); for d = 22 this is 4.
    X = rng.normal(size=(60, 22))
    y = rng.randint(0, 3, size=60)
    forest = RandomForest(n_estimators=1, rng=np.random.RandomState(42)).fit(X, y)
    assert forest.max_features_ == 4, forest.max_features_
    print("  d = 22 -> m = floor(sqrt(22)) = %d  -> PASS" % forest.max_features_)

    # Feature subsets are drawn afresh at every split, not once per tree.
    # XOR target: any tree with m = 1 must use both x0 and x1 to fit it.
    Xx = rng.normal(size=(400, 4))
    yx = ((Xx[:, 0] > 0.0) ^ (Xx[:, 1] > 0.0)).astype(int)
    forest = RandomForest(n_estimators=8, max_features=1,
                          rng=np.random.RandomState(42)).fit(Xx, yx)
    used = {node["feature"] for tree in forest.trees_
            for node in tree.nodes_ if not node["leaf"]}
    assert {0, 1}.issubset(used), used
    print("  m = 1 on XOR data: trees split on both x0 and x1 "
          "(per-node draws)  -> PASS")

    # Same seed -> identical forest; different seed -> different forest.
    Xs = rng.normal(size=(300, 5))
    ys = (Xs[:, 0] + 0.5 * Xs[:, 1] > 0).astype(int)
    p1 = RandomForest(n_estimators=8, rng=np.random.RandomState(42)).fit(Xs, ys).predict(Xs)
    p2 = RandomForest(n_estimators=8, rng=np.random.RandomState(42)).fit(Xs, ys).predict(Xs)
    p3 = RandomForest(n_estimators=8, rng=np.random.RandomState(7)).fit(Xs, ys).predict(Xs)
    assert np.array_equal(p1, p2)
    assert not np.array_equal(p1, p3)
    print("  same seed reproduces the forest; different seed changes it  -> PASS")

    # If none of the sampled features can split with positive gain, the node
    # must become a leaf.  Force feature 1, which is constant here.
    class _FixedChoice:
        def choice(self, n, m, replace=False):
            return np.array([1])

    Xc = np.column_stack([rng.normal(size=100), np.zeros(100)])
    yc = (Xc[:, 0] > 0.0).astype(int)
    tree = DecisionTree(max_features=1, rng=_FixedChoice()).fit(Xc, yc)
    assert tree.n_leaves_ == 1, tree.n_leaves_
    print("  no positive-gain sampled feature -> single leaf  -> PASS")
    print("  All sanity checks passed.\n")


# ---------------------------------------------------------------------------
# Part 3 experiments
# ---------------------------------------------------------------------------


def evaluate(model, X, species, family, train, test, mapping):
    t0 = time.perf_counter()
    model.fit(X[train], species[train])
    fit_time = time.perf_counter() - t0

    pred_train = model.predict(X[train])
    pred_test = model.predict(X[test])
    preds_test = model.predict_all_trees(X[test])

    individual = [accuracy(species[test], row) for row in preds_test]
    return {
        "model": model,
        "fit_time": fit_time,
        "train": accuracy(species[train], pred_train),
        "test": accuracy(species[test], pred_test),
        "family": accuracy(family[test], derive_family(pred_test, mapping)),
        "tree_test": float(np.mean(individual)),
        "agreement": pairwise_agreement(preds_test),
    }


def main():
    sanity_checks()

    X, species, genus, family, record_id, feature_names = load_data()
    train, test = stratified_split(species)
    mapping = species_family_map(species[train], family[train])

    print("Data: %d rows, %d features; split %d train / %d test"
          % (X.shape[0], X.shape[1], train.size, test.size))
    print("Flat task: %d species classes; forest m = %d features per split\n"
          % (np.unique(species).size, int(math.floor(math.sqrt(X.shape[1])))))

    forest_results = {}
    for criterion in ("gini", "entropy"):
        print("Training random forest (%s) ..." % criterion)
        forest = RandomForest(n_estimators=50, criterion=criterion,
                              max_features=None, rng=np.random.RandomState(42))
        forest_results[criterion] = evaluate(forest, X, species, family,
                                             train, test, mapping)
    print()

    print("Part 3 results (results matrix rows)")
    print("  %-13s %-8s %10s %10s %12s %8s %8s %8s"
          % ("method", "criterion", "train %", "test %", "family %",
             "trees", "m", "fit s"))
    for criterion in ("gini", "entropy"):
        r = forest_results[criterion]
        print("  %-13s %-8s %10s %10s %12s %8d %8d %8.2f"
              % ("RandomForest", criterion, pct(r["train"]), pct(r["test"]),
                 pct(r["family"]), len(r["model"].trees_),
                 r["model"].max_features_, r["fit_time"]))
    print()

    # Baselines for the requested comparison: single tree and bagging.
    print("Training baselines (single tree, bagging) ...")
    single, bagging = {}, {}
    for criterion in ("gini", "entropy"):
        tree = DecisionTree(criterion=criterion).fit(X[train], species[train])
        pred_test = tree.predict(X[test])
        single[criterion] = {
            "test": accuracy(species[test], pred_test),
            "family": accuracy(family[test], derive_family(pred_test, mapping)),
        }

        bag = BaggingClassifier(n_estimators=25, criterion=criterion,
                                rng=np.random.RandomState(42))
        bagging[criterion] = evaluate(bag, X, species, family, train, test,
                                      mapping)
    print()

    print("Comparison on the test set")
    print("  %-13s %-8s %10s %10s %10s %10s %10s"
          % ("method", "criterion", "test %", "vs tree", "vs bag",
             "family %", "vs tree"))
    for criterion in ("gini", "entropy"):
        s = single[criterion]
        b = bagging[criterion]
        f = forest_results[criterion]
        print("  %-13s %-8s %10s %10s %10s %10s %10s"
              % ("SingleTree", criterion, pct(s["test"]), "-", "-",
                 pct(s["family"]), "-"))
        print("  %-13s %-8s %10s %+10.2f %10s %10s %+10.2f"
              % ("Bagging", criterion, pct(b["test"]),
                 100.0 * (b["test"] - s["test"]), "-",
                 pct(b["family"]), 100.0 * (b["family"] - s["family"])))
        print("  %-13s %-8s %10s %+10.2f %+10.2f %10s %+10.2f"
              % ("RandomForest", criterion, pct(f["test"]),
                 100.0 * (f["test"] - s["test"]),
                 100.0 * (f["test"] - b["test"]),
                 pct(f["family"]), 100.0 * (f["family"] - s["family"])))
    print()

    print("Extra: individual-tree accuracy and pairwise agreement (test set)")
    print("  %-9s %-8s %14s %16s"
          % ("method", "criterion", "mean tree %", "agreement %"))
    for criterion in ("gini", "entropy"):
        b = bagging[criterion]
        f = forest_results[criterion]
        print("  %-9s %-8s %14s %16s"
              % ("Bagging", criterion, pct(b["tree_test"]), pct(b["agreement"])))
        print("  %-9s %-8s %14s %16s"
              % ("Forest", criterion, pct(f["tree_test"]), pct(f["agreement"])))


if __name__ == "__main__":
    main()
