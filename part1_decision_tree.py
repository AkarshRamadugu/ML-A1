"""BITS F464 Assignment 1 -- Part 1: single decision tree from scratch.

Runs with:  python part1_decision_tree.py
from the folder containing Frogs_MFCCs.csv.  Uses only numpy/pandas/stdlib.

Implements CART for binary splits x_j <= t with either the Gini index or
Shannon entropy (base 2) as impurity, weighted by sample weights.
"""

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

EPS = 1e-12

# ---------------------------------------------------------------------------
# Impurity helpers
# ---------------------------------------------------------------------------


def gini_impurity(counts):
    """Gini index of a vector of (possibly weighted) class counts."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0.0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p * p))


def entropy_impurity(counts):
    """Shannon entropy (bits) of a vector of class counts; 0 log 0 := 0."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0.0:
        return 0.0
    p = counts / total
    nz = p > 0.0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def _xlogx(a):
    """Element-wise a*log2(a) with 0 log2 0 := 0 (no warnings)."""
    out = np.zeros_like(a)
    nz = a > 0.0
    out[nz] = a[nz] * np.log2(a[nz])
    return out


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------


class DecisionTree:
    """CART decision tree with Gini or entropy splits.

    Parameters
    ----------
    criterion : {"gini", "entropy"}
    max_depth : int or None
        None means unconstrained.
    min_samples_split : int
        A node with fewer samples than this is a leaf.
    max_features : int or None
        Number of features sampled (without replacement) at every split;
        None means all features.  Used by the random forest (Part 3).
    rng : numpy.random.RandomState or None
        Source of randomness for max_features.  Defaults to RandomState(42).
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 max_features=None, rng=None):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.RandomState(42) if rng is None else rng

        self.classes_ = None
        self.n_features_ = None
        self.nodes_ = None       # list of dicts, node 0 is the root
        self.root_ = None

    # -- impurity ----------------------------------------------------------
    def _impurity(self, counts):
        if self.criterion == "gini":
            return gini_impurity(counts)
        return entropy_impurity(counts)

    @staticmethod
    def _new_node():
        return {"leaf": True, "pred": 0, "n_samples": 0, "depth": 0,
                "counts": None, "feature": None, "threshold": None,
                "gain": 0.0, "left": None, "right": None}

    # -- fitting -----------------------------------------------------------
    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1-D with one label per row")

        n_samples, n_features = X.shape
        if sample_weight is None:
            w = np.ones(n_samples, dtype=np.float64)
        else:
            w = np.asarray(sample_weight, dtype=np.float64)
            if w.shape != (n_samples,):
                raise ValueError("sample_weight must have shape (n_samples,)")

        self.classes_ = np.unique(y)
        self.n_features_ = n_features
        y_idx = np.searchsorted(self.classes_, y)
        n_classes = self.classes_.shape[0]

        self.nodes_ = [self._new_node()]
        self.root_ = 0

        # Explicit stack instead of recursion: an unconstrained tree on this
        # data can be deeper than Python's recursion limit.
        stack = [(self.root_, np.arange(n_samples), 0)]
        while stack:
            node_id, idx, depth = stack.pop()
            node = self.nodes_[node_id]
            x = X[idx]
            yi = y_idx[idx]
            wi = w[idx]

            counts = np.bincount(yi, weights=wi, minlength=n_classes)
            node["n_samples"] = idx.shape[0]
            node["depth"] = depth
            node["counts"] = counts
            node["pred"] = int(np.argmax(counts))  # ties -> smallest label

            present = int(np.count_nonzero(counts))
            if (present <= 1
                    or idx.shape[0] < self.min_samples_split
                    or (self.max_depth is not None and depth >= self.max_depth)):
                continue

            split = self._best_split(x, yi, wi, counts)
            if split is None:
                continue

            feature, threshold, gain = split
            mask = x[:, feature] <= threshold
            left_idx = idx[mask]
            right_idx = idx[~mask]
            if left_idx.size == 0 or right_idx.size == 0:
                continue

            left_id = len(self.nodes_)
            self.nodes_.append(self._new_node())
            right_id = len(self.nodes_)
            self.nodes_.append(self._new_node())

            node["leaf"] = False
            node["feature"] = int(feature)
            node["threshold"] = float(threshold)
            node["gain"] = float(gain)
            node["left"] = left_id
            node["right"] = right_id

            stack.append((left_id, left_idx, depth + 1))
            stack.append((right_id, right_idx, depth + 1))

        return self

    def _best_split(self, x, y_idx, w, counts):
        """Best (feature, threshold, gain) at one node, or None.

        Each feature is sorted once; class-weight counts are accumulated with
        cumsum so every candidate threshold is scored in one vectorised pass.
        """
        n_samples, n_features = x.shape
        n_classes = self.classes_.shape[0]
        total_w = w.sum()
        parent_imp = self._impurity(counts)

        if self.max_features is None or self.max_features >= n_features:
            features = np.arange(n_features)
        else:
            m = max(1, int(self.max_features))
            features = np.sort(self.rng.choice(n_features, m, replace=False))

        best_gain = EPS
        best = None
        rows = np.arange(n_samples)

        for j in features:
            xj = x[:, j]
            order = np.argsort(xj, kind="mergesort")
            xs = xj[order]
            if xs[0] == xs[-1]:
                continue
            ys = y_idx[order]
            ws = w[order]

            onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
            onehot[rows, ys] = ws
            cum = np.cumsum(onehot, axis=0)

            # Only split between consecutive *distinct* values; midpoint.
            valid = np.nonzero(xs[:-1] < xs[1:])[0]
            if valid.size == 0:
                continue

            left = cum[valid]
            right = cum[-1] - left
            wl = left.sum(axis=1)
            wr = right.sum(axis=1)

            # Guard against zero-weight children (only possible with
            # non-positive user-supplied weights).
            ok = (wl > 0.0) & (wr > 0.0)
            if not np.all(ok):
                valid, left, right, wl, wr = (valid[ok], left[ok], right[ok],
                                              wl[ok], wr[ok])
                if valid.size == 0:
                    continue

            if self.criterion == "gini":
                ssl = np.einsum("ij,ij->i", left, left)
                ssr = np.einsum("ij,ij->i", right, right)
                # wl*G_L + wr*G_R; dividing by W is common to all splits.
                score = (wl - ssl / wl) + (wr - ssr / wr)
            else:
                # wl*H_L + wr*H_R
                score = ((wl * np.log2(wl) - _xlogx(left).sum(axis=1))
                         + (wr * np.log2(wr) - _xlogx(right).sum(axis=1)))

            gains = parent_imp - score / total_w
            k = int(np.argmax(gains))  # first max -> lowest threshold
            gain = float(gains[k])
            if gain > best_gain:       # strict -> first feature wins ties
                threshold = 0.5 * (xs[valid[k]] + xs[valid[k] + 1])
                best = (int(j), float(threshold), gain)
                best_gain = gain

        return best

    # -- prediction --------------------------------------------------------
    def predict(self, X):
        if self.root_ is None:
            raise RuntimeError("call fit before predict")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.empty(X.shape[0], dtype=np.int64)
        stack = [(self.root_, np.arange(X.shape[0]))]
        while stack:
            node_id, idx = stack.pop()
            if idx.size == 0:
                continue
            node = self.nodes_[node_id]
            if node["leaf"]:
                out[idx] = node["pred"]
                continue
            go_left = X[idx, node["feature"]] <= node["threshold"]
            stack.append((node["left"], idx[go_left]))
            stack.append((node["right"], idx[~go_left]))
        return self.classes_[out]

    # -- inspection --------------------------------------------------------
    @property
    def n_leaves_(self):
        return sum(1 for node in self.nodes_ if node["leaf"])

    @property
    def depth_(self):
        return max((node["depth"] for node in self.nodes_ if node["leaf"]), default=0)


# ---------------------------------------------------------------------------
# Sanity checks from the assignment
# ---------------------------------------------------------------------------


def sanity_checks():
    print("Sanity checks")

    # (a) Gini 0.5 and entropy 1 bit for class counts (3, 3).
    g = gini_impurity([3, 3])
    h = entropy_impurity([3, 3])
    assert abs(g - 0.5) < 1e-12, g
    assert abs(h - 1.0) < 1e-12, h
    print("  (a) counts (3,3): G = %.6f, H = %.6f bit  -> PASS" % (g, h))

    # (b) One split at 2.5 with information gain 1, training accuracy 1.0.
    tree = DecisionTree(criterion="entropy").fit([[1.0], [2.0], [3.0], [4.0]],
                                                np.array([0, 0, 1, 1]))
    root = tree.nodes_[tree.root_]
    assert not root["leaf"], "root should split"
    assert root["feature"] == 0 and abs(root["threshold"] - 2.5) < 1e-12, root
    assert abs(root["gain"] - 1.0) < 1e-12, root["gain"]
    assert accuracy(np.array([0, 0, 1, 1]), tree.predict([[1.0], [2.0], [3.0], [4.0]])) == 1.0
    print("  (b) split x0 <= %.1f, gain = %.6f bit, train acc = 100.00%%  -> PASS"
          % (root["threshold"], root["gain"]))

    # (c) Identical samples, weighted leaf must predict class 1.
    tree = DecisionTree().fit([[5.0], [5.0], [5.0]], np.array([0, 0, 1]),
                              sample_weight=np.array([0.1, 0.1, 0.8]))
    assert tree.n_leaves_ == 1, tree.n_leaves_
    assert tree.predict([[5.0]])[0] == 1
    print("  (c) weighted leaf with counts (0.2, 0.8) -> class 1  -> PASS")

    # (d) Pure node / single-class family -> a single working leaf.
    for labels in (np.zeros(7, dtype=int), np.array([2, 2, 2, 2])):
        Xd = np.arange(2.0 * labels.size).reshape(labels.size, 2)
        tree = DecisionTree(criterion="entropy").fit(Xd, labels)
        assert tree.n_leaves_ == 1, tree.n_leaves_
        assert np.all(tree.predict(Xd) == labels[0])
    print("  (d) pure node -> one constant leaf  -> PASS")
    print("  All sanity checks passed.\n")


# ---------------------------------------------------------------------------
# Part 1 experiments
# ---------------------------------------------------------------------------


def fit_and_score(criterion, max_depth, X, species, family, train, test, mapping):
    tree = DecisionTree(criterion=criterion, max_depth=max_depth)
    t0 = time.perf_counter()
    tree.fit(X[train], species[train])
    fit_time = time.perf_counter() - t0

    pred_train = tree.predict(X[train])
    pred_test = tree.predict(X[test])
    fam_test = derive_family(pred_test, mapping)

    return {
        "tree": tree,
        "fit_time": fit_time,
        "train": accuracy(species[train], pred_train),
        "test": accuracy(species[test], pred_test),
        "family": accuracy(family[test], fam_test),
        "pred_test": pred_test,
    }


def main():
    sanity_checks()

    X, species, genus, family, record_id, feature_names = load_data()
    train, test = stratified_split(species)

    print("Data: %d rows, %d features" % X.shape)
    print("Split: %d train / %d test rows (stratified on Species, seed 42)"
          % (train.size, test.size))
    print("Classes: %d species, %d families, %d genera\n"
          % (np.unique(species).size, np.unique(family).size, np.unique(genus).size))

    mapping = species_family_map(species[train], family[train])
    unknown = set(np.unique(species[test])) - set(mapping)
    assert not unknown, "test species unseen in training: %s" % unknown

    results = {}
    for criterion in ("gini", "entropy"):
        results[criterion] = fit_and_score(criterion, None, X, species, family,
                                           train, test, mapping)

    print("Single Decision Tree, unconstrained (results matrix rows)")
    print("  %-8s %10s %10s %12s %8s %8s %8s"
          % ("criterion", "train %", "test %", "family %", "leaves", "depth", "fit s"))
    for criterion in ("gini", "entropy"):
        r = results[criterion]
        print("  %-8s %10s %10s %12s %8d %8d %8.2f"
              % (criterion, pct(r["train"]), pct(r["test"]), pct(r["family"]),
                 r["tree"].n_leaves_, r["tree"].depth_, r["fit_time"]))
    print()

    print("Depth sweep (bias/variance check, not part of the matrix)")
    print("  %-8s %6s %10s %10s %8s %8s %8s"
          % ("criterion", "depth", "train %", "test %", "leaves", "depth", "fit s"))
    for criterion in ("gini", "entropy"):
        for depth in (1, 2, 3, 5, 8, 12, None):
            if depth is None:
                r = results[criterion]
            else:
                r = fit_and_score(criterion, depth, X, species, family,
                                  train, test, mapping)
            label = "None" if depth is None else str(depth)
            print("  %-8s %6s %10s %10s %8d %8d %8.2f"
                  % (criterion, label, pct(r["train"]), pct(r["test"]),
                     r["tree"].n_leaves_, r["tree"].depth_, r["fit_time"]))
    print()

    print("Test-set species accuracy (single tree):  gini %s%%  entropy %s%%"
          % (pct(results["gini"]["test"]), pct(results["entropy"]["test"])))


if __name__ == "__main__":
    main()
