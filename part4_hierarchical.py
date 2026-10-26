"""BITS F464 Assignment 1 -- Part 4: hierarchical decision tree.

Runs with:  python part4_hierarchical.py
from the folder containing Frogs_MFCCs.csv.  Uses only numpy/stdlib plus the
Part 1 DecisionTree.

A router predicts Family; one expert per family (trained on that family's
training rows only, with the Species label) predicts the species inside it.
A single sample wrongly routed can never get the right species, so overall
accuracy = router accuracy - pA, where pA is the fraction of samples whose
family was right but whose expert was wrong.
"""

import numpy as np

from common import (
    accuracy,
    confusion_matrix,
    derive_family,
    load_data,
    pct,
    species_family_map,
    stratified_split,
)
from part1_decision_tree import DecisionTree


# ---------------------------------------------------------------------------
# Hierarchical classifier
# ---------------------------------------------------------------------------


class HierarchicalClassifier:
    """Router over families + one expert per family."""

    def __init__(self, router, experts):
        self.router = router
        self.experts = dict(experts)
        self.families_ = np.array(sorted(self.experts.keys()))
        self.classes_ = np.unique(np.concatenate(
            [np.asarray(expert.classes_) for expert in self.experts.values()]))

    def predict_family(self, X):
        return self.router.predict(X)

    def _predict_grouped(self, X, families):
        """Predict species after grouping rows by a given family assignment.

        Each expert is called once on its whole group, and predictions are
        written back into the original row positions.
        """
        X = np.asarray(X, dtype=np.float64)
        families = np.asarray(families)
        if not np.all(np.isin(families, self.families_)):
            raise ValueError("unknown family in routing assignment")
        out = np.empty(families.shape[0], dtype=self.classes_.dtype)
        for family in self.families_:           # at most 4 calls, not per row
            mask = families == family
            if np.any(mask):
                out[mask] = self.experts[family].predict(X[mask])
        return out

    def predict(self, X):
        """Route every row, then send each group to its expert."""
        families = self.predict_family(X)
        return self._predict_grouped(X, families)

    def predict_oracle(self, X, true_families):
        """Oracle: route each row to the expert of its true family."""
        return self._predict_grouped(X, true_families)

    def predict_one_by_one(self, X):
        """Reference implementation (per-row loop), used only by sanity checks."""
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for i in range(X.shape[0]):
            family = self.router.predict(X[i:i + 1])[0]
            out[i] = self.experts[family].predict(X[i:i + 1])[0]
        return out


def train_hierarchy(X, species, family, train, criterion):
    """Router on all training rows; experts on true-family training subsets."""
    router = DecisionTree(criterion=criterion,
                          rng=np.random.RandomState(42)).fit(X[train], family[train])

    experts = {}
    for fam in np.unique(family[train]):
        mask = family[train] == fam
        experts[fam] = DecisionTree(
            criterion=criterion,
            rng=np.random.RandomState(42)).fit(X[train][mask], species[train][mask])

    return HierarchicalClassifier(router, experts)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def print_confusion(cm, labels, title, short_columns=False):
    labels = [str(label) for label in labels]
    row_width = max([len("true \\ pred")] + [len(label) for label in labels]) + 2
    if short_columns:
        width = 8
    else:
        width = max(9, max(len(label) for label in labels) + 2)
    print(title)
    header = "true \\ pred".ljust(row_width)
    for i in range(len(labels)):
        header += (str(i) if short_columns else labels[i]).rjust(width)
    print(header)
    for label, row in zip(labels, cm):
        line = label.ljust(row_width)
        line += "".join(str(int(v)).rjust(width) for v in row)
        print(line)


def print_species_legend(labels):
    print("  species index:")
    for i, label in enumerate(labels):
        print("    %2d = %s" % (i, label))


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def sanity_checks():
    print("Sanity checks")
    rng = np.random.RandomState(0)

    # Synthetic hierarchy: family A has species a0, a1; family B has b0 only.
    n = 120
    family = np.array(["A"] * 80 + ["B"] * 40)
    species = np.array(["a0"] * 40 + ["a1"] * 40 + ["b0"] * 40)
    X = np.zeros((n, 3))
    X[:80, 0] = rng.normal(0.0, 0.5, 80)
    X[80:, 0] = rng.normal(6.0, 0.5, 40)
    X[:40, 1] = rng.normal(0.0, 0.5, 40)
    X[40:80, 1] = rng.normal(6.0, 0.5, 40)
    X[:, 2] = rng.normal(size=n)

    clf = train_hierarchy(X, species, family, np.arange(n), "gini")
    assert accuracy(family, clf.predict_family(X)) == 1.0
    assert accuracy(species, clf.predict(X)) == 1.0
    print("  synthetic hierarchy: router and species train acc = 100.00%  -> PASS")

    # A family with a single species gives a constant expert.
    assert clf.experts["B"].classes_.size == 1
    assert np.all(clf.experts["B"].predict(X[family == "B"]) == "b0")
    print("  single-species family -> constant expert  -> PASS")

    # Grouped prediction preserves the original row order.
    perm = rng.permutation(n)
    permuted = clf.predict(X[perm])
    assert np.array_equal(permuted[np.argsort(perm)], clf.predict(X))
    print("  grouped predictions returned in original order  -> PASS")

    # Grouped prediction agrees with a slow per-row reference.
    assert np.array_equal(clf.predict(X), clf.predict_one_by_one(X))
    print("  grouped == per-row reference predictions  -> PASS")

    # With a perfect router, oracle predictions are just the normal ones.
    assert np.array_equal(clf.predict(X),
                          clf.predict_oracle(X, family))
    print("  perfect router: overall == oracle  -> PASS")
    print("  All sanity checks passed.\n")


# ---------------------------------------------------------------------------
# Part 4 experiments
# ---------------------------------------------------------------------------


def main():
    sanity_checks()

    X, species, genus, family, record_id, feature_names = load_data()
    train, test = stratified_split(species)
    mapping = species_family_map(species[train], family[train])

    print("Data: %d rows, %d features; split %d train / %d test"
          % (X.shape[0], X.shape[1], train.size, test.size))
    print("Hierarchy: %d families -> %d species\n"
          % (np.unique(family).size, np.unique(species).size))

    print("Training families and their species (experts are trained per family):")
    for fam in np.unique(family[train]):
        fam_species = np.unique(species[train][family[train] == fam])
        note = "  (single species -> constant expert)" if fam_species.size == 1 else ""
        print("  %-18s %2d species%s" % (fam, fam_species.size, note))
    print()

    results = {}
    for criterion in ("gini", "entropy"):
        print("Training hierarchical model (%s) ..." % criterion)
        clf = train_hierarchy(X, species, family, train, criterion)

        fam_pred_train = clf.predict_family(X[train])
        sp_pred_train = clf.predict(X[train])
        fam_pred_test = clf.predict_family(X[test])
        sp_pred_test = clf.predict(X[test])
        oracle_pred = clf.predict_oracle(X[test], family[test])

        router_correct = fam_pred_test == family[test]
        p_a = float(np.mean(router_correct & (sp_pred_test != species[test])))
        p_b = float(np.mean(~router_correct))

        flat = DecisionTree(criterion=criterion,
                            rng=np.random.RandomState(42)).fit(X[train], species[train])
        flat_pred = flat.predict(X[test])

        results[criterion] = {
            "clf": clf,
            "router_train": accuracy(family[train], fam_pred_train),
            "router_test": accuracy(family[test], fam_pred_test),
            "overall_train": accuracy(species[train], sp_pred_train),
            "overall_test": accuracy(species[test], sp_pred_test),
            "oracle": accuracy(species[test], oracle_pred),
            "pA": p_a,
            "pB": p_b,
            "fam_pred_test": fam_pred_test,
            "sp_pred_test": sp_pred_test,
            "oracle_pred": oracle_pred,
            "flat_test": accuracy(species[test], flat_pred),
            "flat_family": accuracy(family[test],
                                    derive_family(flat_pred, mapping)),
        }
    print()

    print("Part 4 results (results matrix rows)")
    print("  %-8s %14s %14s %14s %14s"
          % ("criterion", "species train %", "species test %",
             "router train %", "router test %"))
    for criterion in ("gini", "entropy"):
        r = results[criterion]
        print("  %-8s %14s %14s %14s %14s"
              % (criterion, pct(r["overall_train"]), pct(r["overall_test"]),
                 pct(r["router_train"]), pct(r["router_test"])))
    print()

    print("Table 3: hierarchical failure analysis (test set, %)")
    print("  %-8s %11s %12s %34s %16s"
          % ("criterion", "router acc", "oracle acc",
             "pA (router right, expert wrong)", "pB (router wrong)"))
    for criterion in ("gini", "entropy"):
        r = results[criterion]
        print("  %-8s %11s %12s %34s %16s"
              % (criterion, pct(r["router_test"]), pct(r["oracle"]),
                 pct(r["pA"]), pct(r["pB"])))
    print()

    print("Check A_overall = A_router - pA (test)")
    for criterion in ("gini", "entropy"):
        r = results[criterion]
        lhs = r["overall_test"]
        rhs = r["router_test"] - r["pA"]
        assert abs(lhs - rhs) < 1e-12
        print("  %-8s %.6f = %.6f - %.6f  -> PASS"
              % (criterion, lhs, r["router_test"], r["pA"]))
    print()

    print("Comparison with the flat single tree (test set)")
    print("  %-8s %14s %14s %10s %14s %14s %10s"
          % ("criterion", "flat species %", "hier species %", "delta",
             "flat family %", "router %", "delta"))
    for criterion in ("gini", "entropy"):
        r = results[criterion]
        print("  %-8s %14s %14s %+10.2f %14s %14s %+10.2f"
              % (criterion, pct(r["flat_test"]), pct(r["overall_test"]),
                 100.0 * (r["overall_test"] - r["flat_test"]),
                 pct(r["flat_family"]), pct(r["router_test"]),
                 100.0 * (r["router_test"] - r["flat_family"])))
    print()

    print("Where the losses are (test set, percentage points)")
    for criterion in ("gini", "entropy"):
        r = results[criterion]
        print("  %-8s router mistakes cost %.2f (oracle %.2f -> overall %.2f);"
              " oracle experts vs flat %+.2f"
              % (criterion, 100.0 * (r["oracle"] - r["overall_test"]),
                 100.0 * r["oracle"], 100.0 * r["overall_test"],
                 100.0 * (r["oracle"] - r["flat_test"])))
    print()

    # Detailed per-criterion diagnostics.
    family_labels = np.unique(family)
    species_labels = np.unique(species)
    for criterion in ("gini", "entropy"):
        r = results[criterion]
        print("=" * 70)
        print("Detailed test-set diagnostics: %s" % criterion)
        print("=" * 70)

        router_cm = confusion_matrix(family[test], r["fam_pred_test"],
                                     labels=family_labels)
        species_cm = confusion_matrix(species[test], r["sp_pred_test"],
                                      labels=species_labels)

        print()
        print_confusion(router_cm, family_labels,
                        "Router confusion matrix (rows true, columns predicted):")
        print()
        print_confusion(species_cm, species_labels,
                        "Final species confusion matrix (rows true, columns predicted;"
                        " column indices):", short_columns=True)
        print_species_legend(species_labels)

        print()
        print("Per-family test accuracy (final species prediction):")
        for fam in family_labels:
            mask = family[test] == fam
            print("  %-18s n = %4d  accuracy = %s%%"
                  % (fam, int(mask.sum()),
                     pct(accuracy(species[test][mask], r["sp_pred_test"][mask]))))

        print()
        print("  Router accuracy   (test) = %s%%" % pct(r["router_test"]))
        print("  Overall accuracy  (test) = %s%%" % pct(r["overall_test"]))
        print("  Oracle accuracy   (test) = %s%%" % pct(r["oracle"]))
        print("  pA (router right, expert wrong) = %s%%" % pct(r["pA"]))
        print("  pB (router wrong)               = %s%%" % pct(r["pB"]))
        print("  A_overall = A_router - pA: %.6f = %.6f - %.6f"
              % (r["overall_test"], r["router_test"], r["pA"]))
        print()


if __name__ == "__main__":
    main()
