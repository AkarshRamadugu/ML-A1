"""Shared helpers for BITS F464 Assignment 1 (Anuran Calls, MFCCs).

Everything here uses numpy / pandas / the standard library only.
"""

import numpy as np
import pandas as pd

CSV_PATH = "Frogs_MFCCs.csv"


def load_data(csv_path=CSV_PATH):
    """Load the dataset, returning the 22 MFCC inputs and the label columns.

    Returns
    -------
    X : (n, 22) float array, MFCCs_1 .. MFCCs_22 only
    species, genus, family : (n,) arrays of string labels
    record_id : (n,) array
    feature_names : list[str]
    """
    df = pd.read_csv(csv_path)

    # The UCI file spells the columns "MFCCs_ 1" ... "MFCCs_ 22".
    feature_names = [c for c in df.columns if c.startswith("MFCCs_")]
    if len(feature_names) != 22:
        raise ValueError("expected 22 MFCC columns, found %d" % len(feature_names))

    X = df[feature_names].to_numpy(dtype=np.float64)
    species = df["Species"].to_numpy()
    genus = df["Genus"].to_numpy()
    family = df["Family"].to_numpy()
    record_id = df["RecordID"].to_numpy()
    return X, species, genus, family, record_id, feature_names


def stratified_split(labels, test_frac=0.2, seed=42):
    """The exact split function mandated by the assignment."""
    rng = np.random.RandomState(seed)
    test = []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        test.extend(idx[: int(round(test_frac * len(idx)))])
    test = np.sort(np.array(test))
    train = np.setdiff1d(np.arange(len(labels)), test)
    return train, test


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    """Confusion matrix as a numpy array; rows are true, columns predicted."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    return cm


def species_family_map(species_train, family_train):
    """Species -> family mapping, read off the training labels."""
    mapping = {}
    for sp, fam in zip(species_train, family_train):
        if sp in mapping and mapping[sp] != fam:
            raise ValueError("species %r appears in two families" % sp)
        mapping[sp] = fam
    return mapping


def derive_family(species_pred, mapping):
    """Map predicted species to their family."""
    return np.array([mapping[sp] for sp in species_pred])


def pct(x):
    """Format an accuracy as a percentage with two decimals."""
    return "%.2f" % (100.0 * x)


def majority_vote(predictions, classes):
    """Majority vote over rows of `predictions` (n_models, n_samples).

    Ties are broken towards the smallest class label: `classes` must be
    sorted, and np.argmax returns the first (smallest) index on equal counts.
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
    """Weighted vote: argmax_k sum_t w_t 1[h_t = k]; ties -> smallest label."""
    predictions = np.asarray(predictions)
    classes = np.asarray(classes)
    n_models, n_samples = predictions.shape
    class_idx = np.searchsorted(classes, predictions)

    scores = np.zeros((n_samples, classes.size), dtype=np.float64)
    rows = np.arange(n_samples)
    for t in range(n_models):
        scores[rows, class_idx[t]] += weights[t]
    return classes[np.argmax(scores, axis=1)]
