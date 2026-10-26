BITS F464: Machine Learning
Assignment 1: Flat vs. Hierarchical Classification From Scratch
Semester I, Academic Year 2026-27
Student ID: 2024ADPS0196H


FILES
-----
common.py                   Shared loader, fixed train/test split, metrics,
                            voting helpers, species-to-family mapping.
part1_decision_tree.py      Single CART decision tree (Gini / entropy), with
                            sample weights, max_depth, max_features, rng.
part2_bagging_boosting.py   Bagging and multi-class SAMME AdaBoost.
part3_random_forest.py      Random forest (per-node feature subsets).
part4_hierarchical.py       Router over families + one expert per family.
part5_naive_bayes.py        Flat Gaussian naive Bayes.
results_matrix.pdf          Filled Table 2 and Table 3 (typed).
Frogs_MFCCs.csv             Dataset: UCI Anuran Calls (MFCCs).


ENVIRONMENT
-----------
Python  3.14.0
numpy   2.4.6
pandas  3.0.3

Only numpy, pandas and the Python standard library are used by the five part
scripts. No scikit-learn, PyTorch, TensorFlow, XGBoost, LightGBM, SciPy or
statsmodels, and no library train/test split, metric or confusion-matrix
helpers. reportlab was used only to typeset results_matrix.pdf; it is not
imported by any part script.


HOW TO RUN
----------
Put all files in one folder (Frogs_MFCCs.csv must be next to the scripts; the
code uses the relative path "Frogs_MFCCs.csv", never an absolute path).

Windows PowerShell:

    cd "path\to\folder"
    python part1_decision_tree.py
    python part2_bagging_boosting.py
    python part3_random_forest.py
    python part4_hierarchical.py
    python part5_naive_bayes.py

Linux / macOS:

    python3 part1_decision_tree.py
    python3 part2_bagging_boosting.py
    python3 part3_random_forest.py
    python3 part4_hierarchical.py
    python3 part5_naive_bayes.py

Each script runs standalone with one command. It prints its own sanity checks
first, then the numbers that go into results_matrix.pdf:

    part1_decision_tree.py    single tree, both criteria, plus depth sweep
    part2_bagging_boosting.py bagging and AdaBoost, both criteria, plus stumps
    part3_random_forest.py    random forest, both criteria, plus tree agreement
    part4_hierarchical.py     router/experts, confusion matrices, failure analysis
    part5_naive_bayes.py      Gaussian NB, correlation matrix, comparison

Approximate runtimes on a laptop: part 1 ~5 s, part 2 ~50 s, part 3 ~30 s,
part 4 ~4 s, part 5 ~1 s.


IMPLEMENTATION NOTES
--------------------
- Split: stratified on Species, 80/20, seed 42, exactly the function given in
  the assignment (common.stratified_split). Result: 5,756 train / 1,439 test
  rows. The test set is used only for final evaluation.
- Features: the 22 MFCCs only. Family, Genus, Species and RecordID are never
  used as inputs.
- Randomness: all models use a fresh np.random.RandomState(42); the split has
  its own seed-42 generator. Candidate thresholds are midpoints between
  consecutive distinct sorted values; gain ties go to the lower feature index,
  then the lower threshold; leaf/vote ties go to the smallest class label.
- Part 4 experts are trained on rows of the TRUE family; inference routes each
  row first and then calls each expert once on its whole group.
