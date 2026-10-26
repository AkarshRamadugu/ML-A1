"""Internal helper: recompute every number and build results_matrix.pdf.

This is not part of the submission zip; it exists so the PDF is generated
from the same deterministic code paths that the five part scripts use.
"""

import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from common import (accuracy, derive_family, load_data, species_family_map,
                    stratified_split)
from part1_decision_tree import DecisionTree
from part2_bagging_boosting import AdaBoostSAMME, BaggingClassifier
from part3_random_forest import RandomForest
from part4_hierarchical import train_hierarchy
from part5_naive_bayes import GaussianNB

STUDENT_ID = "2024ADPS0196H"
OUT_PATH = "results_matrix.pdf"


def pct(value):
    return "%.2f" % (100.0 * value)


def main():
    X, species, genus, family, record_id, feature_names = load_data()
    train, test = stratified_split(species)
    mapping = species_family_map(species[train], family[train])

    def flat_scores(model):
        pred_train = model.predict(X[train])
        pred_test = model.predict(X[test])
        return (accuracy(species[train], pred_train),
                accuracy(species[test], pred_test),
                accuracy(family[test], derive_family(pred_test, mapping)))

    # ---- Table 2 numbers -------------------------------------------------
    tree_scores = {}
    for criterion in ("gini", "entropy"):
        tree = DecisionTree(criterion=criterion,
                            rng=np.random.RandomState(42)).fit(X[train], species[train])
        tree_scores[criterion] = flat_scores(tree)

    bag_scores = {}
    ada_scores = {}
    for criterion in ("gini", "entropy"):
        bag = BaggingClassifier(n_estimators=25, criterion=criterion,
                                rng=np.random.RandomState(42)).fit(X[train], species[train])
        bag_scores[criterion] = flat_scores(bag)

        ada = AdaBoostSAMME(n_estimators=50, criterion=criterion, max_depth=3,
                            rng=np.random.RandomState(42)).fit(X[train], species[train])
        ada_scores[criterion] = flat_scores(ada)

    rf_scores = {}
    for criterion in ("gini", "entropy"):
        forest = RandomForest(n_estimators=50, criterion=criterion,
                              max_features=None,
                              rng=np.random.RandomState(42)).fit(X[train], species[train])
        rf_scores[criterion] = flat_scores(forest)

    hier = {}
    for criterion in ("gini", "entropy"):
        clf = train_hierarchy(X, species, family, train, criterion)
        fam_pred_test = clf.predict_family(X[test])
        sp_pred_test = clf.predict(X[test])
        oracle_pred = clf.predict_oracle(X[test], family[test])
        router_correct = fam_pred_test == family[test]

        hier[criterion] = {
            "train": accuracy(species[train], clf.predict(X[train])),
            "test": accuracy(species[test], sp_pred_test),
            "router_train": accuracy(family[train], clf.predict_family(X[train])),
            "router_test": accuracy(family[test], fam_pred_test),
            "oracle": accuracy(species[test], oracle_pred),
            "pA": float(np.mean(router_correct & (sp_pred_test != species[test]))),
            "pB": float(np.mean(~router_correct)),
        }

    nb = GaussianNB(var_smoothing=1e-9).fit(X[train], species[train])
    pred_train = nb.predict(X[train])
    pred_test = nb.predict(X[test])
    nb_scores = (accuracy(species[train], pred_train),
                 accuracy(species[test], pred_test),
                 accuracy(family[test], derive_family(pred_test, mapping)))

    # ---- Build the document ---------------------------------------------
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold",
                           fontSize=16, leading=20, spaceAfter=2)
    subtitle = ParagraphStyle("subtitle", parent=styles["Normal"],
                              fontName="Helvetica", fontSize=11, leading=14,
                              alignment=TA_CENTER, textColor=colors.HexColor("#404040"))
    section = ParagraphStyle("section", parent=styles["Heading2"],
                             fontName="Helvetica-Bold", fontSize=12, leading=15,
                             spaceBefore=12, spaceAfter=5,
                             textColor=colors.HexColor("#1F3864"))
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=9.5, leading=13)
    note = ParagraphStyle("note", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=8, leading=10.5, textColor=colors.HexColor("#404040"))
    cell = ParagraphStyle("cell", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=9, leading=11)
    cell_c = ParagraphStyle("cell_c", parent=cell, alignment=TA_CENTER)
    hdr = ParagraphStyle("hdr", parent=styles["Normal"], fontName="Helvetica-Bold",
                         fontSize=8.5, leading=10, alignment=TA_CENTER,
                         textColor=colors.white)

    doc = SimpleDocTemplate(OUT_PATH, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=14 * mm,
                            title="BITS F464 Assignment 1 - Results Matrix",
                            author=STUDENT_ID)
    story = []
    story.append(Paragraph("BITS F464: Machine Learning", title))
    story.append(Paragraph("Assignment 1: Flat vs. Hierarchical Classification From Scratch",
                           subtitle))
    story.append(Paragraph("Semester I, Academic Year 2026-27 &nbsp;&nbsp;|&nbsp;&nbsp;"
                           "Student ID: %s" % STUDENT_ID, subtitle))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Results Matrix", section))
    story.append(Paragraph(
        "Dataset: Anuran Calls (MFCCs), 7,195 records, 22 MFCC features. "
        "Stratified 80/20 split on Species (seed 42): 5,756 training / 1,439 test rows. "
        "Flat models predict Species directly; hierarchical models route Family first, "
        "then Species. Accuracies are percentages; 100.00 means every training row correct.",
        body))

    data = [
        [Paragraph("Method", hdr), Paragraph("Criterion", hdr),
         Paragraph("Overall accuracy (Species)", hdr), "",
         Paragraph("Family acc. (Test)", hdr)],
        ["", "", Paragraph("Train", hdr), Paragraph("Test", hdr), ""],
        ["1. Single Decision Tree", "Gini"] + [pct(v) for v in tree_scores["gini"]],
        ["", "Entropy"] + [pct(v) for v in tree_scores["entropy"]],
        ["2a. Bagging", "Gini"] + [pct(v) for v in bag_scores["gini"]],
        ["", "Entropy"] + [pct(v) for v in bag_scores["entropy"]],
        ["2b. AdaBoost", "Gini"] + [pct(v) for v in ada_scores["gini"]],
        ["", "Entropy"] + [pct(v) for v in ada_scores["entropy"]],
        ["3. Random Forest", "Gini"] + [pct(v) for v in rf_scores["gini"]],
        ["", "Entropy"] + [pct(v) for v in rf_scores["entropy"]],
        ["4. Hierarchical Decision Tree", "Gini"] + [pct(v) for v in
                                                     (hier["gini"]["train"], hier["gini"]["test"])] +
        ["Router: " + pct(hier["gini"]["router_test"])],
        ["", "Entropy"] + [pct(v) for v in
                           (hier["entropy"]["train"], hier["entropy"]["test"])] +
        ["Router: " + pct(hier["entropy"]["router_test"])],
        ["5. Gaussian Naive Bayes", "-"] + [pct(v) for v in nb_scores],
    ]

    table2 = Table(data, colWidths=[148, 68, 90, 90, 97], repeatRows=2)
    table2.setStyle(TableStyle([
        ("SPAN", (0, 0), (0, 1)),
        ("SPAN", (1, 0), (1, 1)),
        ("SPAN", (2, 0), (3, 0)),
        ("SPAN", (4, 0), (4, 1)),
        ("SPAN", (0, 2), (0, 3)),
        ("SPAN", (0, 4), (0, 5)),
        ("SPAN", (0, 6), (0, 7)),
        ("SPAN", (0, 8), (0, 9)),
        ("SPAN", (0, 10), (0, 11)),
        ("BACKGROUND", (0, 0), (-1, 1), colors.HexColor("#1F3864")),
        ("BACKGROUND", (4, 10), (4, 11), colors.HexColor("#D9D9D9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#7F7F7F")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(Spacer(1, 6))
    story.append(table2)
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "&dagger; For flat models, Family acc. is the derived family accuracy: each predicted "
        "species is mapped to its family using the species-to-family mapping read from the "
        "training data. For the hierarchical rows the shaded Family acc. cell contains the "
        "router accuracy, while the Overall accuracy columns contain the final species "
        "accuracy.", note))

    story.append(Paragraph("Table 3: Supplementary Hierarchical Failure Analysis (Test Set, %)",
                           section))
    data3 = [
        [Paragraph("Criterion", hdr), Paragraph("Router acc.", hdr),
         Paragraph("Oracle acc.", hdr),
         Paragraph("(A) Router right,<br/>expert wrong", hdr),
         Paragraph("(B) Router<br/>wrong", hdr)],
        ["Gini", pct(hier["gini"]["router_test"]), pct(hier["gini"]["oracle"]),
         pct(hier["gini"]["pA"]), pct(hier["gini"]["pB"])],
        ["Entropy", pct(hier["entropy"]["router_test"]), pct(hier["entropy"]["oracle"]),
         pct(hier["entropy"]["pA"]), pct(hier["entropy"]["pB"])],
    ]
    table3 = Table(data3, colWidths=[80, 82, 82, 166, 83])
    table3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#7F7F7F")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table3)
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "A_overall = A_router - pA and pB = 1 - A_router hold exactly at full precision "
        "for both criteria. Oracle accuracy uses the expert of the true family for every test row.",
        note))

    story.append(Paragraph("Fixed Settings", section))
    story.append(Paragraph(
        "Single tree: unconstrained. Bagging: T = 25 bootstrap trees, all 22 features, "
        "majority vote. AdaBoost: multi-class SAMME, T = 50, depth-3 weak learners. "
        "Random forest: T = 50, m = floor(sqrt(22)) = 4 features per split, majority vote. "
        "Hierarchical: one unconstrained router and one unconstrained expert per family, "
        "same criterion. Gaussian NB: variance smoothing 1e-9 * max feature variance. "
        "All randomness comes from np.random.RandomState(42), one fresh generator per model; "
        "the split has its own seed-42 generator.", body))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Generated from the submitted implementations; the five part scripts reproduce these "
        "numbers with one command each (see README.txt).", note))

    doc.build(story)
    print("wrote", OUT_PATH)

    # Echo the numbers so transcription can be checked.
    for name, scores in (("tree", tree_scores), ("bagging", bag_scores),
                         ("adaboost", ada_scores), ("forest", rf_scores)):
        for criterion in ("gini", "entropy"):
            print("%-9s %-8s %s %s %s" % (name, criterion,
                                          *[pct(v) for v in scores[criterion]]))
    for criterion in ("gini", "entropy"):
        h = hier[criterion]
        print("hier      %-8s train %s test %s router %s/%s oracle %s pA %s pB %s"
              % (criterion, pct(h["train"]), pct(h["test"]),
                 pct(h["router_train"]), pct(h["router_test"]), pct(h["oracle"]),
                 pct(h["pA"]), pct(h["pB"])))
    print("nb        -        %s %s %s" % tuple(pct(v) for v in nb_scores))


if __name__ == "__main__":
    main()
