import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==============================
# CONFIG
# ==============================

IMAGE_FOLDER = "data/day_dataset"
GT_FILE = "data/ground_truth.csv"
OUTPUT_DIR = "output/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# IMPORT YOUR FUNCTION
# ==============================

from violation import detect_violations

# ==============================
# LABELS
# ==============================

LABELS = ["No Helmet", "Triple Riding", "No Plate", "None"]

# ==============================
# HELPERS
# ==============================

def get_primary_violation(vlist):
    if not vlist:
        return "None"
    return vlist[0]


def extract_confidence(res):
    return (
        res.get("confidence")
        or res.get("ocr_conf")
        or res.get("ocr_confidence")
        or 0
    )


# ==============================
# METRICS VS CONFIDENCE
# ==============================

def metrics_vs_confidence(y_true, y_pred, conf_scores):

    thresholds = np.linspace(0.1, 1.0, 10)

    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []

    for t in thresholds:

        filtered_true = []
        filtered_pred = []

        for yt, yp, c in zip(y_true, y_pred, conf_scores):
            if c >= t:
                filtered_true.append(yt)
                filtered_pred.append(yp)

        if len(filtered_true) > 0:

            precision = precision_score(filtered_true, filtered_pred, average='weighted', zero_division=0)
            recall = recall_score(filtered_true, filtered_pred, average='weighted', zero_division=0)
            f1 = f1_score(filtered_true, filtered_pred, average='weighted', zero_division=0)
            acc = accuracy_score(filtered_true, filtered_pred)

        else:
            precision, recall, f1, acc = 0, 0, 0, 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        accuracy_list.append(acc)

    # Plot all metrics
    plt.figure()
    plt.plot(thresholds, precision_list, label="Precision")
    plt.plot(thresholds, recall_list, label="Recall")
    plt.plot(thresholds, f1_list, label="F1 Score")
    plt.plot(thresholds, accuracy_list, label="Accuracy")

    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.title("Metrics vs Confidence")
    plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "metrics_vs_conf.png"))
    plt.close()

    print("Saved metrics vs confidence curve.")


# ==============================
# MAIN EVALUATION
# ==============================

def evaluate():

    print("\n===== STARTING EVALUATION =====\n")

    gt_df = pd.read_csv(GT_FILE)

    y_true = []
    y_pred = []
    conf_scores = []

    for _, row in gt_df.iterrows():

        img_name = row["image"]
        true_label = row["violation"]

        img_path = os.path.join(IMAGE_FOLDER, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping {img_name}")
            continue

        results = detect_violations(img)

        if len(results) > 0:
            res = results[0]

            pred_label = get_primary_violation(res.get("violations", []))
            conf = extract_confidence(res)

        else:
            pred_label = "None"
            conf = 0

        y_true.append(true_label)
        y_pred.append(pred_label)
        conf_scores.append(conf)

    # ==============================
    # BASIC METRICS
    # ==============================

    print("\nAccuracy:", accuracy_score(y_true, y_pred))

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    # ==============================
    # CONFUSION MATRIX
    # ==============================

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.colorbar()

    plt.xticks(range(len(LABELS)), LABELS, rotation=45)
    plt.yticks(range(len(LABELS)), LABELS)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    print("Saved confusion matrix.")

    # ==============================
    # CONFIDENCE HISTOGRAM
    # ==============================

    plt.figure()
    plt.hist(conf_scores, bins=20)
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")

    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_hist.png"))
    plt.close()

    print("Saved confidence histogram.")

    # ==============================
    # THRESHOLD CURVE
    # ==============================

    thresholds = np.linspace(0.1, 1.0, 10)
    counts = [sum(c >= t for c in conf_scores) for t in thresholds]

    plt.figure()
    plt.plot(thresholds, counts)
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Detections")
    plt.title("Threshold vs Detections")

    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_curve.png"))
    plt.close()

    print("Saved threshold curve.")

    # ==============================
    # METRICS VS CONFIDENCE 🔥
    # ==============================

    metrics_vs_confidence(y_true, y_pred, conf_scores)

    print("\n===== EVALUATION COMPLETE =====\n")


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    evaluate()