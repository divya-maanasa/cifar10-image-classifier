"""
evaluate.py
-----------
Comprehensive evaluation of the trained CIFAR-10 CNN.

Usage
-----
    python evaluate.py
    python evaluate.py --model_path saved_model/cifar10_model.keras
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support)

from data_loader import load_data, CLASS_NAMES, NUM_CLASSES

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")


def load_model_and_stats(model_path):
    model = tf.keras.models.load_model(model_path)
    stats_path = os.path.join(os.path.dirname(model_path), "norm_stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    mean = np.array(stats["mean"])
    std  = np.array(stats["std"])
    return model, mean, std


def normalize(x, mean, std):
    return (x - mean) / (std + 1e-7)


def get_predictions(model, x_test, mean, std, batch_size=256):
    x_norm = normalize(x_test, mean, std)
    probs  = model.predict(x_norm, batch_size=batch_size, verbose=0)
    preds  = np.argmax(probs, axis=1)
    return preds, probs


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion Matrix — CIFAR-10", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def plot_per_class_accuracy(cm, save_path):
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(CLASS_NAMES, per_class_acc * 100, color="steelblue", edgecolor="black")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy")
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 100 + 1,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Per-class accuracy saved → {save_path}")


def plot_sample_predictions(x_test, y_true, y_pred, probs, save_path, n=25):
    """Plot a 5×5 grid of test samples with true vs predicted labels."""
    indices = np.random.choice(len(x_test), n, replace=False)
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for idx, ax in zip(indices, axes.flatten()):
        ax.imshow(x_test[idx])
        true_lbl  = CLASS_NAMES[y_true[idx]]
        pred_lbl  = CLASS_NAMES[y_pred[idx]]
        confidence = probs[idx, y_pred[idx]] * 100
        color = "green" if y_true[idx] == y_pred[idx] else "red"
        ax.set_title(f"T: {true_lbl}\nP: {pred_lbl} ({confidence:.0f}%)",
                     color=color, fontsize=7)
        ax.axis("off")
    plt.suptitle("Sample Predictions (green=correct, red=wrong)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Sample predictions saved → {save_path}")


def evaluate(model_path):
    # ── Load ─────────────────────────────────────────────────────────────────
    print("Loading model …")
    model, mean, std = load_model_and_stats(model_path)

    print("Loading test data …")
    _, _, (x_test, y_test_ohe) = load_data()
    y_true = np.argmax(y_test_ohe, axis=1)

    # ── Predict ───────────────────────────────────────────────────────────────
    print("Running inference …")
    y_pred, probs = get_predictions(model, x_test, mean, std)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted")

    print("\n========== Evaluation Results ==========")
    print(f"Overall Accuracy : {acc * 100:.2f} %")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall  : {recall:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # ── Plots ─────────────────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm,
                          os.path.join(SAVE_DIR, "confusion_matrix.png"))
    plot_per_class_accuracy(cm,
                             os.path.join(SAVE_DIR, "per_class_accuracy.png"))

    # Reload original (un-normalized) test images for visualization
    (_, _), (_, _), (x_test_raw, _) = load_data()
    # x_test_raw is float32 in [0,1]; imshow-compatible
    plot_sample_predictions(x_test_raw, y_true, y_pred, probs,
                            os.path.join(SAVE_DIR, "sample_predictions.png"))

    # ── Save metrics JSON ────────────────────────────────────────────────────
    metrics = {
        "accuracy": round(float(acc), 4),
        "weighted_precision": round(float(precision), 4),
        "weighted_recall": round(float(recall), 4),
        "weighted_f1": round(float(f1), 4),
    }
    with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {SAVE_DIR}/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 CNN")
    parser.add_argument(
        "--model_path",
        default=os.path.join(SAVE_DIR, "cifar10_model.keras"),
        help="Path to the saved Keras model",
    )
    args = parser.parse_args()
    evaluate(args.model_path)
