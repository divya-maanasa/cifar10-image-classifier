"""
train.py
--------
Training script for the CIFAR-10 CNN.

Usage
-----
    python train.py --arch cnn --epochs 100 --batch 128
    python train.py --arch resnet --epochs 150 --batch 128
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_data, build_augmentation_pipeline, compute_mean_std, CLASS_NAMES
from model import get_model

# ── reproducibility ─────────────────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")


# ── helpers ──────────────────────────────────────────────────────────────────

def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["accuracy"],     label="train acc")
    axes[0].plot(history["val_accuracy"], label="val acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["loss"],     label="train loss")
    axes[1].plot(history["val_loss"], label="val loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")


def make_tf_dataset(x, y, batch_size, augment_fn=None, shuffle=False):
    """Wrap numpy arrays in a tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=SEED)
    ds = ds.batch(batch_size)
    if augment_fn is not None:
        ds = ds.map(lambda img, lbl: (augment_fn(img, training=True), lbl),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── main ─────────────────────────────────────────────────────────────────────

def train(arch="cnn", epochs=100, batch_size=128, lr=1e-3):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    mean, std = compute_mean_std(x_train)

    aug_pipeline = build_augmentation_pipeline(mean, std)

    train_ds = make_tf_dataset(x_train, y_train, batch_size,
                               augment_fn=aug_pipeline, shuffle=True)
    val_ds   = make_tf_dataset(x_val,   y_val,   batch_size)
    test_ds  = make_tf_dataset(x_test,  y_test,  batch_size)

    # ── 2. Model ─────────────────────────────────────────────────────────────
    model = get_model(arch)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── 3. Callbacks ─────────────────────────────────────────────────────────
    checkpoint_path = os.path.join(SAVE_DIR, "best_model.keras")

    callbacks = [
        # Save the best checkpoint (by val_accuracy)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Reduce LR on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
        # Early stopping (generous patience)
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        # TensorBoard logs
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(SAVE_DIR, "logs"),
            histogram_freq=0,
        ),
    ]

    # ── 4. Training ───────────────────────────────────────────────────────────
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n=== Test Evaluation ===")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy : {test_acc * 100:.2f} %")
    print(f"Test loss     : {test_loss:.4f}")

    # ── 6. Save artifacts ─────────────────────────────────────────────────────
    # Full model (for serving)
    model.save(os.path.join(SAVE_DIR, "cifar10_model.keras"))

    # Normalization stats (needed by inference)
    stats = {"mean": mean.tolist(), "std": std.tolist(), "classes": CLASS_NAMES}
    with open(os.path.join(SAVE_DIR, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Training curves
    plot_history(history.history,
                 os.path.join(SAVE_DIR, "training_curves.png"))

    print(f"\nAll artifacts saved to: {SAVE_DIR}/")
    return model, history


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    parser.add_argument("--arch",   default="cnn",  choices=["cnn", "resnet"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max training epochs")
    parser.add_argument("--batch",  type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr",     type=float, default=1e-3,
                        help="Initial learning rate")
    args = parser.parse_args()

    train(arch=args.arch, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
