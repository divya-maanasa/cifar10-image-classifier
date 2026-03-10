"""
data_loader.py
--------------
Handles CIFAR-10 data loading, preprocessing, and augmentation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# CIFAR-10 class labels
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

NUM_CLASSES = 10
IMG_SHAPE = (32, 32, 3)


def load_data():
    """
    Load and split the CIFAR-10 dataset.

    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
        - Images are float32 normalized to [0, 1]
        - Labels are one-hot encoded
    """
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # One-hot encode labels
    y_train_full = to_categorical(y_train_full, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    # Split training set: 45,000 train / 5,000 validation
    x_train, x_val = x_train_full[:45000], x_train_full[45000:]
    y_train, y_val = y_train_full[:45000], y_train_full[45000:]

    print(f"Train:      {x_train.shape[0]} samples")
    print(f"Validation: {x_val.shape[0]} samples")
    print(f"Test:       {x_test.shape[0]} samples")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def compute_mean_std(x_train):
    """Compute per-channel mean and std for standardization."""
    mean = x_train.mean(axis=(0, 1, 2))
    std = x_train.std(axis=(0, 1, 2))
    return mean, std


def build_augmentation_pipeline(mean, std):
    """
    Build a Keras Sequential preprocessing pipeline with augmentation.

    Args:
        mean: per-channel mean (used for normalization layer)
        std:  per-channel std  (used for normalization layer)

    Returns:
        A tf.keras.Sequential model (preprocessing only)
    """
    pipeline = tf.keras.Sequential([
        # Per-channel normalization
        tf.keras.layers.Normalization(mean=mean, variance=std ** 2),
        # Random horizontal flip
        tf.keras.layers.RandomFlip("horizontal"),
        # Random translation ±10 %
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        # Random rotation ±15 degrees
        tf.keras.layers.RandomRotation(0.042),   # 15/360 ≈ 0.042
        # Random zoom ±10 %
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")
    return pipeline


def preprocess_single_image(image_array, mean=None, std=None):
    """
    Preprocess a single numpy image array for inference.

    Args:
        image_array: numpy array of shape (H, W, 3), uint8 or float32
        mean: per-channel mean (optional; if None, normalizes to [0,1] only)
        std:  per-channel std  (optional)

    Returns:
        Preprocessed numpy array of shape (1, 32, 32, 3) ready for model.predict()
    """
    img = tf.image.resize(image_array, (32, 32)).numpy()
    img = img.astype("float32") / 255.0

    if mean is not None and std is not None:
        img = (img - mean) / (std + 1e-7)

    return np.expand_dims(img, axis=0)   # (1, 32, 32, 3)
