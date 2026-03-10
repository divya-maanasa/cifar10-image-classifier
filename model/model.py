"""
model.py
--------
CNN architectures for CIFAR-10 image classification.

Two options are provided:
  - build_cnn()       : Custom lightweight CNN (fast to train, ~85 % accuracy)
  - build_resnet_like(): Deeper residual-style CNN (~90 %+ with enough epochs)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ---------------------------------------------------------------------------
# Option 1 — Custom Lightweight CNN
# ---------------------------------------------------------------------------

def build_cnn(input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.4):
    """
    A compact but effective CNN for CIFAR-10.

    Architecture summary
    --------------------
    Block 1 : Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout
    Block 2 : Conv(128) → BN → ReLU → Conv(128) → BN → ReLU → MaxPool → Dropout
    Block 3 : Conv(256) → BN → ReLU → Conv(256) → BN → ReLU → MaxPool → Dropout
    Head    : GlobalAvgPool → Dense(512) → BN → ReLU → Dropout → Dense(10, softmax)
    """
    weight_decay = 1e-4

    inputs = tf.keras.Input(shape=input_shape, name="image_input")

    # ---- Block 1 ----
    x = layers.Conv2D(64, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      name="conv1_1")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      name="conv1_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # ---- Block 2 ----
    x = layers.Conv2D(128, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      name="conv2_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(128, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      name="conv2_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # ---- Block 3 ----
    x = layers.Conv2D(256, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      name="conv3_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(256, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      name="conv3_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # ---- Head ----
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="cifar10_cnn")
    return model


# ---------------------------------------------------------------------------
# Option 2 — Residual-style CNN (deeper, higher accuracy)
# ---------------------------------------------------------------------------

def _residual_block(x, filters, stride=1, weight_decay=1e-4):
    """
    Pre-activation residual block (He et al., Identity Mappings in ResNets).
    """
    shortcut = x

    # First sub-layer
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same",
                                 kernel_regularizer=regularizers.l2(weight_decay),
                                 use_bias=False)(x)

    x = layers.Conv2D(filters, 3, strides=stride, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      use_bias=False)(x)

    # Second sub-layer
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      use_bias=False)(x)

    x = layers.Add()([x, shortcut])
    return x


def build_resnet_like(input_shape=(32, 32, 3), num_classes=10):
    """
    A compact ResNet-inspired model for CIFAR-10.
    Achieves ~91-93 % test accuracy with cosine-annealing LR.
    """
    weight_decay = 1e-4
    inputs = tf.keras.Input(shape=input_shape, name="image_input")

    # Initial conv
    x = layers.Conv2D(64, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      use_bias=False, name="stem_conv")(inputs)

    # Stage 1
    for _ in range(2):
        x = _residual_block(x, 64, weight_decay=weight_decay)

    # Stage 2
    x = _residual_block(x, 128, stride=2, weight_decay=weight_decay)
    for _ in range(1):
        x = _residual_block(x, 128, weight_decay=weight_decay)

    # Stage 3
    x = _residual_block(x, 256, stride=2, weight_decay=weight_decay)
    for _ in range(1):
        x = _residual_block(x, 256, weight_decay=weight_decay)

    # Head
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="cifar10_resnet")
    return model


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(architecture="cnn", **kwargs):
    """
    Factory function. architecture ∈ {"cnn", "resnet"}
    """
    builders = {
        "cnn": build_cnn,
        "resnet": build_resnet_like,
    }
    if architecture not in builders:
        raise ValueError(f"Unknown architecture '{architecture}'. Choose from {list(builders)}")
    return builders[architecture](**kwargs)
