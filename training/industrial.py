import os
import shutil
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, regularizers, constraints
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------------------------
# 0.  Custom augmentation layers  (Prompt-2 integration)
#     Defined here — before Section 1 — so they are available at model-build
#     time and serialisable when the .keras file is saved.
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class AddGaussianNoise(layers.Layer):
    """
    Add zero-mean Gaussian noise with a randomly sampled stddev in
    [min_stddev, max_stddev].  Active only during training (training=True).

    For images normalised to [0, 1], stddev 0.01-0.05 simulates realistic
    camera sensor noise (ISO grain) and JPEG compression artefacts without
    destroying discriminative texture — critical for rust vs. scratch.
    """
    def __init__(self, min_stddev: float = 0.01,
                       max_stddev: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.min_stddev = min_stddev
        self.max_stddev = max_stddev

    def call(self, inputs, training=False):
        if not training:
            return inputs
        # Re-sample stddev every call so the network cannot memorise a
        # fixed noise level and cheat.
        stddev = tf.random.uniform(
            shape=(), minval=self.min_stddev, maxval=self.max_stddev
        )
        noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0,
                                 stddev=stddev, dtype=inputs.dtype)
        return tf.clip_by_value(inputs + noise, 0.0, 1.0)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"min_stddev": self.min_stddev,
                    "max_stddev": self.max_stddev})
        return cfg

@tf.keras.utils.register_keras_serializable()
class RandomGaussianBlur(layers.Layer):
    """
    Apply a random-sigma Gaussian blur via depthwise convolution.
    Simulates:
      - Camera lens defocus / motion blur
      - Low-quality industrial CCTV footage
      - JPEG soft edges after heavy compression

    kernel_size : must be odd (5 recommended for 64×64 input)
    sigma_range : (min, max) stddev of the Gaussian kernel
    prob        : probability of applying blur per image
    """
    def __init__(self, kernel_size: int = 5,
                       sigma_range: tuple = (0.5, 1.5),
                       prob: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.prob        = prob

    def _gaussian_kernel(self, sigma: tf.Tensor) -> tf.Tensor:
        """Build a (k, k, 1, 1) Gaussian kernel for depthwise_conv2d."""
        k    = self.kernel_size
        half = k // 2
        coords = tf.cast(tf.range(-half, half + 1), tf.float32)
        g1d    = tf.exp(-0.5 * (coords / sigma) ** 2)
        g1d    = g1d / tf.reduce_sum(g1d)
        g2d    = tf.einsum('i,j->ij', g1d, g1d)
        g2d    = g2d / tf.reduce_sum(g2d)
        return tf.reshape(g2d, [k, k, 1, 1])

    def call(self, inputs, training=False):
        # Ép training về kiểu tf.bool để tính toán trên Graph
        is_training = tf.convert_to_tensor(training, dtype=tf.bool)

        # Điều kiện: Phải đang training VÀ quay số trúng xác suất áp dụng blur
        should_blur = tf.logical_and(is_training, tf.random.uniform(()) <= self.prob)

        # Định nghĩa luồng xử lý tính toán Blur (Giữ nguyên thuật toán gốc của bạn)
        def perform_blur():
            sigma  = tf.random.uniform(
                (), minval=self.sigma_range[0], maxval=self.sigma_range[1]
            )
            kernel = self._gaussian_kernel(sigma)   # (k, k, 1, 1)

            x = inputs
            was_unbatched = (len(x.shape) == 3)
            if was_unbatched:
                x = tf.expand_dims(x, 0)           # → (1, H, W, C)

            # Blur từng channel độc lập bằng depthwise_conv2d
            channels = tf.unstack(x, axis=-1)
            blurred  = []
            for ch in channels:
                ch_4d = tf.expand_dims(ch, -1)      # (B, H, W, 1)
                ch_bl = tf.nn.depthwise_conv2d(
                    ch_4d, kernel,
                    strides=[1, 1, 1, 1],
                    padding="SAME"
                )
                blurred.append(tf.squeeze(ch_bl, axis=-1))

            out = tf.stack(blurred, axis=-1)         # (B, H, W, C)
            if was_unbatched:
                out = tf.squeeze(out, axis=0)
            return tf.clip_by_value(out, 0.0, 1.0)

        # Sử dụng tf.cond để rẽ nhánh an toàn trên đồ thị tĩnh TensorFlow
        return tf.cond(
            should_blur,
            true_fn=perform_blur,
            false_fn=lambda: inputs
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernel_size": self.kernel_size,
                    "sigma_range": self.sigma_range,
                    "prob":        self.prob})
        return cfg

@tf.keras.utils.register_keras_serializable()
class RandomHueSaturation(layers.Layer):
    """
    Jointly randomise Hue and Saturation via tf.image (no external deps).

    Prevents the model from memorising the specific synthetic "rust orange"
    hex code — teaches it that rust is a TEXTURE, not a colour.

    max_hue_delta : hue shift range [−delta, +delta] (fraction of full
                    colour wheel; 0.05 ≈ ±18°, spans the orange-brown band)
    sat_range     : multiplicative saturation factor [lower, upper]
                    (0.5–1.5 covers greyscale CCTV → vivid real rust)
    """
    def __init__(self, max_hue_delta: float = 0.05,
                       sat_range: tuple = (0.5, 1.5), **kwargs):
        super().__init__(**kwargs)
        self.max_hue_delta = max_hue_delta
        self.sat_range     = sat_range

    def call(self, inputs, training=False):
        if not training:
            return inputs
        x = tf.image.random_hue(inputs,       max_delta=self.max_hue_delta)
        x = tf.image.random_saturation(x,
                lower=self.sat_range[0], upper=self.sat_range[1])
        return tf.clip_by_value(x, 0.0, 1.0)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_hue_delta": self.max_hue_delta,
                    "sat_range":     self.sat_range})
        return cfg


# ---------------------------------------------------------------------------
# 1.  Configuration
# ---------------------------------------------------------------------------
SEED          = 42
IMG_SIZE      = 64
NUM_CLASSES   = 5          # crack / hole / rust / scratch / normal
BATCH_SIZE    = 64
EPOCHS        = 175
LEARNING_RATE = 1e-3
DROPOUT_FLAT  = 0.3        # slightly stronger than fer2013 to fight overfitting
DROPOUT_DENSE = 0.5
L2_CONV       = 1e-4       # L2 on conv kernels (new — industrial data overfits fast)
L2_DENSE      = 1e-3

# --- Google Drive paths (source) ---
DRIVE_TRAIN_DIR      = '/content/drive/MyDrive/DATN/industrial_defect_dataset/train'
DRIVE_VALIDATION_DIR = '/content/drive/MyDrive/DATN/industrial_defect_dataset/val'
WEIGHTS_FILE         = '/content/drive/MyDrive/DATN/industrial_defect_30_7_2023.keras'

# --- Local SSD paths (fast) ---
LOCAL_TRAIN_DIR      = '/content/industrial_local/train'
LOCAL_VALIDATION_DIR = '/content/industrial_local/val'

# Class names MUST match the sub-folder names exactly (case-sensitive)
DEFECT_LABELS = ['crack', 'hole', 'rust', 'scratch', 'normal']

tf.random.set_seed(SEED)
np.random.seed(SEED)

# NOTE: mixed_float16 is intentionally left OFF here.
# The industrial dataset is small enough that float32 training is fast,
# and mixed precision can amplify gradient noise on tiny val batches.
# tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ---------------------------------------------------------------------------
# 2.  Copy dataset from Drive → local SSD
# ---------------------------------------------------------------------------

def copy_to_local(drive_dir: str, local_dir: str, name: str):
    if os.path.exists(local_dir):
        print(f"[INFO] '{local_dir}' đã tồn tại, bỏ qua bước copy.")
        return
    print(f"[INFO] Đang copy {name} từ Drive → SSD cục bộ ...")
    shutil.copytree(drive_dir, local_dir)
    print(f"[INFO] Copy {name} hoàn tất.")


# ---------------------------------------------------------------------------
# 3.  Class weights (computed from train split only)
# ---------------------------------------------------------------------------

def compute_class_weights(train_dir: str) -> dict:
    counts = {}
    for idx, defect in enumerate(DEFECT_LABELS):
        folder = os.path.join(train_dir, defect)
        if os.path.exists(folder):
            counts[idx] = len([
                f for f in os.listdir(folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ])
        else:
            counts[idx] = 1

    total = sum(counts.values())
    weights = {idx: total / (NUM_CLASSES * cnt) for idx, cnt in counts.items()}

    print("[INFO] Class weights:")
    for idx, defect in enumerate(DEFECT_LABELS):
        print(f"  {defect:12s}: {weights[idx]:.4f}  (n={counts[idx]})")
    return weights


# ---------------------------------------------------------------------------
# 4.  Path collection
#
#  train/ and val/ are now split manually beforehand (by folder), so no
#  in-script stratified 50/50 split is needed anymore — each folder is
#  used as-is.
# ---------------------------------------------------------------------------

def _collect_image_paths(root_dir: str):
    """Return (paths, integer_labels) for every image under root_dir."""
    root = pathlib.Path(root_dir)
    paths, labels = [], []
    for idx, defect in enumerate(DEFECT_LABELS):
        class_dir = root / defect
        if not class_dir.exists():
            print(f"[WARN] Folder not found: {class_dir}")
            continue
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}:
                paths.append(str(img_path))
                labels.append(idx)
    return paths, labels


def _paths_to_dataset(paths, labels, batch_size, augment_fn=None):
    """
    Build a tf.data.Dataset from file-path lists.
    Labels are converted to one-hot vectors (NUM_CLASSES).
    """
    path_ds  = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.one_hot(labels, depth=NUM_CLASSES)
    )
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def load_and_preprocess(path, label):
        raw   = tf.io.read_file(path)
        image = tf.image.decode_image(raw, channels=1, expand_animations=False)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()

    if augment_fn is not None:
        ds = ds.shuffle(buffer_size=4096, seed=SEED)
        ds = ds.map(
            lambda x, y: (augment_fn(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_datasets(train_dir: str, validation_dir: str):
    # ------------------------------------------------------------------ #
    #  Train / Val: both folders are pre-split manually, use as-is       #
    # ------------------------------------------------------------------ #
    train_paths, train_labels = _collect_image_paths(train_dir)
    print(f"[INFO] Train images : {len(train_paths)}")

    val_paths, val_labels = _collect_image_paths(validation_dir)
    print(f"[INFO] Val images   : {len(val_paths)}")

    # ------------------------------------------------------------------ #
    #  Augmentation pipeline  (upgraded — Prompt-2 integration)           #
    #                                                                      #
    #  Design rationale                                                    #
    #  ─────────────────────────────────────────────────────────────────  #
    #  RandomFlip / RandomRotation(0.5) / RandomTranslation / RandomZoom  #
    #    Geometric variance — unchanged from original.                     #
    #    RandomRotation(0.5) → 0.5 × 360° = ±180°  ✓                     #
    #                                                                      #
    #  RandomBrightness(0.30)                                              #
    #    ±30 % brightness shift simulates dim factory floors, outdoor      #
    #    inspection lighting, and torch-lit underside of machinery.        #
    #                                                                      #
    #  RandomContrast(0.35)  (raised from original 0.10)                  #
    #    Rust on steel ranges from washed-out (low contrast) to sharply   #
    #    textured (high contrast).                                         #
    #                                                                      #
    #  RandomHueSaturation(max_hue_delta=0.05, sat_range=(0.5, 1.5))      #
    #    Hue ±18° smears across the orange-brown-red rust band.           #
    #    Saturation 0.5–1.5 spans greyscale CCTV → vivid real rust.       #
    #                                                                      #
    #  AddGaussianNoise(min_stddev=0.01, max_stddev=0.05)                 #
    #    Camera sensor noise (ISO grain) and JPEG artefacts.              #
    #                                                                      #
    #  RandomGaussianBlur(kernel_size=5, sigma=(0.5,1.5), prob=0.5)       #
    #    50 % chance per image; simulates defocus, motion blur, CCTV.     #
    #    Applied AFTER noise so the two effects do not cancel out.         #
    # ------------------------------------------------------------------ #
    augment = keras.Sequential([

        # ── Geometric transforms ─────────────────────────────────────── #
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.5),               # up to ±180 degrees
        layers.RandomTranslation(0.06, 0.06),
        layers.RandomZoom(0.08),

        # ── Illumination: brightness & contrast ──────────────────────── #
        layers.RandomBrightness(factor=0.30,
                                value_range=(0.0, 1.0)),
        layers.RandomContrast(factor=0.35),

        # ── Sensor noise ─────────────────────────────────────────────── #
        AddGaussianNoise(min_stddev=0.01,
                         max_stddev=0.05,
                         name="gaussian_noise"),

        # ── Blur (50 % probability per image) ────────────────────────── #
        RandomGaussianBlur(kernel_size=5,
                           sigma_range=(0.5, 1.5),
                           prob=0.5,
                           name="gaussian_blur"),

    ], name="augmentation_realworld")

    train_ds = _paths_to_dataset(
        train_paths, train_labels,
        batch_size=BATCH_SIZE,
        augment_fn=augment
    )
    val_ds = _paths_to_dataset(val_paths, val_labels, batch_size=BATCH_SIZE)

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# 5.  Model  (VGG9 — 7 CONV + 2 Dense, identical layer names to fer2013)
#     Changes vs. fer2013:
#       - L2 regularization added to conv kernels (L2_CONV)
#       - dense2 output = 5  (NUM_CLASSES)
#       - DROPOUT_FLAT raised from 0.2 → 0.3
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class ClipConvWeights(constraints.Constraint):
    """Hard constraint: keep conv kernels within (-0.99, 0.99) so they stay
    inside the Q1.15 fixed-point range used by the MATLAB quantization
    step (int16(w * 2^15))."""
    def __call__(self, w):
        return tf.clip_by_value(w, -0.99, 0.99)

    def get_config(self):
        return {}

conv_clip = ClipConvWeights()


def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    conv_reg  = regularizers.l2(L2_CONV)
    dense_reg = regularizers.l2(L2_DENSE)
    inp = keras.Input(shape=input_shape, name="input")

    # --- Block 1 ---
    x = layers.Conv2D(32, (3,3), padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=conv_reg,
                      kernel_constraint=conv_clip, name="conv1")(inp)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(32, (3,3), padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=conv_reg,
                      kernel_constraint=conv_clip, name="conv2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2,2), strides=2, name="pool1")(x)

    # --- Block 2 ---
    x = layers.Conv2D(32, (3,3), padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=conv_reg,
                      kernel_constraint=conv_clip, name="conv3")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(32, (3,3), padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=conv_reg,
                      kernel_constraint=conv_clip, name="conv4")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2,2), strides=2, name="pool2")(x)

    # --- Block 3 ---
    x = layers.Conv2D(64, (3,3), padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=conv_reg,
                      kernel_constraint=conv_clip, name="conv5")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, (3,3), padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=conv_reg,
                      kernel_constraint=conv_clip, name="conv6")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2,2), strides=2, name="pool3")(x)

    # --- Block 4 ---
    x = layers.Conv2D(128, (3,3), padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=conv_reg,
                      kernel_constraint=conv_clip, name="conv7")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2,2), strides=2, name="pool4")(x)

    # --- Classifier head (no clip — scope is conv layers only) ---
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(DROPOUT_FLAT, name="dropout_flat")(x)

    x = layers.Dense(128, kernel_initializer="he_normal",
                     kernel_regularizer=dense_reg, name="dense1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(DROPOUT_DENSE, name="dropout_dense")(x)

    # 5 output nodes — one per defect class
    out = layers.Dense(num_classes, activation="softmax", dtype="float32",
                       kernel_initializer="he_normal", name="dense2")(x)

    return keras.Model(inputs=inp, outputs=out,
                       name="VGG9_IndustrialDefect_DE10_v6_biasonly_clipped")


# ---------------------------------------------------------------------------
# 6.  Callbacks
# ---------------------------------------------------------------------------

def get_callbacks(weights_file: str):
    return [
        callbacks.ModelCheckpoint(
            filepath=weights_file, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.6, patience=8,
            min_lr=1e-6, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20,
            restore_best_weights=True, verbose=1
        ),
    ]


# ---------------------------------------------------------------------------
# 7.  Training
# ---------------------------------------------------------------------------

def train():
    copy_to_local(DRIVE_TRAIN_DIR,      LOCAL_TRAIN_DIR,      "train")
    copy_to_local(DRIVE_VALIDATION_DIR, LOCAL_VALIDATION_DIR, "val")

    print("\n[INFO] Đang tính class weights...")
    class_weights = compute_class_weights(LOCAL_TRAIN_DIR)

    print("\n[INFO] Đang tạo tf.data pipeline từ SSD cục bộ...")
    train_ds, val_ds = build_datasets(LOCAL_TRAIN_DIR, LOCAL_VALIDATION_DIR)

    model = build_model()
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n[INFO] Bắt đầu huấn luyện...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=get_callbacks(WEIGHTS_FILE),
        verbose=1
    )

    print("\n[INFO] Đánh giá trên tập Validation...")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"[RESULT] Val loss: {val_loss:.4f}  |  Val accuracy: {val_acc*100:.2f}%")

    model.save(WEIGHTS_FILE)
    print(f"[INFO] Đã lưu → '{WEIGHTS_FILE}'")

    return model, history, val_ds


# ---------------------------------------------------------------------------
# 8.  Visualisation
# ---------------------------------------------------------------------------

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True)
    plt.tight_layout(); plt.show()


def plot_confusion_matrix(model, val_ds):
    y_pred_list, y_true_list = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred_list.append(np.argmax(preds,          axis=1))
        y_true_list.append(np.argmax(labels.numpy(), axis=1))
    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=DEFECT_LABELS, yticklabels=DEFECT_LABELS)
    plt.title("Confusion Matrix (normalised)")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout(); plt.show()
    print("\n[INFO] Classification report:")
    print(classification_report(y_true, y_pred, target_names=DEFECT_LABELS, digits=4))


# ---------------------------------------------------------------------------
# 9.  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from google.colab import drive
    drive.mount('/content/drive')

    model, history, val_ds = train()

    if history is not None:
        plot_training_history(history)
        plot_confusion_matrix(model, val_ds)
        print("\n[DONE] Hoàn tất.")