# Training Pipeline

## Overview

This folder contains the Keras/TensorFlow training script for the VGG-9 surface defect detection model. The pipeline handles data loading, augmentation, model building, training, and validation on industrial surface defect datasets.

## Directory Structure
```
training/
├── industrial.py # Main Keras training script 
├── requirements.txt # Python dependencies
└── README.md # This file
```

### Files

| File | Purpose | Lines |
|------|---------|-------|
| `industrial.py` | Main training script with custom augmentation layers, VGG-9 model definition, training loop | 580 |
| `requirements.txt` | Python package dependencies | — |
| `README.md` | Documentation | — |

---

## Requirements

- Python 3.8+
- TensorFlow 2.11+
- See `requirements.txt` for full dependency list

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Format

Prepare your industrial defect dataset with the following structure:
```
dataset/
├── class_0/ (e.g., Rust)
│ ├── img_001.jpg
│ ├── img_002.jpg
│ └── ...
├── class_1/ (e.g., Scratch)
│ ├── img_011.jpg
│ ├── img_012.jpg
│ └── ...
├── class_2/ (e.g., Hole)
│ └── ...
├── class_3/ (e.g., Crack)
│ └── ...
└── class_4/ (e.g., Normal)
└── ...
```

**Class Mapping (5 classes):**
- `class_0` → Rust
- `class_1` → Scratch
- `class_2` → Hole
- `class_3` → Crack
- `class_4` → Normal (Good surface)

---

## Training

### Quick Start

```bash
python industrial.py
```

The script will:
1. Load data from `dataset/` (assumes it exists in the same directory)
2. Apply stratified train/validation split (80/20)
3. Build VGG-9 model
4. Train for 175 epochs
5. Save trained weights to `model.weights.h5`
6. Generate training history plots and confusion matrix

### Key Parameters (in `industrial.py`)

**Model Architecture:**
```python
num_classes = 5                       # Defect classes
input_shape = (64, 64, 1)            # Grayscale 64×64 (FPGA constraint)
vgg9_layers = [32, 32, 64, 64, 128, 128, 256]  # Filter per conv layer
```

**Training Hyperparameters:**
```python
batch_size = 32
epochs = 200
learning_rate = 0.001
optimizer = 'adam'
loss = 'categorical_crossentropy'
```

**Data Augmentation (9 layers):**
- Random rotation (±20°)
- Random crop (90% of image)
- Random zoom (±10%)
- Horizontal flip
- HSV color space shifts (hue ±10°, saturation ±10%, value ±10%)
- Gaussian noise (stddev 0.01–0.05)
- Brightness adjustment (±10%)
- Contrast adjustment (±10%)

**Validation:**
```python
validation_split = 0.2
stratified = True  # sklearn.train_test_split(..., stratify=labels)
```

---

## Output

After successful training:

| Output | Location | Description |
|--------|----------|-------------|
| Model weights | `model.weights.h5` | Trained VGG-9 in Keras native format |
| Training history | `training_history.png` | Loss/accuracy curves (train vs validation) |
| Confusion matrix | `confusion_matrix.png` | Per-class accuracy visualization |
| Logs | Console output | Epoch-by-epoch metrics |

### Expected Results

- **Validation Accuracy (float32):** ~99%
- **On-Board Accuracy (INT16 quantized):** ~88% (after FPGA deployment)
- **Training Time:** ~15–30 min (on Colab GPU) or ~60–120 min (CPU)

---

## Architecture

### VGG-9 Model
```
Input: (64, 64, 1)
├── Conv2D(32, 3×3) + ReLU + MaxPool(2×2) → (32, 32)
├── Conv2D(32, 3×3) + ReLU + MaxPool(2×2) → (16, 16)
├── Conv2D(64, 3×3) + ReLU + MaxPool(2×2) → (8, 8)
├── Conv2D(64, 3×3) + ReLU + MaxPool(2×2) → (4, 4)
├── Conv2D(128, 3×3) + ReLU + MaxPool(2×2) → (2, 2)
├── Conv2D(128, 3×3) + ReLU + MaxPool(2×2) → (1, 1)
├── Conv2D(256, 3×3) + ReLU → (1, 1)
├── Flatten → (256,)
├── Dense(128) + ReLU + Dropout(0.5) → (128,)
└── Dense(5) + Softmax → (5,) [class logits]
```
Total parameters: ~1.3M
Trainable parameters: ~1.3M


### Custom Layers

The script defines a custom **`AddGaussianNoise`** layer for sensor-realistic noise injection during training. This simulates camera ISO grain and JPEG compression artifacts without destroying discriminative texture (critical for rust vs. scratch distinction).

---

## Known Issues & Fixes

### Mode Collapse (FER2013 variant)

**Symptom:** Model predicts only one class (usually neutral).

**Causes & Solutions:**
- ✅ Reduce augmentation intensity (rotation ±10° instead of ±20°)
- ✅ Ensure balanced class distribution in training set
- ✅ Lower learning rate (try 0.0005 instead of 0.001)
- ✅ Remove aggressive augmentation (e.g., ColorJitter with ±50% values)

### Gradient Noise on Small Validation Sets

**Symptom:** Validation loss oscillates wildly; accuracy jerks between 50–99%.

**Cause:** Mixed precision (`mixed_float16`) amplifies rounding errors.

**Solution:** Use `float32` (single precision) throughout—this script already does this.

```python
# NOT recommended:
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# YES, use float32:
tf.keras.mixed_precision.set_global_policy('float32')
```

### Validation Accuracy Oscillation

**Symptom:** Accuracy jumps 10–20% between epochs.

**Cause:** Imbalanced class distribution in validation fold.

**Solution:** Always use **stratified split** with `sklearn.train_test_split`:

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,  # ← IMPORTANT
    random_state=42
)
```

---

## Next Steps

Once training completes and `model.weights.h5` is generated:

1. **Move** `model.weights.h5` to `../weight_export/`
2. Follow instructions in `../weight_export/README.md` to export quantized weights to `.txt` format
3. Copy exported weights to `../firmware/` for on-board deployment

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: dataset/` | Place dataset folder in same directory as `industrial.py` |
| `Out of memory` | Reduce `batch_size` (try 16) or use smaller input images |
| `NaN loss after epoch 5` | Lower `learning_rate` (try 0.0001) or add gradient clipping |
| Model stuck at 20% accuracy | Check class imbalance; ensure stratified split is used |
| Training very slow (CPU) | Use Google Colab with GPU (free tier available) |

---

## References

- VGG: Simonyan, K., & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- Keras Documentation: https://keras.io/
- TensorFlow: https://www.tensorflow.org/

---

**Last Updated:** September 2026
