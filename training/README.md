# Training Pipeline

## Overview

This folder contains the Keras/TensorFlow training script for the VGG-9 surface defect detection model.

## Requirements

- Python 3.8+
- TensorFlow 2.11+
- MATLAB (for weight export, in `../weight_export/`)

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Format

Prepare your industrial defect dataset with the following structure:

dataset/
├── class_0/ (e.g., Rust)
│ ├── img_001.jpg
│ ├── img_002.jpg
│ └── ...
├── class_1/ (e.g., Scratch)
│ └── ...
├── class_2/ (e.g., Paint Defect)
│ └── ...
├── class_3/ (e.g., Dent)
│ └── ...
└── class_4/ (e.g., Normal/Good)
└── ...


## Training

```bash
python industrial.py
```

### Key Parameters (in `industrial.py`)

- `num_classes = 5` — Number of defect classes
- `input_shape = (64, 64, 1)` — Grayscale 64×64 input (matches FPGA constraints)
- `batch_size = 32`
- `epochs = 200` (adjust based on convergence)
- Augmentation: 9 layers including rotation, crop, HSV shift, Gaussian noise

### Output

The script generates:
- **`model.weights.h5`** — Trained model weights in HDF5 format
- **Plots** — Training history and confusion matrix

### Expected Results

- **Validation Accuracy:** ~99% (on Colab / high-precision float32)
- **On-Board Accuracy:** ~88% (after INT16 quantization and FPGA deployment)

## Next Step

Once training completes and `model.weights.h5` is generated:
1. Move `model.weights.h5` to `../weight_export/`
2. Follow instructions in `../weight_export/README.md` to export quantized weights

## Troubleshooting

**Mode Collapse (FER2013 variant):** If the model predicts only one class, check:
- Augmentation intensity (reduce `RandomRotation`, `RandomZoom` ranges)
- Class balance in training set
- Learning rate (try `0.0005` instead of `0.001`)

**Gradient Noise:** Avoid `mixed_float16` policy on small validation sets; use `float32` instead.

**Validation Accuracy Oscillation:** Always use `sklearn.train_test_split(..., stratify=labels)` to ensure balanced class distribution in validation.