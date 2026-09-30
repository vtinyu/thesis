# Industrial Surface Defect Detection on DE10-Standard FPGA

A complete pipeline for training, quantizing, and deploying a VGG-9 convolutional neural network onto a Terasic DE10-Standard FPGA for real-time industrial material surface defect detection.

## Project Overview

This repository implements an **embedded deep learning system** across three interconnected layers:

1. **Training** (`training/`) — Keras/TensorFlow VGG-9 model training on industrial defect dataset
2. **Weight Export** (`weight_export/`) — MATLAB scripts to quantize weights to INT16 (Q1.15) and export to C-compatible text files
3. **Firmware** (`firmware/`) — ARM HPS C/C++ firmware that reads quantized weights and performs inference via the custom CNN IP core on Cyclone V SoC

## Hardware & Target

- **FPGA Board:** Terasic DE10-Standard (Altera Cyclone V SoC)
- **CNN IP Core:** 32 Processing Units × 4 Processing Elements (INT16, based on Pham The Vinh's architecture)
- **Input:** 64×64 grayscale images
- **Output:** 5-class defect classification (5 industrial defect types)
- **Resource Usage:** 22,494 ALMs (54%), 112 DSP (100%), 553 RAM blocks (100%)
- **Inference Latency:** 15–34 ms/frame @ 81.01 MHz (Fmax at 85°C)

## Directory Structure

## Directory Structure

```
.
├── training/                   # Python training pipeline
│   ├── industrial.py          # Main Keras training script (580 lines)
│   ├── requirements.txt        # Python dependencies
│   └── README.md              # Training instructions
│
├── weight_export/             # MATLAB quantization & export scripts
│   ├── export.m               # Main export orchestrator
│   ├── write_weight_vgg_face_keras.m  # Layer-by-layer weight extraction
│   ├── write_txt_c.m          # Binary formatting to C-readable text
│   ├── reshape_arr.m          # Tensor reshaping utility
│   └── README.md              # Quantization format documentation
│
├── firmware/
|   ├── .gitignore                  # ARM HPS C/C++ code
│   ├── main.cpp               # Entry point
│   ├── VideoMaterialDetector/  # Main detector module
│   ├── conv/                  # Convolution layer handlers
│   ├── dense/                 # Dense/FC layer handlers
│   ├── hwlib/                 # Hardware register interface
│   ├── Makefile               # Build configuration
│   └── README.md              # Build & deployment guide
│
├── docs/                      # Documentation
│
├── .gitignore                 # Git ignore rules
└── README.md                  # Main project readme
```


## Quick Start

### Training a New Model

```bash
cd training
pip install -r requirements.txt
python industrial.py
```

Expected output: `model.weights.h5` (trained VGG-9 model file)

### Exporting Weights for FPGA

1. Ensure `model.weights.h5` is in the `weight_export/` directory
2. Open MATLAB and navigate to `weight_export/`
3. Run:
```matlab
   export
```
   This will generate `.txt` weight files in the output directory

### Building & Deploying Firmware

See `firmware/README.md` for Quartus project setup, HPS compilation, and on-board deployment.

## Key Features

- **Data Augmentation:** 9-layer augmentation pipeline (random crop, rotation, HSV shifts, Gaussian noise) for domain robustness
- **Mixed-Precision Training:** Disabled due to gradient noise on small validation sets; switched to single-precision float32
- **Stratified Splitting:** Uses `sklearn.train_test_split(..., stratify=labels)` to prevent validation accuracy oscillation
- **Fixed-Point Quantization:** INT16 Q1.15 for convolutional layers, float32 for dense layers (HPS memory constraint)
- **Hardware-Software Co-Design:** Dataflow synchronized with Nimap=3 CNN IP core streaming architecture; 16-bit word packing for SDRAM access

## Hardware Constraints & Design Decisions

- **Pixel Encoding:** `uint8 - 128` (not `/255`) for FPGA integer-only processing
- **CONV Weight Format:** INT16, arranged in two-column layout per filter group (32 filters at a time)
- **Dense Weight Format:** float32, single-column per neuron
- **Input Padding:** SDRAM traversal with 66×66 zero-padded buffer; 16-bit word packing: `(top << 16) | (bottom & 0xFFFF)`
- **Inference Mode:** Single 64×64 frame resized immediately on load; exactly one classification per frame (no multi-ROI dispatch)

## Performance

| Metric | Value |
|--------|-------|
| Colab Validation Accuracy | 99.07% |
| On-Board Accuracy (HPS+FPGA) | ~88% (resource-constrained int16) |
| Inference Latency | 15–34 ms/frame |
| Max Frequency (Fmax) | 81.01 MHz @ 85°C |
| Resource Usage | 22,494 ALMs (54%), 112 DSP (100%), 553 RAM blocks (100%) |

## Thesis & Acknowledgments

This project is part of the graduation thesis:
- **Title:** "Industrial Material Surface Defect Detection using the Terasic DE10-Standard FPGA Kit"
- **Team:** Blade Nguyễn Quốc Tín, Hà Xuân Cát
- **Supervisor:** PGS.TS Trương Quang Vinh
- **Institution:** HCMUT (Ho Chi Minh City University of Technology), Electronics–Telecommunications Engineering
- **Submission Date:** September 2026

**Base CNN IP Architecture Credit:** Pham The Vinh's research on optimized CNN hardware deployment on Cyclone V.

## License

[Choose a license, e.g., MIT, Apache 2.0, or CC-BY-4.0 for academic work]

## Contact & Support

For questions or contributions, open an issue in this repository or contact the thesis authors.

---

**Last Updated:** September 2026