# Weight Export & Quantization

## Overview

This folder contains MATLAB scripts to extract trained weights from the Keras `.h5` model, quantize them to fixed-point INT16 (Q1.15 format), and export to C-compatible text files for FPGA loading.

## Quantization Format

### Convolutional Layers (INT16, Q1.15)

- **Format:** Signed 16-bit integer, Q1.15 fixed-point
  - Range: [-1.0, ~0.99997]
  - Scaling factor: 2^15 = 32768
  - Example: weight `0.5` → `int16(0.5 * 32768) = 16384`

- **Tensor Layout:** 
  - Keras: `(height, width, input_channels, output_filters)`
  - FPGA: Reorganized to `(output_filters_group, input_channels, height, width)`
  - Output: 32 filters at a time (matches 32-PU hardware), arranged in **two-column** binary format

### Dense/FC Layers (float32)

- **Format:** Single-precision floating-point
- **Range:** Standard IEEE 754 float32
- **Output:** One value per line, column-major order

## Usage

### Prerequisites

- MATLAB R2020a+ (with HDF5 support)
- Keras/TensorFlow-generated `.h5` model file

### Step 1: Place Model File

Copy `model.weights.h5` (from `../training/`) into this folder:

weight_export/
├── model.weights.h5
├── export.m
├── write_weight_vgg_face_keras.m
├── write_txt_c.m
├── reshape_arr.m
└── README.md


### Step 2: Update Output Path (IMPORTANT)

**Open `write_txt_c.m` and modify lines 10–11:**

**BEFORE:**
```matlab
file_name_w_t = strcat('C:/Users/DEL/Downloads/DATN/industrial_defect_30_7_2023_new/',file_name_w,'.txt');
file_name_b_t = strcat('C:/Users/DEL/Downloads/DATN/industrial_defect_30_7_2023_new/',file_name_b,'.txt');
```

**AFTER** (example for Windows):
```matlab
output_dir = pwd; % Current directory
file_name_w_t = fullfile(output_dir, [file_name_w, '.txt']);
file_name_b_t = fullfile(output_dir, [file_name_b, '.txt']);
```

Or for a specific output folder:
```matlab
output_dir = 'C:\your_output_path\';
```

### Step 3: Run Export

```matlab
cd weight_export
export
```

Or simply:
```matlab
export
```

### Step 4: Verify Output

Check the output folder for generated `.txt` files:

conv0_W.txt conv0_b.txt
conv1_W.txt conv1_b.txt
...
dense1_W.txt dense1_b.txt
dense2_W.txt dense2_b.txt


## File Descriptions

| File | Purpose |
|------|---------|
| `export.m` | Main orchestrator; calls `write_weight_vgg_face_keras()` for conv & dense layers |
| `write_weight_vgg_face_keras.m` | Iterates over Keras layer structure; extracts `/layers/convXd/vars/0` (kernel), `/vars/1` (bias) |
| `write_txt_c.m` | Quantizes weights, formats into text, writes to file (core quantization logic) |
| `reshape_arr.m` | Tensor transpose utility: Keras NCHW → HWIO → custom FPGA ordering |

## Architecture

### VGG-9 Layers (as exported)

Layer 0: conv2d (3×3 kernel, 1 input channel, 32 output channels) → conv0_W.txt, conv0_b.txt
Layer 1: conv2d_1 (3×3 kernel, 32 input, 32 output) → conv1_W.txt, conv1_b.txt
Layer 2: conv2d_2 (3×3 kernel, 32 input, 64 output) → conv2_W.txt, conv2_b.txt
Layer 3: conv2d_3 (3×3 kernel, 64 input, 64 output) → conv3_W.txt, conv3_b.txt
Layer 4: conv2d_4 (3×3 kernel, 64 input, 128 output) → conv4_W.txt, conv4_b.txt
Layer 5: conv2d_5 (3×3 kernel, 128 input, 128 output) → conv5_W.txt, conv5_b.txt
Layer 6: conv2d_6 (3×3 kernel, 128 input, 256 output) → conv6_W.txt, conv6_b.txt

Dense 1: dense (256 × 128 neurons) → dense1_W.txt, dense1_b.txt
Dense 2: dense_1 (128 × 5 neurons, output classes) → dense2_W.txt, dense2_b.txt


## Known Issues & Fixes

1. **Path errors in `write_txt_c.m`:**
   - Ensure line 10–11 are updated to a valid, writable directory

2. **HDF5 Layer Structure:**
   - This script assumes Keras v3 layer naming: `/layers/convXd/vars/0` (kernel), `/vars/1` (bias)
   - If using older Keras/TensorFlow, paths may differ (check with `h5info('model.weights.h5')`)

3. **Quantization Loss:**
   - INT16 Q1.15 introduces ~0.003 quantization error per weight
   - On-board accuracy typically drops 10–15% vs. float32 validation accuracy
   - This is acceptable for edge inference

## Next Step

Once `.txt` weights are generated, copy them to `../firmware/` and follow the firmware README to compile and deploy to the DE10-Standard.

---

**Last Updated:** September 2026