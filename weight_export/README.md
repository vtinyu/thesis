# Weight Export & Quantization

## Overview

This folder contains MATLAB scripts to extract trained weights from the Keras `.h5` model file, quantize them to fixed-point INT16 (Q1.15 format), and export to C-compatible text files for FPGA hardware loading.

## Directory Structure
```
weight_export/
├── export.m # Main export orchestrator
├── write_weight_vgg_face_keras.m # Layer-by-layer weight extraction
├── write_txt_c.m # Binary formatting to C-readable text
├── reshape_arr.m # Tensor reshaping utility
└── README.md # This file
```

### Files

| File | Purpose | Function |
|------|---------|----------|
| `export.m` | Main orchestrator script | Calls weight extraction for all conv & dense layers |
| `write_weight_vgg_face_keras.m` | Layer iterator | Extracts Keras HDF5 layer paths; calls quantization |
| `write_txt_c.m` | Core quantization & formatting | INT16 Q1.15 conversion, tensor reshape, file output |
| `reshape_arr.m` | Tensor utility | Converts Keras NHWC → custom FPGA layout |
| `README.md` | Documentation | This file |

---

## Requirements

- MATLAB R2020a or later (with HDF5 support)
- Trained model file: `model.weights.h5` (from `../training/`)
- Output directory with **write permissions**

---

## Quantization Concepts

### Fixed-Point Representation

**INT16 Q1.15 Format** (for convolutional layers):
- **Bits:** 16-bit signed integer
- **Range:** [-1.0, +0.9999695] (approximately)
- **Scale factor:** 2^15 = 32,768
- **Precision:** ~0.00003 (1/32768)

**Conversion formula:**

quantized_value = round(float_value × 2^15)
dequantized_value = quantized_value / 2^15


**Example:**

float_weight: 0.5
quantized: int16(0.5 × 32768) = 16384
dequantized: 16384 / 32768 = 0.5 (exact)


**Float32 Format** (for dense layers):
- Standard IEEE 754 single-precision floating-point
- No quantization needed
- Direct export to text

---

## Workflow

### Step 1: Prepare Model File

Copy `model.weights.h5` from `../training/` to this directory:

```bash
# Example (Windows Command Prompt)
cd C:\dev\thesis\weight_export
copy ..\training\model.weights.h5 .
```

Verify file exists:
```bash
dir model.weights.h5
```

### Step 2: Update Output Path in `write_txt_c.m`

**IMPORTANT:** The script has a hardcoded output path. You MUST update it.

**Open `write_txt_c.m` and find lines 10–11:**

**BEFORE (hardcoded — DON'T USE):**
```matlab
file_name_w_t = strcat('C:/Users/DEL/Downloads/DATN/industrial_defect_30_7_2023_new/',file_name_w,'.txt');
file_name_b_t = strcat('C:/Users/DEL/Downloads/DATN/industrial_defect_30_7_2023_new/',file_name_b,'.txt');
```

**AFTER (dynamic — use this):**

**Option A: Current directory (recommended)**
```matlab
output_dir = pwd;  % Current working directory
file_name_w_t = fullfile(output_dir, [file_name_w, '.txt']);
file_name_b_t = fullfile(output_dir, [file_name_b, '.txt']);
```

**Option B: Custom output folder**
```matlab
output_dir = 'C:\dev\thesis\weight_export\output';  % Your path
% Create directory if it doesn't exist
if ~isfolder(output_dir)
    mkdir(output_dir);
end
file_name_w_t = fullfile(output_dir, [file_name_w, '.txt']);
file_name_b_t = fullfile(output_dir, [file_name_b, '.txt']);
```

**Save the file (Ctrl+S).**

### Step 3: Run Export in MATLAB

```matlab
cd C:\dev\thesis\weight_export

% Run main export script
export
```

**Expected console output:**

==> THÀNH CÔNG! Đã xuất toàn bộ file .txt!
(Success! Exported all .txt files!)


### Step 4: Verify Output

Check the output directory for generated `.txt` files:

conv0_W.txt (conv layer 0 weights)
conv0_b.txt (conv layer 0 bias)
conv1_W.txt
conv1_b.txt
...
conv6_W.txt
conv6_b.txt
dense1_W.txt (dense layer 1 weights)
dense1_b.txt
dense2_W.txt (dense layer 2 weights)
dense2_b.txt


**Total:** 14 files (7 conv + 2 dense × 2 for weights & bias)

---

## HDF5 Model Structure

### Expected Keras Model Hierarchy

The script assumes the following HDF5 structure (Keras v3 native format):

```
model.weights.h5
├── layers/
│ ├── conv2d/
│ │ └── vars/
│ │ ├── 0 (kernel weights, shape: H×W×C_in×C_out)
│ │ └── 1 (bias, shape: C_out)
│ ├── conv2d_1/
│ │ └── vars/
│ │ ├── 0
│ │ └── 1
│ ├── ... (conv2d_2 through conv2d_6)
│ ├── dense/
│ │ └── vars/
│ │ ├── 0 (kernel weights)
│ │ └── 1 (bias)
│ └── dense_1/
│ └── vars/
│ ├── 0
│ └── 1
```

### Verify Model Structure

To check the actual HDF5 structure:

```matlab
% List all datasets in the HDF5 file
h5info('model.weights.h5')

% Or focus on layer paths
h5disp('model.weights.h5')
```

If your model uses a different layer naming (e.g., `model/layers/...` or older TensorFlow format), update `write_weight_vgg_face_keras.m` accordingly.

---

## VGG-9 Layer Mapping

### Convolutional Layers (INT16 Q1.15)

| Layer | Input Channels | Output Filters | Kernel Size | Input Shape | Output Shape |
|-------|---|---|---|---|---|
| conv0 (conv2d) | 1 | 32 | 3×3 | 64×64 | 64×64 |
| conv1 (conv2d_1) | 32 | 32 | 3×3 | 64×64 | 64×64 |
| conv2 (conv2d_2) | 32 | 64 | 3×3 | 32×32 | 32×32 |
| conv3 (conv2d_3) | 64 | 64 | 3×3 | 32×32 | 32×32 |
| conv4 (conv2d_4) | 64 | 128 | 3×3 | 16×16 | 16×16 |
| conv5 (conv2d_5) | 128 | 128 | 3×3 | 16×16 | 16×16 |
| conv6 (conv2d_6) | 128 | 256 | 3×3 | 8×8 | 8×8 |

**Conv Weight Tensor Sizes:**

Keras shape: (kernel_h, kernel_w, input_channels, output_filters)
conv0_W shape: (3, 3, 1, 32) → after reshape: (32, 1, 3, 3)
conv6_W shape: (3, 3, 128, 256) → after reshape: (256, 128, 3, 3)


### Dense Layers (float32)

| Layer | Input Units | Output Units |
|-------|---|---|
| dense (dense) | 256 | 128 |
| dense_1 (dense_1) | 128 | 5 |

**Dense Weight Sizes:**

dense_W shape: (256, 128)
dense_b shape: (128,)
dense_1_W shape: (128, 5)
dense_1_b shape: (5,)


---

## File Format Details

### Convolutional Weight File Format

**Output:** `convX_W.txt` (one weight value per line, grouped by 16-filter batches)

**Example `conv0_W.txt` structure:**

16384 16383 16500 16234 ... (row 1)
14567 15234 ... (row 2)
...


**Layout logic:**
- Process filters in batches of 32 (hardware constraint: 32 Processing Units)
- For each batch, iterate over input channels and kernel positions (3×3)
- Each weight is converted to INT16 and output as unsigned integer
- Two values per line (16-bit packing compatibility)

**Quantization function** (in `write_txt_c.m`):
```matlab
kernel_1_0 = int16(reshape_arr(h5read(in,weight)) * 2^15);
```

### Bias File Format

**Output:** `convX_b.txt` (one bias value per line, INT16 Q1.15)

**Example:**

1234
5678
...


### Dense Weight File Format

**Output:** `denseX_W.txt` (one float32 value per line)

**Example `dense1_W.txt`:**

0.123456
-0.234567
0.345678
...


No quantization; direct float32 export.

---

## Known Issues & Troubleshooting

### Issue 1: "Index exceeded array bounds"

**Cause:** HDF5 path mismatch (layer names don't match Keras v3 structure)

**Solution:**
```matlab
% Debug: print available layers
h5info('model.weights.h5')

% Update layer paths in write_weight_vgg_face_keras.m if needed
```

### Issue 2: "Cannot open file for writing"

**Cause:** Output directory doesn't exist or no write permission

**Solution:**
```matlab
% Ensure directory exists
output_dir = pwd;
if ~isfolder(output_dir)
    mkdir(output_dir);
end
```

### Issue 3: Quantization causes NaN or Inf

**Cause:** Weight magnitudes exceed Q1.15 range (> ±1.0)

**Solution:**
```matlab
% Normalize weights to [-1, +1] range before quantizing
weights_scaled = weights / max(abs(weights(:)));
quantized = int16(weights_scaled * 2^15);
```

---

## Quantization Error Analysis

**Expected error** from INT16 Q1.15 quantization:

| Scenario | Quantization Error | Accuracy Impact |
|----------|---|---|
| Weight = 0.5 | 0 (exact representation) | 0% |
| Weight = 0.123 | ~0.00003 (1 LSB) | <0.01% |
| Weight near ±1.0 | ~0.00006 | <0.1% |
| Accumulated (1000s of ops) | Cumulative rounding | 5–15% validation→on-board drop |

**Empirical results** from this thesis:
- Float32 validation accuracy: 99.07%
- INT16 on-board accuracy: ~88% (11% drop due to quantization + FPGA routing)

---

## Next Steps

Once `.txt` weight files are generated:

1. **Copy** all `.txt` files to `../firmware/` (or point firmware Makefile to this directory)
2. Verify firmware can read `.txt` at runtime
3. Follow `../firmware/README.md` to compile and deploy to DE10-Standard

---

## References

- Q1.15 Fixed-Point Format: ARM NEON documentation
- HDF5 in MATLAB: https://www.mathworks.com/help/matlab/hdf5-files.html
- Keras Model Serialization: https://keras.io/api/models/

---

**Last Updated:** September 2026
**Author:** Blade Nguyễn Quốc Tín & Hà Xuân Cát