# Firmware — ARM HPS C/C++ Code

## Overview

This folder contains the ARM HPS (Hard Processor System) firmware for the DE10-Standard FPGA board. The firmware loads quantized weights, reads input images, and dispatches inference through the custom CNN IP core, outputting classification results.

## Directory Structure
```
firmware/
├── main.cpp # Entry point
├── VideoMaterialDetector/ # Main detector module
│ ├── VideoMaterialDetector.cpp
│ └── VideoMaterialDetector.h
├── conv/ # Convolution layer handlers
│ ├── conv_layer.cpp
│ └── conv_layer.h
├── dense/ # Dense/FC layer handlers
│ ├── dense_layer.cpp
│ └── dense_layer.h
├── hwlib/ # Hardware register interface
│ ├── hw_registers.cpp
│ └── hw_registers.h
├── Makefile # Build configuration
└── README.md # This file
```

### Files & Responsibilities

| File/Folder | Purpose | Responsibility |
|---|---|---|
| `main.cpp` | Program entry point | Initialize HPS, load weights, loop frame processing |
| `VideoMaterialDetector/` | High-level detector API | Orchestrate input→inference→output pipeline |
| `conv/` | Convolution execution | Quantization checks, register writes, DMA setup |
| `dense/` | Dense/FC execution | Weight reads, accumulation, activation functions |
| `hwlib/` | Register interface | MMAP to CNN IP core, read/write registers |
| `Makefile` | Build system | Compile flags, linking, ARM cross-compile setup |

---

## Hardware Target

| Property | Value |
|---|---|
| **Board** | Terasic DE10-Standard |
| **SoC** | Altera Cyclone V (5CSXFC6D6F31C8) |
| **HPS Processor** | ARM Cortex-A9 (dual-core, 925 MHz) |
| **FPGA Fabric** | Cyclone V (not used directly; HPS communicates with CNN IP) |
| **CNN IP Core** | Custom Nimap=3 architecture (32 PU × 4 PE) |
| **Memory** | 1 GB SDRAM (shared with FPGA) |
| **I/O** | USB, Ethernet, GPIO, UART, etc. |

---

## Constraints & Design Decisions

### 1. Pixel Encoding

**Input pixels:** `uint8 - 128` (NOT `/255`)

**Rationale:** FPGA uses integer-only arithmetic; shifting by 128 centers grayscale values around 0 without division.

**Example:**
```cpp
// Input: uint8 grayscale (0–255)
uint8_t pixel_raw = 128;  // Mid-gray

// Conversion for FPGA
int8_t pixel_centered = pixel_raw - 128;  // → 0

pixel_raw = 255;  // Bright
pixel_centered = 255 - 128;  // → 127
```

### 2. CONV Weight Format (INT16, Q1.15)

**Layout:** 32 filters at a time (matching 32 PU hardware)

**Tensor arrangement:**
- Keras: `(kernel_h, kernel_w, input_channels, output_filters)`
- FPGA: Reorganized to `(output_filters_group_of_32, input_channels, kernel_h, kernel_w)`

**File reading:**
```cpp
// Read conv0_W.txt (3×3 kernel, 1 input, 32 output)
// Expected: 32 * 1 * 3 * 3 = 288 int16 values
FILE *f = fopen("conv0_W.txt", "r");
int16_t weights[32][1][3][3];
for (int i = 0; i < 32 * 1 * 3 * 3; i++) {
    fscanf(f, "%hd", &weights[...]);
}
```

### 3. Dense Weight Format (float32)

**Layout:** Single-column, one neuron per line

**File structure:**

dense1_W.txt (256×128 matrix, stored column-major)
0.123456 (neuron 0, input 0)
0.234567 (neuron 0, input 1)
...
-0.456789 (neuron 1, input 0)


**Reading:**
```cpp
FILE *f = fopen("dense1_W.txt", "r");
float weights[128][256];  // [output_neurons][input_neurons]
for (int i = 0; i < 128 * 256; i++) {
    fscanf(f, "%f", &weights[...]);
}
```

### 4. SDRAM Access Pattern

**CNN IP core interface:** Avalon Memory-Mapped (MM) master

**Data packing for image row:**
- 2 pixels per 32-bit word
- **Byte arrangement:** `(top_pixel << 16) | (bottom_pixel & 0xFFFF)`

**64×64 image storage:**
- Padded to 66×66 (zero-padding for boundary handling)
- Total words: (66 × 66) / 2 = 2,178 words
- Memory footprint: ~8.7 KB

**Code example:**
```cpp
uint32_t *img_buffer = (uint32_t *)img_base_addr;
uint16_t pixel_row[66];

// Pack two pixels per 32-bit word
for (int y = 0; y < 66; y++) {
    for (int x = 0; x < 66; x += 2) {
        uint16_t px0 = pixel_row[x];
        uint16_t px1 = pixel_row[x + 1];
        img_buffer[y * 33 + x/2] = (px0 << 16) | (px1 & 0xFFFF);
    }
}
```

### 5. Inference Mode

**Single-patch-per-frame design:**
- Input image resized immediately to 64×64
- Exactly one classification per frame
- No multi-ROI dispatch (previous architecture)

**Latency budget:** 15–34 ms/frame (81 MHz)

---

## Building & Compilation

### Prerequisites

- **Altera SoC EDS 16.1+** (or Intel Quartus Lite 17.0+)
- **ARM cross-compiler:** arm-linux-gnueabihf-gcc (usually bundled with SoC EDS)
- **Target OS:** Linux (Yocto-based or custom)

### Build Steps

#### Option 1: Using Provided Makefile

```bash
cd C:\dev\thesis\firmware

# Clean previous builds
make clean

# Compile
make

# Expected output:
# arm-linux-gnueabihf-g++ -c main.cpp -o build/main.o
# arm-linux-gnueabihf-g++ -c VideoMaterialDetector/... -o build/...
# ...
# Linking → firmware.elf
```

#### Option 2: Manual Compilation

```bash
# Compile each source file
arm-linux-gnueabihf-g++ -c -g -Wall \
  -I. \
  -I/path/to/altera/embedded_tools/include \
  main.cpp -o main.o

arm-linux-gnueabihf-g++ -c VideoMaterialDetector/VideoMaterialDetector.cpp -o VideoMaterialDetector.o
arm-linux-gnueabihf-g++ -c conv/conv_layer.cpp -o conv_layer.o
arm-linux-gnueabihf-g++ -c dense/dense_layer.cpp -o dense_layer.o
arm-linux-gnueabihf-g++ -c hwlib/hw_registers.cpp -o hw_registers.o

# Link
arm-linux-gnueabihf-g++ -o firmware.elf \
  main.o VideoMaterialDetector.o conv_layer.o dense_layer.o hw_registers.o \
  -L/path/to/altera/libs \
  -laltera_hal -lm
```

### Makefile Variables

**Edit `Makefile` to set your paths:**

```makefile
# ARM Cross-compiler path
ARM_COMPILER = arm-linux-gnueabihf-g++
ARM_CFLAGS = -mcpu=cortex-a9 -mfpu=neon -O2

# Altera SoC EDS paths
ALTERA_HWLIB = /opt/altera/embedded_tools/include
ALTERA_LIBS = /opt/altera/embedded_tools/lib

# Output
BUILD_DIR = build
OUTPUT = firmware.elf
```

---

## Deployment to DE10-Standard

### Step 1: Generate SD Card Image

1. Use **Altera SoC EDS** to create a bootable SD card
2. Include Linux kernel, device tree, and root filesystem
3. Copy `firmware.elf` to root filesystem: `/root/firmware.elf`

### Step 2: Transfer Weight Files

Copy quantized weights from `../weight_export/` to the board:

```bash
# On host machine (Windows/Linux)
scp conv0_W.txt root@<board_ip>:/root/weights/

# Or via USB:
# 1. Insert SD card
# 2. Mount on host
# 3. Copy weights to SD card weights folder
```

### Step 3: Configure Board

**On DE10-Standard (over SSH or serial terminal):**

```bash
# SSH
ssh root@<board_ip>

# Or serial (if using UART)
# Connect via USB-to-UART adapter, use minicom/putty at 115200 baud
```

### Step 4: Run Firmware

```bash
cd /root
./firmware.elf

# Expected console output:
# [INFO] CNN IP Core initialized
# [INFO] Loading weights from weights/conv0_W.txt
# [INFO] Loading weights from weights/conv0_b.txt
# ...
# [INFO] Inference ready. Press ENTER to classify image...
```

---

## Register Map (CNN IP Core)

The firmware communicates with the custom CNN IP core via memory-mapped registers. Common registers:

| Offset | Register | R/W | Purpose |
|---|---|---|---|
| 0x00 | CONTROL | W | Start, reset, mode select |
| 0x04 | STATUS | R | Done, error, busy flags |
| 0x08 | IMG_ADDR | W | Image buffer SDRAM address |
| 0x0C | WEIGHT_ADDR | W | Weight buffer SDRAM address |
| 0x10 | OUTPUT_ADDR | W | Output buffer SDRAM address |
| 0x14 | CONFIG | W | Layer select, filter count, etc. |

**Typical register write sequence:**

```cpp
// Write image address
write_register(IMG_ADDR, 0x10000000);

// Write weight address
write_register(WEIGHT_ADDR, 0x20000000);

// Select layer (e.g., conv0)
write_register(CONFIG, (0 << 16) | 32);  // layer=0, filters=32

// Start inference
write_register(CONTROL, 0x01);  // START bit

// Poll status
while (!(read_register(STATUS) & 0x01)) {  // Wait for DONE
    usleep(1000);
}
```

---

## Performance Metrics

### Latency Breakdown (15–34 ms total per frame)

| Stage | Time |
|-------|------|
| Image load & preprocess | 2–3 ms |
| Conv0–Conv6 (7 layers) | 10–25 ms |
| Dense1–Dense2 (2 layers) | 1–3 ms |
| Result readback | <1 ms |

### Resource Usage

| Resource | Usage | Capacity | Utilization |
|---|---|---|---|
| ALMs (adaptive logic modules) | 22,494 | 41,910 | 53.7% |
| DSP blocks | 112 | 112 | **100%** |
| RAM blocks | 553 | 553 | **100%** |
| M10K (10Kb) blocks | 553 | 553 | **100%** |

**Note:** DSP and RAM at 100% means little room for additional IP. CNN core is **compute-bound**.

### Max Frequency (Fmax)

TimeQuest analysis (Quartus Prime 17.0):
Clock domain: CLOCK_50 (50 MHz input)
Fmax at 85°C: 81.01 MHz
Fmax at 0°C: 83.63 MHz


**Operating point:** 81 MHz at 85°C (worst-case)

---

## Debugging

### Enable Serial Console

Connect USB-to-UART adapter to the board's UART header:

```bash
# On host (Linux/WSL)
minicom -D /dev/ttyUSB0 -b 115200

# On Windows
# Use PuTTY: Serial line `/dev/ttyUSB0`, Speed 115200
```

### Add Debug Logging

Edit `main.cpp` to enable verbose output:

```cpp
#define DEBUG 1

#if DEBUG
  printf("[DEBUG] Loading weights...\n");
  fflush(stdout);
#endif
```

### Probe CNN IP Registers

```cpp
uint32_t status = read_register(STATUS);
printf("Status: 0x%08X\n", status);
printf("  Done: %d\n", (status >> 0) & 1);
printf("  Busy: %d\n", (status >> 1) & 1);
printf("  Error: %d\n", (status >> 2) & 1);
```

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|---|---|---|
| "Cannot open weights file" | Path incorrect or SD card not mounted | Verify weights directory; use `ls -la weights/` |
| Segmentation fault | NULL pointer in `hw_registers.c` | Add bounds checking on MMAP address |
| Inference produces zeros | Weights not loaded or register init failed | Check weight file read loops; add printf debugging |
| Board reboots during inference | Excessive current draw (DSP blocks at 100%) | May be thermal throttling; ensure adequate cooling |
| Output doesn't match validation | Quantization error (expected) | Compare INT16 outputs to float32 reference; check by-layer |

---

## Next Steps

1. **Compile** using Makefile
2. **Deploy** `firmware.elf` and weight `.txt` files to DE10-Standard
3. **Test** with known images
4. **Validate** accuracy against reference (see `../docs/results.md`)
5. **Optimize** register sequences if latency budget exceeded

---

## References

- Terasic DE10-Standard User Manual: https://www.terasic.com.tw/
- Altera Cyclone V Device Documentation: https://www.intel.com/
- ARM Cortex-A9 Processor: https://www.arm.com/

---

**Last Updated:** September 2026
**Maintainers:** Blade Nguyễn Quốc Tín, Hà Xuân Cát
**Supervisor:** PGS.TS Trương Quang Vinh