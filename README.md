# SC25-lsCOMP

This repository is for AD/AE process for SC'25 accepted paper "lsCOMP: Efficient Light Source Compression".

## 1. Introduction

This repository contains source code and executing scripts for SC'25 AD/AE process.
lsCOMP (<u>l</u>ight <u>s</u>ource <u>COMP</u>ression) is an efficient GPU lossless/lossy compressor designed for light source X-ray.
In short, this repository contains two sections.
- **2. Configuring lsCOMP and Datasets**: In this section, reviewer can check software/hardware dependencies, configure lsCOMP to executable binaries, and set up light source datasets.
- **3. Reproducing Paper Results**: In this section, reviewer can reproduce paper results step-by-step with a set of wrapped-up Python scripts. Since lsCOMP is majorly executed in GPU, whole scripts can be executed within around 10~20 minutes.

## 2. Configuring lsCOMP and Datasets

### 2.1 Software/Hardware Dependencies

Following software/hardware dependencies are neccessary to compile and execute lsCOMP.

- A Linux Machine (we use Ubuntu 20.04, but other OSs should also be good)
- Git 2.15 or newer
- CMake 3.21 or newer
- CUDA 11.0 or newer
- One NVIDIA A100 GPU (either 40 GB or 80 GB video memory works)
- Python 3 (this is for executing the wrapped up scripts)

### 2.2 Compiling lsCOMP into Executable Binaries

lsCOMP can be installed by following commands.

```shell
# First, git clone this repository.
$ git clone https://github.com/hyfshishen/cuLSZ.git

# Then, change directory to this repository.
$ cd cuLSZ

# Finally, compile lsCOMP via a Python script.
$ python3 0-compile-lsCOMP.py
```

After compilation, you will see two executable binaries ```lsCOMP_cssi``` and ```lsCOMP_xpcs``` in folder ```./lsCOMP/build/```.
To check whether installation is succesfull, you can execute ```./lsCOMP_cssi```. If you see results like below code block, then your installation is succesful and please feel free to go to the next step.
```shell
$ ./lsCOMP_cssi 
Usage:
   ./lsCOMP_cssi -i oriFilePath -d dims.x dims.y dims.z -b quantBins.x quantBins.y quantBins.z quantBins.w -p value -o decFilePath
Options:
   -i oriFilePath: Path to the original data file
   -d dims.x dims.y dims.z: Dimensions of the original data, where dim.z is the fastest dimension.
   -b quantBins.x quantBins.y quantBins.z quantBins.w: Quantization bins for the 4 levels, where x is the base one and x<=y<=z<=w.
   -p value: Pooling threshold for a data block.
   -o decFilePath: Path to the decompressed data file (optional).
Examples:
   ./lsCOMP_cssi -i data/cssi.bin -d 600 1813 1558 -b 3 5 10 15 -p 0.5
   ./lsCOMP_cssi -i data/cssi.bin -d 600 1813 1558 -b 3 5 10 15 -p 0.5 -o data/cssi-dec.bin
```

### 2.3 Setting Up Light Source Datasets

All light source datasets can be downloaded through a publically available [ANL BOX Link](https://anl.box.com/s/25q3jjm40ppdgf4173auc1fyuback2zl).
Please note that direct download via ```wget``` is not supported, as this feature is restricted in ANL Box's instance. 
We apologize for any inconvenience this may cause.
To proceed, please create a folder named ```datasets``` in your local repository and manually place all downloaded datasets inside it.
After this step, your local repository should have the following structure:
```shell
-- lsCOMP
    -- CMakeLists.txt
    -- build/
    -- example/
    -- include/
    -- src/
-- datasets
    -- 0-sfc.uint16         # only used in Section 6.1
    -- 1-spdi-m.uint16      # only used in Section 6.1
    -- 2-sfc-1.uint16       # only used in Section 6.1
    -- 2-sfc-2.uint16       # only used in Section 6.1
    -- 3-pcg.uint16         # only used in Section 6.1
    -- cssi-600.bin         # used in main evaluation 
    -- cssi-128.bin         # used in main evaluation
    -- xpcs-512-1.bin       # used in main evaluation
    -- xpcs-512-2.bin       # used in main evaluation
    -- xpcs-512-3.bin       # used in main evaluation
    -- xpcs-512-4.bin       # used in main evaluation
    -- xpcs-512-5.bin       # used in main evaluation
    -- xpcs-512-6.bin       # used in main evaluation
-- README.md
-- 0-compile-lsCOMP.py
-- # ... (other Python scripts)
```

## 3. Reproducing Paper Results

This section provides instructions on reproducing paper results.
Note that 3.1-3.7 are key results of this work (as main evaluation).
Other subsections are results for discussion sections, which are optional.
In each subsection, we provide a Python script to simpliy the execution.
Since lsCOMP is a GPU compressor targeting both throughputs and compression ratios, execution scripts for each Figure/Table only take several seconds.

### 3.1 Figure 7: Throughput of lsCOMP Lossless Compression

Following command can reproduce results for this step.

```shell
$ python3 1-reproducing-main-3-1.py 
lsCOMP for CSSI, lossless
Compression throughput: 531.146892 GB/s
Decompression throughput: 352.988355 GB/s

lsCOMP for XPCS, lossless
Compression throughput: 379.264894 GB/s
Decompression throughput: 264.238082 GB/s
```

The throughput results should match or close to what is reported in Figure 7.
Note that throughput may vary due to several factors; reviewers can re-run this script to obtain consistent results.

### 3.2 Figure 8: Throughput of lsCOMP Lossy Compression

Following command can reproduce results for this step.

```shell
$ python3 1-reproducing-main-3-2.py 
lsCOMP for CSSI, lossy, error=3
Compression throughput: 555.842907 GB/s
Decompression throughput: 711.187969 GB/s
lsCOMP for CSSI, lossy, error=5
Compression throughput: 496.135967 GB/s
Decompression throughput: 792.176081 GB/s
lsCOMP for CSSI, lossy, error=10
Compression throughput: 538.566104 GB/s
Decompression throughput: 1534.229353 GB/s

lsCOMP for XPCS, lossy, error=3
Compression throughput: 307.4942 GB/s
Decompression throughput: 525.718896 GB/s
lsCOMP for XPCS, lossy, error=5
Compression throughput: 445.995685 GB/s
Decompression throughput: 710.776869 GB/s
lsCOMP for XPCS, lossy, error=10
Compression throughput: 471.804976 GB/s
Decompression throughput: 950.508296 GB/s
```

The throughput results should match or close to what is reported in Figure 8.
Note that throughput may vary due to several factors; reviewers can re-run this script to obtain consistent results.


### 3.3 Figure 9: Throughput of Different lsCOMP Lossy Settings

Following command can reproduce results for this step.

```shell
$ python3 1-reproducing-main-3-3.py 
(a) CSSI-CMP curve:
[359.404003, 570.767084, 574.589896, 576.603661, 574.936516, 584.311336, 587.253021, 581.996939]
(a) CSSI-DEC curve:
[831.087969, 1034.268238, 1203.319866, 1358.940185, 1469.292066, 934.892754, 1724.436652, 1073.826046]
(a) XPCS-CMP curve:
[332.356338, 298.568845, 450.482332, 457.808089, 360.827112, 443.195884, 471.080443, 474.647045]
(a) XPCS-DEC curve:
[457.130717, 604.69012, 696.922094, 454.493961, 826.319033, 883.738253, 483.355376, 618.757639]
(b) CSSI-CMP curve:
[583.467615, 578.67797, 580.909241, 579.720732, 579.883829, 579.722417, 577.143935, 576.0923, 576.712263, 576.291131, 574.581714]
(b) CSSI-DEC curve:
[1493.932931, 1363.608671, 703.966976, 1463.396015, 965.289065, 679.321593, 812.266292, 1164.822685, 828.62697, 1211.953037, 675.632309]
(b) CSSI-CMP curve:
[438.874811, 370.57787, 348.386422, 441.941636, 420.551001, 447.103729, 348.262552, 405.054148, 434.479481, 446.647367, 311.718138]
(b) CSSI-DEC curve:
[828.901359, 745.379065, 317.933384, 804.412042, 729.322977, 355.975485, 708.318068, 440.90184, 460.793715, 697.542439, 701.30496]
```

The throughput results should match or close to what is reported in Figure 9.
Note that throughput may vary due to several factors; reviewers can re-run this script to obtain consistent results.

### 3.4 Table 2: Compression Ratios of lsCOMP Lossless Compression

Following command can reproduce results for this step.

```shell
$ python3 1-reproducing-main-3-4.py 
lsCOMP for CSSI, lossless
Compression ratios: 16.171549 -- 23.326942

lsCOMP for XPCS, lossless
Compression ratios: 6.999132 -- 7.009157
```

Compression ratios should be the same with what are reported in Table 2.

### 3.5 Table 3: Compression Ratios of lsCOMP Lossy Compression

Following command can reproduce results for this step.

```shell
$ python3 1-reproducing-main-3-5.py 
lsCOMP for CSSI, lossy, error=3
Compression ratios: 39.242572 -- 73.566316
lsCOMP for CSSI, lossy, error=5
Compression ratios: 55.070746 -- 103.093413
lsCOMP for CSSI, lossy, error=10
Compression ratios: 89.254915 -- 155.365298

lsCOMP for XPCS, lossy, error=3
Compression ratios: 18.036999 -- 18.079538
lsCOMP for XPCS, lossy, error=5
Compression ratios: 25.09974 -- 25.149117
lsCOMP for XPCS, lossy, error=10
Compression ratios: 37.226222 -- 37.274165
```

Compression ratios should be the same with what are reported in Table 3.

### 3.6 Figure 10: Compression Ratios of Different lsCOMP Lossy Settings

Following command can reproduce results for this step.

```shell
$ python3 1-reproducing-main-3-6.py 
(a) CSSI-CR curve:
[37.383003, 46.317276, 54.613671, 62.357348, 69.600159, 76.359907, 82.690847, 88.628849]
(a) XPCS-CR curve:
[17.214548, 21.350993, 24.801411, 27.913062, 30.652058, 32.982208, 34.976892, 36.744703]
(b) CSSI-CR curve:
[64.973693, 44.307414, 53.973115, 49.935922, 46.56608, 42.100633, 40.339365, 38.718434, 37.54292, 37.085601, 36.907487]
(b) XPCS-CR curve:
[22.472947, 18.322211, 20.573087, 19.54485, 18.838688, 17.66897, 17.067625, 16.453163, 16.012171, 15.791199, 15.752833]
```

Compression ratios should be the same with what are reported in Figure 10.

### 3.7 Figure 12: Data Transfer Time Simulation

### 3.8 (Optional) Figure 13: lsCOMP Throughput on Other Light Source Datasets

### 3.9 (Optional) Table 5: lsCOMP Compression Ratios on Other Light Source Datasets

### 3.10 (Optional) Figure 15: lsCOMP Throughput on Other Domain Datasets

### 3.11 (Optional) Table 7: lsCOMP Compression Ratios on Other Domain Datasets














<!-- ## Requirements
- CMake > 3.21
- CUDA > 11.0

## Compilation
```shell
mkdir build && cd build
cmake ..
make -j
```

The dimension of ```cssi.bin``` is (600, 1813, 1558). -->