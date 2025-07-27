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



### 3.2 Figure 8: Throughput of lsCOMP Lossy Compression


### 3.3 Figure 9: Throughput of Different lsCOMP Lossy Settings

### 3.4 Table 2: Compression Ratios of lsCOMP Lossless Compression

### 3.5 Table 3: Compression Ratios of lsCOMP Lossy Compression

### 3.6 Figure 10: Compression Ratios of Different lsCOMP Lossy Settings

### 3.7 Figure 12: Data Transfer Time Simulation

### 3.8 Figure 13: lsCOMP Throughput on Other Light Source Datasets

### 3.9 Table 5: lsCOMP Compression Ratios on Other Light Source Datasets

### 3.10 Figure 15: lsCOMP Throughput on Other Domain Datasets

### 3.11 Table 7: lsCOMP Compression Ratios on Other Domain Datasets














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