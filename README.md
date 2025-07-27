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
- CUDA 11.2 or newer
- One NVIDIA A100 GPU (either 40 GB or 80 GB video memory works)
- Python 3 (this is for executing the wrapped up scripts)

### 2.2 Compiling lsCOMP into Executable Binaries



### 2.3 Setting Up Light Source Datasets

## 3. Reproducing Paper Results

### Figure


## Requirements
- CMake > 3.21
- CUDA > 11.0

## Compilation
```shell
mkdir build && cd build
cmake ..
make -j
```

The dimension of ```cssi.bin``` is (600, 1813, 1558).