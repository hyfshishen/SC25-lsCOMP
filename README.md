# cuLSZ: Fast Lossy GPU Compression for Light Source

## Requirements
- CMake > 3.21
- CUDA > 11.0

## Compilation
```shell
mkdir build && cd build
cmake ..
make -j
```

This branch consists of baseline methods of cuLSZ.

1. **Conversion** indicates converting uint32/uint16 data into floating point data and use cuSZp2 to compress.
2. **Truncation** indicates truncating (more specifically thresholding) part data into 0 then do cuSZp2 lossless compression.