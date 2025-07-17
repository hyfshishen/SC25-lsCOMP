# lsCOMP: Efficient Light Source Compression

This branch is for uint16 visualization data compression.
Original design is accepted at [SC'25] "lsCOMP: Efficient Light Source Compression".

## Requirements
- CMake > 3.21
- CUDA > 11.0

## Compilation
```shell
mkdir build && cd build
cmake ..
make -j
```
The compiled executable binary can be found as ```lsCOMP_uint16``` and ```lsCOMP_uint32``` in the ```build``` folder.


## To Use lsCOMP
lsCOMP supports both ```uint16``` and ```uint32``` compression.
The whole compression process is within a single GPU kernel, guaranteeing end-to-end throughput.
lsCOMP supports both lossless and lossy compression for ```uint16``` and ```uint32``` compression.

Takes ```uint16``` compression as an example. The API explanation is shown below:
```shell
yafan.huang@gpu3:~$ ./lsCOMP_uint16 
Usage:
   ./lsCOMP_uint16 -i oriFilePath -d dims.x dims.y dims.z -b quantBins.x quantBins.y quantBins.z quantBins.w -p value -o decFilePath
Options:
   -i oriFilePath: Path to the original data file
   -d dims.x dims.y dims.z: Dimensions of the original data, where dim.z is the fastest dimension.
   -b quantBins.x quantBins.y quantBins.z quantBins.w: Quantization bins for the 4 levels, where x is the base one and x<=y<=z<=w.
   -p value: Pooling threshold for a data block.
   -o decFilePath: Path to the decompressed data file (optional).
Examples:
   ./lsCOMP_uint16 -i data/xpcs.bin -d 1024 1813 1558 -b 3 5 10 15 -p 0.5
   ./lsCOMP_uint16 -i data/xpcs.bin -d 1024 1813 1558 -b 3 5 10 15 -p 0.5 -o data/xpcs-dec.bin
```

Assuming your dataset is TEST_DATA with dimension (2048, 1024, 512), where 512 is the fastest dimension.
The lossless compression can be executed as below:
```shell
./lsCOMP_uint16 -i TEST_DATA -d 2048 1024 512 -b 1 1 1 1 -p 1 -o REC_TEST_DATA
# The REC_TEST_DATA is TEST_DATA after compression and decompression. Since we are working with lossless compression. They should be identical.
# "-o XXXXX" is optional.
```

There are two modes of lossy. If you want to stick with error-bounded lossy compression (for example, error bound is 3). The executed command is:
```shell
./lsCOMP_uint16 -i TEST_DATA -d 2048 1024 512 -b 3 3 3 3 -p 1 -o REC_TEST_DATA
# The 4 quantization bins are all 3, so the error bound is 3.
```

Other lossy modes are for light source dataset. You can ignore it in your case.
