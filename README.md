# lsCOMP

This branch contains instructions to setup Podman or Docker container for running lsCOMP compressor.
The original image is created using Docker environments and is compatible with Podman as well.
The prepared image is uploaded in Dockerhub with [link](https://hub.docker.com/repository/docker/hyfshishen/lscomp/general).
Inside this image, lsCOMP compressor and NVIDIA nvCOMP compressors (along with other related software) are configured.

## Prerequisite
- A Linux machine
- An NVIDIA GPU
- Docker or Podman installed

## Launching Container

If you are using Podman 4.0+ and NVIDIA driviers are setup correctly:

```shell
# Pull the image
podman pull docker.io/hyfshishen/lscomp:cuda12-container

# Launch the container with GPU access (NVIDIA runtime must be configured)
podman run --rm -it --hooks-dir=/usr/share/containers/oci/hooks.d \
    --device nvidia.com/gpu=all \
    hyfshishen/lscomp:cuda12-container bash
```

If you are using Docker, please make sure Docker and NVIDIA Container Toolkit are installed and working.

```shell
# Pull the image
docker pull hyfshishen/lscomp:cuda12-container

# Launch an interactive container with GPU access
docker run --rm -it --gpus all hyfshishen/lscomp:cuda12-container bash
```

The executable binaries for lsCOMP and nvCOMP compressors are setup in local path already inside this container.
So in later steps, we assume we are already in ```~/``` path inside this container.

## Executing lsCOMP and nvCOMP Compressors

### Setting Up Datasets

To make sure light source datasets can be executed with lsCOMP and nvCOMP command line interfaces, we need to extract it from the original HDF5 files and save it as binary format.

Taking xx as an example.


### Using lsCOMP

There are two executable binaries for lsCOMP, ```lsCOMP_xpcs``` for uint16 type and ```lsCOMP_cssi``` for uint32 data type. If XPCS dataset has an uint32 type, it should be compressed using ```lsCOMP_cssi``` command here (sorry for the confusion I made here...).
Taking XPCS dataset with uint16 data type as an example, its usage can be shown as below:

```shell
Usage:
   lsCOMP_xpcs -i oriFilePath -d dims.x dims.y dims.z -b quantBins.x quantBins.y quantBins.z quantBins.w -p value -o decFilePath
Options:
   -i oriFilePath: Path to the original data file
   -d dims.x dims.y dims.z: Dimensions of the original data, where dim.z is the fastest dimension.
   -b quantBins.x quantBins.y quantBins.z quantBins.w: Quantization bins for the 4 levels, where x is the base one and x<=y<=z<=w.
   -p value: Pooling threshold for a data block.
   -x cmpFilePath: Path to the compressed data file (optional).
   -o decFilePath: Path to the decompressed data file (optional).
Examples:
   lsCOMP_xpcs -i data/xpcs.bin -d 1024 1813 1558 -b 3 5 10 15 -p 0.5
   lsCOMP_xpcs -i data/xpcs.bin -d 1024 1813 1558 -b 3 5 10 15 -p 0.5 -x data/xpcs-cmp.bin
   lsCOMP_xpcs -i data/xpcs.bin -d 1024 1813 1558 -b 3 5 10 15 -p 0.5 -o data/xpcs-dec.bin
```

Assuming the light source dataset has 3 dimension 300 1813 1558, then it means there are 300 2D images and each image has dimension 1813 x 1558.
There are two lossy modes, adaptive scalar quantization bin ```-b``` and selective pooling ```-p```, they are lossy modes. If you want to use lossless compression, you can set all of them as 1. For example:

```shell
lsCOMP_xpcs -i your-data.bin -d 500 1813 1558 -b 1 1 1 1 -p 1
```

Then it will be compressed using lossless format.
To save the compressed data, you can use ```-x``` flag; to save the reconstructed data (if you are using lossless, the reconstructed data is identical), you can use ```-o``` flag. Note that ```-x``` and ```-o``` flags are optional in execution.

A sample output after compression can be shown as below:

```shell
lsCOMP_xpcs -i xpcs-512-1.bin -d 512 1813 1558 -b 1 1 1 1 -p 1 -x xpcs-cmp.bin -o xpcs-dec.bin
GPU warmup finished!

Dataset information:
  - dims:   512 x 1813 x 1558
  - length: 1446222848
  - size:   2.693800 GB
Input arguments:
  - quantBins: 1 1 1 1
  - poolingTH: 1.000000

lsCOMP compression   end-to-end speed: 272.850304 GB/s
lsCOMP decompression end-to-end speed: 215.919109 GB/s
lsCOMP compression ratio: 6.999132
```
Above results are tested on my local PC with a RTX 3080 GPU.

```lsCOMP_cssi``` can be executed bin the same way.

### Using nvCOMP Compressors

nvCOMP compressors are also configured in local path and can be executed directly. The commands include:
```shell
benchmark_allgather         benchmark_deflate_chunked   benchmark_lz4_synth
benchmark_ans_chunked       benchmark_gdeflate_chunked  benchmark_snappy_chunked
benchmark_bitcomp_chunked   benchmark_hlif              benchmark_snappy_synth
benchmark_cascaded_chunked  benchmark_lz4_chunked       benchmark_zstd_chunked
```

Assuming we have a dataset ```data.bin```, it can be compressed using commands:

```shell
# compress with cascaded, the fastest one
benchmark_cascaded_chunked -f data.bin

# compress with lz4, high speed and relatively high ratio
benchmark_lz4_chunked -f data.bin

# compress with zstd, the highest ratio one
benchmark_zstd_chunked -f data.bin
```

A sample output can be shown as below:
```shell
benchmark_zstd_chunked -f pawpawsaurus_958x646x1088_uint16.raw 
----------
files: 1
uncompressed (B): 1346656768
comp_size: 1121236226, compressed ratio: 1.2010
compression throughput (GB/s): 1.5662
decompression throughput (GB/s): 27.0891
```
Above results are tested on my local PC with a RTX 3080 GPU.

## Contact
Yafan Huang, yafan-huang@uiowa.edu