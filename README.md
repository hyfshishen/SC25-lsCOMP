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
