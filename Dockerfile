FROM nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu22.04 AS builder

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 cmake

COPY . /build

RUN cd /build/ && \
    python3 0-compile-lsCOMP.py


FROM nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu22.04 AS final
COPY --from=builder /build/lsCOMP/build/lsCOMP_* /bin/
