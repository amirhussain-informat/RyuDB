# RyuDB — single-artifact GPU image (Tier C "single binary", reframed for a
# CUDA app). A conda env with the full CUDA toolkit (cuDF + nvcc + libnvcomp) +
# the built fused kernel + the ryudb-server entrypoint, runnable with --gpus all.
#
# The conda env (docker/env.yml) brings nvcc, cudart, and libnvcomp, so the base
# is plain ubuntu; the NVIDIA driver is injected by the NVIDIA Container Toolkit
# (--gpus all mounts the host driver into the container). No PyInstaller blob —
# bundling cuDF + RAPIDS + the CUDA runtime in a frozen binary is fragile; a
# Docker image is the standard "single artifact" for a GPU app.
#
#   docker build -t ryudb -f docker/Dockerfile .
#   docker run --gpus all -p 5430:5430 -p 5432:5432 -v ryudb-data:/data ryudb
#   psql -h 127.0.0.1 -p 5432 -U ryudb ryudb        # Postgres wire front
#
# Target GPU arch is chosen at build time (fat binary; default Ampere/Ada/Hopper):
#   docker build --build-arg CUDA_ARCH=sm_70,sm_75 -t ryudb .   # Turing-only

########################################
# Builder: create the conda env, install ryudb, compile the fused kernel.
########################################
FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl bzip2 ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*
# Miniforge = conda + mamba (fast solves) with conda-forge as the default channel.
RUN curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p $CONDA_DIR && rm /tmp/miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH
WORKDIR /build
COPY docker/env.yml /build/env.yml
# Reproduce the exact working env (docker/env.yml is a --no-builds export of the
# dev env). mamba clean drops the package cache (~GBs). If the `defaults`
# channel triggers an anaconda.com TOS error, remove the `- defaults` line from
# docker/env.yml (nearly everything resolves from conda-forge + rapidsai).
RUN mamba env create -f /build/env.yml && mamba clean -afy
# Install ryudb from source into the env. --no-deps: the deps (cudf, sqlglot,
# pyarrow, websockets, pybind11) are already satisfied by the conda env, and
# `cudf` is not on PyPI under that name, so a dep-resolving install would fail.
COPY . /build/ryudb
RUN $CONDA_DIR/envs/ryudb/bin/pip install /build/ryudb --no-deps
# Compile the fused kernel. CONDA_PREFIX + PATH must point at the env so
# build.py finds nvcc, the conda host compiler (x86_64-conda-linux-gnu-g++), and
# libnvcomp. RYUDB_CUDA_ARCH selects the target compute capability(ies).
ENV CONDA_PREFIX=$CONDA_DIR/envs/ryudb
ENV PATH=$CONDA_DIR/envs/ryudb/bin:$PATH
ARG CUDA_ARCH=sm_80,sm_86,sm_89,sm_90
ENV RYUDB_CUDA_ARCH=$CUDA_ARCH
RUN ryudb build

########################################
# Runtime: just the built env (fused.so is in site-packages) + the entrypoint.
########################################
FROM ubuntu:22.04 AS runtime
ENV RYUDB_ENV=/opt/conda/envs/ryudb
COPY --from=builder $RYUDB_ENV $RYUDB_ENV
ENV PATH=$RYUDB_ENV/bin:$PATH
ENV LD_LIBRARY_PATH=$RYUDB_ENV/lib
ENV CONDA_PREFIX=$RYUDB_ENV
# Container defaults: bind all interfaces (so host port mapping works — the
# server has NO AUTH, so only expose on a trusted network / behind a proxy);
# enable the Postgres wire front; data lives on a volume. Override with -e or
# the matching --flag (RYUDB_* env vars are read by ryudb-server).
ENV RYUDB_HOST=0.0.0.0 RYUDB_PORT=5430 RYUDB_PG_PORT=5432 RYUDB_DATA=/data
VOLUME /data
EXPOSE 5430 5432
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]