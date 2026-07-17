"""Build the C++/CUDA fused-kernel extension with nvcc (opt-in).

    python -m ryudb.kernels.build

Compiles ``fused.cu`` (pybind11 + CUDA) plus the host-only Parquet page-header
parser ``pqpages.cpp`` to ``fused.so`` next to this file using the nvcc already
present in the ``ryudb`` conda env and the conda host compiler
(``x86_64-conda-linux-gnu-g++``). Links ``libnvcomp`` for GPU Snappy batch
decompression (Phase 5 cold reader). The resulting extension is loaded by
``ryudb.kernels``; if it is absent, the executor falls back to the Numba/cuDF
paths, so building is never required for correctness.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "fused.cu"
PQSRC = HERE / "pqpages.cpp"
OUT = HERE / "fused.so"


def _resolve_nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise SystemExit(
            "nvcc not found on PATH. The ryudb conda env should ship it "
            "(CUDA 13.x). Activate the env: `conda activate ryudb`."
        )
    return nvcc


def _host_compiler() -> str:
    ccbin = os.environ.get("CONDA_PREFIX", "") + "/bin/x86_64-conda-linux-gnu-g++"
    if not Path(ccbin).exists():
        # Fall back to whatever g++ nvcc can find on PATH.
        ccbin = shutil.which("x86_64-conda-linux-gnu-g++") or shutil.which("g++") or ""
    if not ccbin or not Path(ccbin).exists():
        raise SystemExit(
            "No host compiler for nvcc. Install one in the env:\n"
            "  conda install -n ryudb -c conda-forge gxx_linux-64 gcc_linux-64 sysroot_linux-64"
        )
    return ccbin


def _conda_prefix() -> str:
    pre = os.environ.get("CONDA_PREFIX", "")
    if not pre or not Path(pre).exists():
        raise SystemExit(
            "CONDA_PREFIX not set or invalid. Activate the ryudb env: "
            "`conda activate ryudb` (needed for libnvcomp headers/libs)."
        )
    return pre


def build() -> Path:
    import pybind11
    import sysconfig

    if not SRC.exists():
        raise SystemExit(f"missing source: {SRC}")
    if not PQSRC.exists():
        raise SystemExit(f"missing source: {PQSRC}")

    py_inc = sysconfig.get_path("include")
    pybind_inc = pybind11.get_include()
    ccbin = _host_compiler()
    nvcc = _resolve_nvcc()
    pre = _conda_prefix()
    nvcomp_inc = f"{pre}/include"
    nvcomp_lib = f"{pre}/lib"
    pq_obj = HERE / "pqpages.o"

    # 1) Compile the host-only Parquet page-header parser (Thrift
    # CompactProtocol) to an object with the conda g++. No CUDA, no pybind.
    gpp_cmd = [
        ccbin, "-O2", "-std=c++17", "-fPIC", "-c", str(PQSRC), "-o", str(pq_obj),
        f"-I{HERE}",
    ]
    print("[build] g++ -> pqpages.o")
    print("[build] " + " ".join(gpp_cmd))
    subprocess.run(gpp_cmd, check=True)

    # 2) Compile+link fused.cu (CUDA + pybind11) with pqpages.o + libnvcomp.
    # nvcc drives the link and hands pqpages.o + -lnvcomp to the host linker.
    # -rpath to CONDA_PREFIX/lib so fused.so finds libnvcomp at import time
    # without relying on LD_LIBRARY_PATH.
    cmd = [
        nvcc, "-O3", "-arch=sm_86", "-std=c++17",
        "-shared", "-Xcompiler", "-fPIC",
    ] + (["-DRYUDB_SCAN_PROFILE"] if os.environ.get("RYUDB_SCAN_PROFILE") else []) + [
        "-ccbin", ccbin,
        f"-I{pybind_inc}", f"-I{py_inc}", f"-I{nvcomp_inc}", f"-I{HERE}",
        str(SRC), str(pq_obj),
        f"-L{nvcomp_lib}", "-lnvcomp",
        "-Xlinker", "-rpath", "-Xlinker", nvcomp_lib,
        "-o", str(OUT),
    ]
    print("[build] nvcc + pybind11 + nvcomp -> fused.so")
    print("[build] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[build] wrote {OUT}")
    return OUT


if __name__ == "__main__":
    build()