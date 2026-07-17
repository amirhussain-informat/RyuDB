"""Build the C++/CUDA fused kernel extension with nvcc (opt-in).

    python -m ryudb.kernels.build

Compiles ``fused.cu`` (pybind11 + CUDA) to ``fused.so`` next to this file using
the nvcc already present in the ``ryudb`` conda env and the conda host compiler
(``x86_64-conda-linux-gnu-g++``). The resulting extension is loaded by
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


def build() -> Path:
    import pybind11
    import sysconfig

    if not SRC.exists():
        raise SystemExit(f"missing source: {SRC}")

    py_inc = sysconfig.get_path("include")
    pybind_inc = pybind11.get_include()
    ccbin = _host_compiler()
    nvcc = _resolve_nvcc()

    # Compute capability 8.6 = RTX 3090 (Ampere).
    cmd = [
        nvcc, "-O3", "-arch=sm_86", "-std=c++17",
        "-shared", "-Xcompiler", "-fPIC",
        "-ccbin", ccbin,
        f"-I{pybind_inc}", f"-I{py_inc}",
        str(SRC), "-o", str(OUT),
    ]
    print("[build] nvcc + pybind11 -> fused.so")
    print("[build] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[build] wrote {OUT}")
    return OUT


if __name__ == "__main__":
    build()