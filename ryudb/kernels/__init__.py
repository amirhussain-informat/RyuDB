"""C++/CUDA fused-kernel extension (optional, nvcc-built).

Loads ``fused.so`` (built by :mod:`ryudb.kernels.build`) and re-exports
``fused_agg``. If the extension is not built, importing this package raises
``ImportError`` with a clear build hint; callers should catch it and fall back to
the Numba/cuDF paths (see :mod:`ryudb.exec.fused`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_EXT = Path(__file__).resolve().parent / "fused.so"


def _load():
    if not _EXT.exists():
        raise ImportError(
            f"C++ fused kernel not built ({_EXT} missing). "
            "Build it with:  python -m ryudb.kernels.build  "
            "(requires nvcc + a host compiler; see ryudb/kernels/build.py)."
        )
    # The spec name's final component must match the PYBIND11_MODULE name ("fused")
    # so CPython finds PyInit_fused.
    spec = importlib.util.spec_from_file_location("ryudb.kernels.fused", _EXT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


try:
    _mod = _load()
    fused_agg = _mod.fused_agg
    fused_join_agg = _mod.fused_join_agg
    is_available = True
except ImportError:
    fused_agg = None  # type: ignore[assignment]
    fused_join_agg = None  # type: ignore[assignment]
    is_available = False