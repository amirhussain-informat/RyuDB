"""C++/CUDA fused-kernel extension (optional, nvcc-built).

Loads ``fused.so`` (built by :mod:`ryudb.kernels.build`) and re-exports
``fused_agg``. If the extension is not built, importing this package raises
``ImportError`` with a clear build hint; callers should catch it and fall back to
the Numba/cuDF paths (see :mod:`ryudb.exec.fused`).

A staleness guard refuses to load a ``fused.so`` whose sources (``fused.cu`` or
``pqpages.cpp``) are newer than the binary: a stale module with a changed ABI
would feed wrong descriptors to the kernel (CUDA context poison), so it is
treated as unavailable and the executor falls back to cuDF. Rebuild with
``python -m ryudb.kernels.build`` after editing the sources.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXT = _HERE / "fused.so"
_SRCS = (_HERE / "fused.cu", _HERE / "pqpages.cpp")


def _stale() -> bool:
    """True if a source is newer than the built extension (or it is missing)."""
    if not _EXT.exists():
        return True
    so_mtime = _EXT.stat().st_mtime
    return any(src.exists() and src.stat().st_mtime > so_mtime for src in _SRCS)


def _load():
    if _stale():
        missing = "" if _EXT.exists() else f" ({_EXT} missing)"
        raise ImportError(
            f"C++ fused kernel not built or stale{missing}. "
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
    fused_scan_agg = _mod.fused_scan_agg
    fused_scan_finalize = _mod.fused_scan_finalize
    pqpages_probe = _mod.pqpages_probe
    is_available = True
except ImportError:
    fused_agg = None  # type: ignore[assignment]
    fused_join_agg = None  # type: ignore[assignment]
    fused_scan_agg = None  # type: ignore[assignment]
    fused_scan_finalize = None  # type: ignore[assignment]
    pqpages_probe = None  # type: ignore[assignment]
    is_available = False