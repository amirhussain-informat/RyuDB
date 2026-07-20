"""Packaging tests — the kernel sources ship with the installed package.

``ryudb.kernels.build`` reads ``fused.cu`` / ``pqpages.cpp`` / ``pqpages.h``
from ``Path(__file__).parent``. With an editable install that is the source
tree; with a built wheel it is the installed package dir — so the sources MUST
be declared as ``package-data`` in ``pyproject.toml`` or the build fails from a
non-editable install ("missing source"). This test pins that the sources are
reachable via ``importlib.resources`` (true for both editable and wheel
installs once ``package-data`` ships them).
"""

from __future__ import annotations

import importlib.resources as resources


def test_kernel_sources_ship_with_package():
    """fused.cu / pqpages.cpp / pqpages.h are reachable as package resources."""
    root = resources.files("ryudb.kernels")
    for name in ("fused.cu", "pqpages.cpp", "pqpages.h"):
        res = root.joinpath(name)
        assert res.is_file(), f"kernel source {name!r} not shipped with the package"


def test_build_module_is_importable():
    """``ryudb.kernels.build`` imports (it is shipped, even if nvcc is absent —
    importing it does not build; only ``build()`` does)."""
    import ryudb.kernels.build as b  # noqa: F401
    assert callable(b.build)