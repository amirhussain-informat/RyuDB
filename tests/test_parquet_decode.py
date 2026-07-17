"""Tests for the host Parquet page-header parser + nvCOMP decode path (Phase 5).

The cold reader depends on `ryudb/kernels/pqpages.cpp` (a hand-rolled Thrift
CompactProtocol PageHeader parser) and `ryudb/kernels/fused.cu`'s nvCOMP Snappy
decompress. The parser is exercised through the `pqpages_probe` pybind hook,
which mmaps a file and returns one (type, header_len, comp_size, num_values,
encoding) tuple per page in a column chunk. These tests pin the parser's two
load-bearing invariants against DuckDB-written Snappy parquet (the same writer
the bench uses):

  * `sum(header_len + comp_size) == total_compressed_size` for every chunk --
    a wrong field-skip or a missed page corrupts every subsequent offset, so
    this catches drift immediately.
  * `sum(data-page num_values) == cc.num_values` per chunk -- the decode kernel
    reads exactly this many values.

The end-to-end decode+filter+aggregate correctness lives in
`test_fused_scan_agg.py`; this file is the lower-level parser guard.
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from ryudb.exec import fused

# ryudb_pq PageType / Encoding enums (see pqpages.h).
_PT_DATA = 0
_PT_DICT = 2


def _kernels():
    return fused._kernels


@pytest.fixture(scope="module")
def typed_path(typed_lineitem_dir):
    return str(typed_lineitem_dir / "lineitem" / "0.parquet")


def test_page_parser_sums_match_per_chunk(typed_path):
    """For every (column, row-group) chunk the parser must exactly consume
    total_compressed_size and report the chunk's num_values on its data page(s).
    """
    k = _kernels()
    if not k.is_available:
        pytest.skip("C++ fused kernel not built")
    pf = pq.ParquetFile(typed_path)
    md = pf.metadata
    names = pf.schema_arrow.names
    saw_plain = saw_dict = False
    for rg in range(md.num_row_groups):
        for j, name in enumerate(names):
            cc = md.row_group(rg).column(j)
            off = cc.dictionary_page_offset
            if off is None:
                off = cc.data_page_offset
            pages = k.pqpages_probe(typed_path, int(off), int(cc.total_compressed_size))
            # header_len + comp_size must sum to total_compressed_size exactly.
            total = sum(p[1] + p[2] for p in pages)
            assert total == cc.total_compressed_size, (
                f"{name} rg{rg}: header+comp {total} != total_compressed_size "
                f"{cc.total_compressed_size} ({len(pages)} pages)"
            )
            # num_values on the data page(s) must equal the chunk's num_values.
            data_vals = sum(p[3] for p in pages if p[0] == _PT_DATA)
            assert data_vals == cc.num_values, (
                f"{name} rg{rg}: data-page num_values {data_vals} != "
                f"cc.num_values {cc.num_values}"
            )
            if any(p[0] == _PT_DICT for p in pages):
                saw_dict = True
            else:
                saw_plain = True
    # The fixture deliberately has both a PLAIN column (l_orderkey) and
    # PLAIN_DICTIONARY columns (l_discount, l_tax) so both parse branches run.
    assert saw_plain and saw_dict, "fixture should exercise both PLAIN and DICT chunks"


def test_page_parser_dict_page_has_no_def_section(typed_path):
    """A DICTIONARY page carries PLAIN uniques with no def-level section; the
    parser reports its num_values and the decode path relies on this shape."""
    k = _kernels()
    if not k.is_available:
        pytest.skip("C++ fused kernel not built")
    pf = pq.ParquetFile(typed_path)
    md = pf.metadata
    names = pf.schema_arrow.names
    j = names.index("l_discount")  # PLAIN_DICTIONARY in the fixture
    cc = md.row_group(0).column(j)
    off = cc.dictionary_page_offset
    assert off is not None, "l_discount should have a dictionary page"
    pages = k.pqpages_probe(typed_path, int(off), int(cc.total_compressed_size))
    dict_pages = [p for p in pages if p[0] == _PT_DICT]
    assert len(dict_pages) == 1, f"expected exactly 1 dict page, got {len(dict_pages)}"
    # dict page num_values is the dictionary cardinality (small, < num_values).
    assert 0 < dict_pages[0][3] < cc.num_values


def test_page_parser_raises_on_bad_offset(typed_path):
    """A truncated/garbage chunk offset must raise (the caller catches and defers
    to cuDF) rather than silently return wrong pages."""
    k = _kernels()
    if not k.is_available:
        pytest.skip("C++ fused kernel not built")
    pf = pq.ParquetFile(typed_path)
    file_len = pf.metadata.num_rows  # nonzero; use a bogus tiny total size
    with pytest.raises(Exception):  # noqa: B017 -- any parse/RuntimeError is fine
        k.pqpages_probe(typed_path, 0, 10)
    _ = file_len