// Host-only Parquet page-header parser (Thrift CompactProtocol).
//
// Parses the per-page `PageHeader` structs of a Parquet column chunk so the
// GPU reader can locate each page's compressed payload and its
// value/def-level/rep-level layout. Pure host C++ (compiled by the conda g++,
// not nvcc) -- the heavy decompress + decode runs on the GPU in fused.cu.
//
// Scope (matches the TPC-H Parquet files the bench generates): DATA_PAGE (v1)
// and DICTIONARY_PAGE; fields type(1), uncompressed_page_size(2),
// compressed_page_size(3), data_page_header(5){num_values, encoding,
// def_level_encoding, rep_level_encoding}, dictionary_page_header(7)
// {num_values, encoding}. Unknown/optional fields (crc, statistics, v2 header)
// are skipped with a correct CompactProtocol field-skip so offsets stay exact.
// Any unparseable header throws std::runtime_error -> the caller defers the
// column chunk to the cuDF path.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ryudb_pq {

// Parquet PageType enum values.
enum PageType { PT_DATA = 0, PT_INDEX = 1, PT_DICT = 2, PT_DATA_V2 = 3 };

// Parquet Encoding enum values we care about.
enum Encoding {
    ENC_PLAIN = 0,
    ENC_PLAIN_DICT = 2,
    ENC_RLE = 3,
    ENC_RLE_DICT = 8,
};

// One page within a column chunk. `payload_off` is the absolute file offset of
// the compressed page payload (just past the Thrift header). `header_len` is
// the Thrift header byte count (so next page = payload_off + comp_size).
struct PageDesc {
    int type;            // PT_*
    int64_t payload_off; // absolute offset of compressed payload in the file
    int header_len;      // Thrift PageHeader byte count
    int comp_size;       // compressed_page_size
    int uncomp_size;     // uncompressed_page_size
    int num_values;      // from data/dict page header
    int value_encoding;  // PLAIN / PLAIN_DICT / RLE_DICT / ...
    int def_level_encoding;
    int rep_level_encoding;
};

// Parse every page in the column chunk at file offset `chunk_off` spanning
// `total_compressed_size` bytes. `file_data` must point to the full file
// (mmap'd or read) and `file_len` is its size. Throws on any parse error or if
// the page stream does not exactly consume `total_compressed_size` bytes.
std::vector<PageDesc> parse_column_chunk_pages(const uint8_t *file_data, size_t file_len,
                                               size_t chunk_off, int total_compressed_size);

} // namespace ryudb_pq