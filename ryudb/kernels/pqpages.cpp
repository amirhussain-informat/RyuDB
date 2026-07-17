// Host-only Parquet page-header parser (Thrift CompactProtocol). See pqpages.h.
//
// CompactProtocol recap (only what Parquet PageHeader needs):
//   field header: one byte (delta<<4 | type) when delta in 1..15; else 0x00
//     followed by an i16 zigzag varint field id, then a type byte.
//   types: STOP=0, BOOL_TRUE=1, BOOL_FALSE=2, BYTE=3, I16=4, I32=5, I64=6,
//     DOUBLE=7, BINARY=8, LIST=9, SET=10, MAP=11, STRUCT=12.
//   i16/i32/i64: zigzag varint (LE base-128, zigzag on the integer value).
//   double: 8 bytes (little-endian).  bool: value carried in the field header
//   (no extra bytes).  binary: varint length then bytes.
//   struct: fields until a STOP (0x00) byte.
//   list: element-type byte then varint count then `count` elements.
//
// Parquet pages are laid out `[Thrift PageHeader][compressed payload]` with NO
// length prefix; the header is self-delimiting via its STOP byte, and
// compressed_page_size (field 3) gives the payload length. The next page begins
// at payload_off + comp_size. total_compressed_size == sum(header_len) +
// sum(comp_size), which parse_column_chunk_pages asserts.

#include "pqpages.h"

#include <cstring>

namespace ryudb_pq {

namespace {

// Cursor over a byte range; throws on overrun.
struct Cursor {
    const uint8_t *base;
    const uint8_t *p;
    const uint8_t *end;
    explicit Cursor(const uint8_t *b, size_t len) : base(b), p(b), end(b + len) {}

    uint8_t byte() {
        if (p >= end) throw std::runtime_error("pq: eof reading byte");
        return *p++;
    }
    void skip(size_t n) {
        if (p + n > end) throw std::runtime_error("pq: eof skipping");
        p += n;
    }
    size_t pos() const { return (size_t)(p - base); }
    size_t remaining() const { return (size_t)(end - p); }

    // zigzag varint -> int64. CompactProtocol varints are base-128 LE with a
    // continuation bit; the integer is then zigzag-decoded.
    int64_t zigzag_varint() {
        uint64_t u = 0;
        int shift = 0;
        for (int i = 0; i < 10; i++) {
            uint8_t b = byte();
            u |= (uint64_t)(b & 0x7f) << shift;
            if (!(b & 0x80)) {
                // zigzag decode
                return (int64_t)((u >> 1) ^ -(int64_t)(u & 1));
            }
            shift += 7;
        }
        throw std::runtime_error("pq: varint too long");
    }
};

// CompactProtocol type ids.
enum CType {
    CT_STOP = 0, CT_BOOL_T = 1, CT_BOOL_F = 2, CT_BYTE = 3, CT_I16 = 4,
    CT_I32 = 5, CT_I64 = 6, CT_DOUBLE = 7, CT_BINARY = 8, CT_LIST = 9,
    CT_SET = 10, CT_MAP = 11, CT_STRUCT = 12,
};

// Skip one field value of the given CompactProtocol type (used to skip
// unknown/optional PageHeader fields such as crc, statistics, v2 headers).
// Struct recurses to its STOP; list reads elem-type + count then skips each.
void skip_value(Cursor &c, int type) {
    switch (type) {
        case CT_BOOL_T:
        case CT_BOOL_F:
            return; // value carried in the field header
        case CT_BYTE:
            c.skip(1);
            return;
        case CT_I16:
        case CT_I32:
        case CT_I64:
            c.zigzag_varint();
            return;
        case CT_DOUBLE:
            c.skip(8);
            return;
        case CT_BINARY: {
            int64_t n = c.zigzag_varint();
            if (n < 0) throw std::runtime_error("pq: negative binary length");
            c.skip((size_t)n);
            return;
        }
        case CT_STRUCT:
            for (;;) {
                uint8_t fh = c.byte();
                int ftype = fh & 0x0f;
                if (ftype == CT_STOP) return;
                int delta = (fh >> 4) & 0x0f;
                if (delta == 0) c.zigzag_varint(); // explicit field id (i16)
                skip_value(c, ftype);
            }
        case CT_LIST:
        case CT_SET: {
            uint8_t et = c.byte();
            int elem_type = et & 0x0f;
            int64_t n = c.zigzag_varint();
            if (n < 0) throw std::runtime_error("pq: negative list size");
            for (int64_t i = 0; i < n; i++) skip_value(c, elem_type);
            return;
        }
        case CT_MAP: {
            int64_t n = c.zigzag_varint();
            if (n < 0) throw std::runtime_error("pq: negative map size");
            if (n == 0) return;
            uint8_t kt = c.byte();
            uint8_t vt = c.byte();
            for (int64_t i = 0; i < n; i++) {
                skip_value(c, kt & 0x0f);
                skip_value(c, vt & 0x0f);
            }
            return;
        }
        default:
            throw std::runtime_error("pq: unknown compact type in skip");
    }
}

// Read a field header. Returns the field type (CT_STOP if end-of-struct) and
// writes the field id (absolute, using `last_id` as the previous field id).
int read_field(Cursor &c, int &last_id) {
    uint8_t fh = c.byte();
    int ftype = fh & 0x0f;
    if (ftype == CT_STOP) return CT_STOP;
    int delta = (fh >> 4) & 0x0f;
    if (delta == 0) {
        // explicit field id: i16 zigzag varint
        last_id = (int)c.zigzag_varint();
    } else {
        last_id += delta;
    }
    return ftype;
}

// Parse DataPageHeader (struct field 5 of PageHeader). num_values(1, i32),
// encoding(2, i32), def_level_encoding(3, i32), rep_level_encoding(4, i32);
// statistics(5) skipped.
void parse_data_page_header(Cursor &c, PageDesc &pg) {
    int last = 0;
    for (;;) {
        int ftype = read_field(c, last);
        if (ftype == CT_STOP) return;
        if (ftype != CT_I32) { // unexpected; skip to be safe
            skip_value(c, ftype);
            continue;
        }
        int64_t v = c.zigzag_varint();
        switch (last) {
            case 1: pg.num_values = (int)v; break;
            case 2: pg.value_encoding = (int)v; break;
            case 3: pg.def_level_encoding = (int)v; break;
            case 4: pg.rep_level_encoding = (int)v; break;
            default: break; // statistics handled below if STRUCT, but it's field 5
        }
    }
}

// Parse DictionaryPageHeader (struct field 7 of PageHeader). num_values(1),
// encoding(2); both i32.
void parse_dict_page_header(Cursor &c, PageDesc &pg) {
    int last = 0;
    for (;;) {
        int ftype = read_field(c, last);
        if (ftype == CT_STOP) return;
        if (ftype != CT_I32) { skip_value(c, ftype); continue; }
        int64_t v = c.zigzag_varint();
        switch (last) {
            case 1: pg.num_values = (int)v; break;
            case 2: pg.value_encoding = (int)v; break;
            default: break;
        }
    }
}

// Parse one PageHeader struct starting at the cursor. Fills pg.type /
// uncomp_size / comp_size / num_values / encodings, and advances the cursor
// past the STOP byte (so the cursor is then at the compressed payload).
void parse_page_header(Cursor &c, PageDesc &pg) {
    pg.num_values = 0;
    pg.value_encoding = -1;
    pg.def_level_encoding = -1;
    pg.rep_level_encoding = -1;
    int last = 0;
    for (;;) {
        int ftype = read_field(c, last);
        if (ftype == CT_STOP) return;
        switch (last) {
            case 1: // type (i32 enum)
                if (ftype != CT_I32) throw std::runtime_error("pq: bad type field");
                pg.type = (int)c.zigzag_varint();
                break;
            case 2: // uncompressed_page_size (i32)
                if (ftype != CT_I32) throw std::runtime_error("pq: bad uncomp field");
                pg.uncomp_size = (int)c.zigzag_varint();
                break;
            case 3: // compressed_page_size (i32)
                if (ftype != CT_I32) throw std::runtime_error("pq: bad comp field");
                pg.comp_size = (int)c.zigzag_varint();
                break;
            case 5: // data_page_header (struct)
                if (ftype != CT_STRUCT) { skip_value(c, ftype); break; }
                parse_data_page_header(c, pg);
                break;
            case 7: // dictionary_page_header (struct)
                if (ftype != CT_STRUCT) { skip_value(c, ftype); break; }
                parse_dict_page_header(c, pg);
                break;
            default: // crc(4), index_page_header(6), data_page_header_v2(8), ...
                skip_value(c, ftype);
                break;
        }
    }
}

} // namespace

std::vector<PageDesc> parse_column_chunk_pages(const uint8_t *file_data, size_t file_len,
                                               size_t chunk_off, int total_compressed_size) {
    if (chunk_off + (size_t)total_compressed_size > file_len)
        throw std::runtime_error("pq: column chunk extends past file end");
    Cursor c(file_data + chunk_off, (size_t)total_compressed_size);
    std::vector<PageDesc> pages;
    size_t consumed = 0;
    while (consumed < (size_t)total_compressed_size) {
        size_t header_start = c.pos();
        PageDesc pg{};
        pg.payload_off = 0; // set below relative to file base
        parse_page_header(c, pg);
        size_t header_len = c.pos() - header_start;
        pg.header_len = (int)header_len;
        if (pg.comp_size <= 0)
            throw std::runtime_error("pq: non-positive compressed_page_size");
        // payload starts right after the header, within the chunk.
        pg.payload_off = (int64_t)(chunk_off + c.pos());
        if (c.remaining() < (size_t)pg.comp_size)
            throw std::runtime_error("pq: page payload extends past chunk");
        pages.push_back(pg);
        c.skip((size_t)pg.comp_size);
        consumed = c.pos();
    }
    if (consumed != (size_t)total_compressed_size)
        throw std::runtime_error("pq: page stream did not consume total_compressed_size");
    return pages;
}

} // namespace ryudb_pq