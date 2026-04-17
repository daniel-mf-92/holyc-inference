// metadata.HC
// GGUF metadata key/value parser for HolyC inference runtime.

#define GGUF_META_PARSE_OK 0
#define GGUF_META_PARSE_ERR_NULL_PTR 1
#define GGUF_META_PARSE_ERR_TRUNCATED 2
#define GGUF_META_PARSE_ERR_BAD_TYPE 3
#define GGUF_META_PARSE_ERR_BAD_STRING_LEN 4
#define GGUF_META_PARSE_ERR_BAD_ARRAY_LEN 5
#define GGUF_META_PARSE_ERR_ALLOC 6
#define GGUF_META_PARSE_ERR_NESTED_ARRAY 7
#define GGUF_META_PARSE_ERR_BAD_COUNT 8
#define GGUF_META_PARSE_ERR_NOT_FOUND 9
#define GGUF_META_PARSE_ERR_TYPE_MISMATCH 10

#define GGUF_META_TABLE_OK 0
#define GGUF_META_TABLE_ERR_NULL_PTR 1
#define GGUF_META_TABLE_ERR_BAD_PARAM 2
#define GGUF_META_TABLE_ERR_OVERFLOW 3
#define GGUF_META_TABLE_ERR_OUT_OF_BOUNDS 4

#define GGUF_MAX_METADATA_COUNT (1 << 20)
#define GGUF_MAX_STRING_BYTES   (1 << 20)
#define GGUF_MAX_ARRAY_ELEMS    (1 << 24)

#define GGUF_TYPE_UINT8   0
#define GGUF_TYPE_INT8    1
#define GGUF_TYPE_UINT16  2
#define GGUF_TYPE_INT16   3
#define GGUF_TYPE_UINT32  4
#define GGUF_TYPE_INT32   5
#define GGUF_TYPE_FLOAT32 6
#define GGUF_TYPE_BOOL    7
#define GGUF_TYPE_STRING  8
#define GGUF_TYPE_ARRAY   9
#define GGUF_TYPE_UINT64  10
#define GGUF_TYPE_INT64   11
#define GGUF_TYPE_FLOAT64 12

class GGUFString
{
    U64 len;
    U8 *data;
};

class GGUFMetaValue
{
    U32 type;

    // Scalar slots. Float payloads are preserved as raw bits (u32/u64) because
    // TempleOS runtime stays integer-only.
    U64 u64;
    I64 i64;
    Bool b;
    GGUFString str;

    // Array descriptor. Payload bytes remain in file buffer; we store offsets so
    // later systems can decode lazily without reparsing the table.
    U32 array_elem_type;
    U64 array_len;
    U64 array_payload_off;
    U64 array_payload_bytes;
};

class GGUFMetadataKV
{
    GGUFString key;
    GGUFMetaValue value;
};

class GGUFMetadataTable
{
    U64 count;
    GGUFMetadataKV *items;
};

I32 GGUFMetadataCursorCanAdvanceChecked(U64 cursor,
                                        U64 need,
                                        U64 table_end,
                                        U64 *out_next_cursor);

I32 GGUFMetadataReadU8Checked(U8 *buf,
                              U64 buf_nbytes,
                              U64 *cursor,
                              U64 table_end,
                              U8 *out_value);

I32 GGUFMetadataReadI8Checked(U8 *buf,
                              U64 buf_nbytes,
                              U64 *cursor,
                              U64 table_end,
                              I8 *out_value);

I32 GGUFMetadataReadU8PairChecked(U8 *buf,
                                  U64 buf_nbytes,
                                  U64 *cursor,
                                  U64 table_end,
                                  U8 *out_first,
                                  U8 *out_second);

I32 GGUFMetadataReadU16LEChecked(U8 *buf,
                                 U64 buf_nbytes,
                                 U64 *cursor,
                                 U64 table_end,
                                 U16 *out_value);

I32 GGUFMetadataReadU16PairChecked(U8 *buf,
                                   U64 buf_nbytes,
                                   U64 *cursor,
                                   U64 table_end,
                                   U16 *out_first,
                                   U16 *out_second);

I32 GGUFMetadataReadI16LEChecked(U8 *buf,
                                 U64 buf_nbytes,
                                 U64 *cursor,
                                 U64 table_end,
                                 I16 *out_value);

I32 GGUFMetadataReadU32LEChecked(U8 *buf,
                                 U64 buf_nbytes,
                                 U64 *cursor,
                                 U64 table_end,
                                 U32 *out_value);

I32 GGUFMetadataReadU32PairChecked(U8 *buf,
                                   U64 buf_nbytes,
                                   U64 *cursor,
                                   U64 table_end,
                                   U32 *out_first,
                                   U32 *out_second);

I32 GGUFMetadataReadU64LEChecked(U8 *buf,
                                 U64 buf_nbytes,
                                 U64 *cursor,
                                 U64 table_end,
                                 U64 *out_value);

I32 GGUFMetadataReadU64LECheckedNoPartial(U8 *buf,
                                          U64 buf_nbytes,
                                          U64 *cursor,
                                          U64 table_end,
                                          U64 *out_value);

I32 GGUFMetadataReadI32LEChecked(U8 *buf,
                                 U64 buf_nbytes,
                                 U64 *cursor,
                                 U64 table_end,
                                 I32 *out_value);

I32 GGUFMetadataReadI64LEChecked(U8 *buf,
                                 U64 buf_nbytes,
                                 U64 *cursor,
                                 U64 table_end,
                                 I64 *out_value);

I32 GGUFMetadataReadI64LECheckedNoPartial(U8 *buf,
                                          U64 buf_nbytes,
                                          U64 *cursor,
                                          U64 table_end,
                                          I64 *out_value);

I32 GGUFMetadataReadF32BitsLEChecked(U8 *buf,
                                     U64 buf_nbytes,
                                     U64 *cursor,
                                     U64 table_end,
                                     U32 *out_bits);

I32 GGUFMetadataReadF64BitsLEChecked(U8 *buf,
                                     U64 buf_nbytes,
                                     U64 *cursor,
                                     U64 table_end,
                                     U64 *out_bits);

I32 GGUFMetadataReadBoolChecked(U8 *buf,
                                U64 buf_nbytes,
                                U64 *cursor,
                                U64 table_end,
                                Bool *out_value);

I32 GGUFMetadataReadBoolPairChecked(U8 *buf,
                                    U64 buf_nbytes,
                                    U64 *cursor,
                                    U64 table_end,
                                    Bool *out_first,
                                    Bool *out_second);

I32 GGUFMetadataReadStringLenU64Checked(U8 *buf,
                                        U64 buf_nbytes,
                                        U64 *cursor,
                                        U64 table_end,
                                        U64 *out_len);

I32 GGUFMetadataReadStringU64HeaderAndBoundsChecked(U8 *buf,
                                                    U64 buf_nbytes,
                                                    U64 *cursor,
                                                    U64 table_end,
                                                    U64 *out_len,
                                                    U64 *out_payload_end);

I32 GGUFMetadataReadStringBytesChecked(U8 *buf,
                                       U64 buf_nbytes,
                                       U64 *cursor,
                                       U64 table_end,
                                       U64 str_len,
                                       U8 *out_bytes,
                                       U64 out_nbytes);

I32 GGUFMetadataReadStringChecked(U8 *buf,
                                  U64 buf_nbytes,
                                  U64 *cursor,
                                  U64 table_end,
                                  U8 *out_bytes,
                                  U64 out_nbytes,
                                  U64 *out_len);

I32 GGUFMetadataReadArrayHeaderChecked(U8 *buf,
                                       U64 buf_nbytes,
                                       U64 *cursor,
                                       U64 table_end,
                                       U32 *out_elem_type,
                                       U64 *out_elem_count);

I32 GGUFMetadataArrayPayloadSpanValidateChecked(U64 payload_start,
                                                U64 payload_bytes,
                                                U64 table_end,
                                                U64 buf_nbytes,
                                                U64 *out_payload_end);

I32 GGUFMetadataArrayFixedElemWidthBytesChecked(U32 elem_type,
                                                U64 *out_elem_width_bytes);

I32 GGUFMetadataArrayFixedPayloadBytesChecked(U64 elem_count,
                                              U64 elem_width_bytes,
                                              U64 *out_payload_bytes);

// Validate metadata table byte-window math before any parser cursor movement.
// Contract: `out_table_end` is written only on success.
I32 GGUFMetadataTableSpanValidateChecked(U64 table_start,
                                         U64 table_nbytes,
                                         U64 file_nbytes,
                                         U64 *out_table_end)
{
    U64 table_end;

    if (!out_table_end)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    // Parser cursors are signed in multiple call sites; reject spans that do
    // not fit cleanly in positive I64 address space.
    if (table_start > 0x7FFFFFFFFFFFFFFF)
        return GGUF_META_TABLE_ERR_OVERFLOW;
    if (table_nbytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_META_TABLE_ERR_OVERFLOW;
    if (file_nbytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_META_TABLE_ERR_OVERFLOW;

    if (table_start > file_nbytes)
        return GGUF_META_TABLE_ERR_BAD_PARAM;

    if (table_nbytes > (0xFFFFFFFFFFFFFFFF - table_start))
        return GGUF_META_TABLE_ERR_OVERFLOW;

    table_end = table_start + table_nbytes;
    if (table_end > file_nbytes)
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS;

    *out_table_end = table_end;
    return GGUF_META_TABLE_OK;
}

// Validate metadata table bounds, then derive a checked cursor position at
// `table_start + cursor_advance`. This helper keeps parser entrypoints on one
// auditable range-math path before any field reads happen.
// Contract: out pointers are written only on success.
I32 GGUFMetadataTableCursorAdvanceChecked(U64 table_start,
                                          U64 table_nbytes,
                                          U64 file_nbytes,
                                          U64 cursor_advance,
                                          U64 *out_cursor,
                                          U64 *out_table_end)
{
    U64 table_end;
    I32 err;

    if (!out_cursor || !out_table_end)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    err = GGUFMetadataTableSpanValidateChecked(table_start,
                                               table_nbytes,
                                               file_nbytes,
                                               &table_end);
    if (err != GGUF_META_TABLE_OK)
        return err;

    // Advance is relative to table_start. Equal-to-end is legal and matches
    // cursor semantics used by metadata table loops.
    err = GGUFMetadataCursorCanAdvanceChecked(table_start,
                                              cursor_advance,
                                              table_end,
                                              out_cursor);
    if (err != GGUF_META_TABLE_OK)
        return err;

    *out_table_end = table_end;
    return GGUF_META_TABLE_OK;
}

// Validate `cursor + need <= table_end` with explicit signed/unsigned overflow
// guards. On success, `out_next_cursor` is the advanced cursor position.
// Contract: out pointer is written only on success.
I32 GGUFMetadataCursorCanAdvanceChecked(U64 cursor,
                                        U64 need,
                                        U64 table_end,
                                        U64 *out_next_cursor)
{
    U64 next_cursor;

    if (!out_next_cursor)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    // Keep cursor math in positive I64 space because parser entrypoints route
    // through signed cursors.
    if (cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_META_TABLE_ERR_OVERFLOW;
    if (need > 0x7FFFFFFFFFFFFFFF)
        return GGUF_META_TABLE_ERR_OVERFLOW;
    if (table_end > 0x7FFFFFFFFFFFFFFF)
        return GGUF_META_TABLE_ERR_OVERFLOW;

    if (cursor > table_end)
        return GGUF_META_TABLE_ERR_BAD_PARAM;

    if (cursor > (0xFFFFFFFFFFFFFFFF - need))
        return GGUF_META_TABLE_ERR_OVERFLOW;

    next_cursor = cursor + need;
    if (next_cursor > table_end)
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS;

    *out_next_cursor = next_cursor;
    return GGUF_META_TABLE_OK;
}

// Validate a metadata-array payload window `[payload_start, payload_end)` where
// `payload_end = payload_start + payload_bytes`.
// Contract: `*out_payload_end` is written only on success.
I32 GGUFMetadataArrayPayloadSpanValidateChecked(U64 payload_start,
                                                U64 payload_bytes,
                                                U64 table_end,
                                                U64 buf_nbytes,
                                                U64 *out_payload_end)
{
    U64 payload_end;
    I32 err;

    if (!out_payload_end)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    // Keep data-plane indexing in positive I64 space, matching all checked
    // metadata cursor/table helpers used by parser entrypoints.
    if (buf_nbytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_META_TABLE_ERR_OVERFLOW;

    // Compose payload window math through shared checked cursor helper so
    // signed-space + unsigned-overflow surfaces stay identical.
    err = GGUFMetadataCursorCanAdvanceChecked(payload_start,
                                              payload_bytes,
                                              table_end,
                                              &payload_end);
    if (err != GGUF_META_TABLE_OK)
        return err;

    // Backing byte span must contain the validated payload window.
    if (payload_end > buf_nbytes)
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS;

    *out_payload_end = payload_end;
    return GGUF_META_TABLE_OK;
}

// Return canonical fixed-width byte size for metadata ARRAY element types.
// Reject variable-width ARRAY element tags (STRING/ARRAY) because payload-byte
// math for those must use per-element decoding instead of fixed `count*width`.
// Contract: `*out_elem_width_bytes` is written only on success.
I32 GGUFMetadataArrayFixedElemWidthBytesChecked(U32 elem_type,
                                                U64 *out_elem_width_bytes)
{
    U64 width_bytes;

    if (!out_elem_width_bytes)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    switch (elem_type)
    {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            width_bytes = 1;
            break;

        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            width_bytes = 2;
            break;

        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            width_bytes = 4;
            break;

        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            width_bytes = 8;
            break;

        // Variable-width element payloads require non-fixed decoding paths.
        case GGUF_TYPE_STRING:
        case GGUF_TYPE_ARRAY:
        default:
            return GGUF_META_TABLE_ERR_BAD_PARAM;
    }

    *out_elem_width_bytes = width_bytes;
    return GGUF_META_TABLE_OK;
}

// Checked multiplication helper used by metadata table math where callers keep
// cursor/index values in positive I64 space.
// Contract: writes `*out_product` only on success.
I32 GGUFMetadataMulU64Checked(U64 lhs,
                              U64 rhs,
                              U64 *out_product)
{
    U64 product;

    if (!out_product)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    // Keep table/index arithmetic in positive I64 space for compatibility with
    // parser entrypoints that route through signed cursors.
    if (lhs > 0x7FFFFFFFFFFFFFFF)
        return GGUF_META_TABLE_ERR_OVERFLOW;
    if (rhs > 0x7FFFFFFFFFFFFFFF)
        return GGUF_META_TABLE_ERR_OVERFLOW;

    if (rhs && lhs > (0xFFFFFFFFFFFFFFFF / rhs))
        return GGUF_META_TABLE_ERR_OVERFLOW;

    product = lhs * rhs;
    if (product > 0x7FFFFFFFFFFFFFFF)
        return GGUF_META_TABLE_ERR_OVERFLOW;

    *out_product = product;
    return GGUF_META_TABLE_OK;
}

// Compute fixed-width metadata ARRAY payload bytes (`elem_count*elem_width`).
// Intended for callers that first validate/resolve fixed element widths, then
// compose this payload-byte result with payload-span validation helpers.
// Contract: writes `*out_payload_bytes` only on success.
I32 GGUFMetadataArrayFixedPayloadBytesChecked(U64 elem_count,
                                              U64 elem_width_bytes,
                                              U64 *out_payload_bytes)
{
    U64 payload_bytes;
    I32 err;

    if (!out_payload_bytes)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    // Compose through shared checked multiply guards so all fixed-width array
    // payload math follows one overflow/sign-domain contract.
    err = GGUFMetadataMulU64Checked(elem_count,
                                    elem_width_bytes,
                                    &payload_bytes);
    if (err != GGUF_META_TABLE_OK)
        return err;

    *out_payload_bytes = payload_bytes;
    return GGUF_META_TABLE_OK;
}

// Checked 1-byte metadata reader used by table-backed decoders.
// Contract: writes `*out_value` and advances `*cursor` only on success.
I32 GGUFMetadataReadU8Checked(U8 *buf,
                              U64 buf_nbytes,
                              U64 *cursor,
                              U64 table_end,
                              U8 *out_value)
{
    U64 cur;
    U64 next_cursor;
    I32 err;

    if (!buf || !cursor || !out_value)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    // Keep data-plane indexing in positive I64 space to match parser cursor
    // contracts used across GGUF metadata entrypoints.
    if (buf_nbytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_META_TABLE_ERR_OVERFLOW;

    cur = *cursor;
    err = GGUFMetadataCursorCanAdvanceChecked(cur, 1, table_end, &next_cursor);
    if (err != GGUF_META_TABLE_OK)
        return err;

    // The metadata span must live inside the provided byte buffer.
    if (next_cursor > buf_nbytes)
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS;

    *out_value = buf[cur];
    *cursor = next_cursor;
    return GGUF_META_TABLE_OK;
}

// Checked signed-I8 metadata reader.
// Contract: decodes through GGUFMetadataReadU8Checked and commits `*out_value`
// plus `*cursor` only on success (strict no-partial-write semantics).
I32 GGUFMetadataReadI8Checked(U8 *buf,
                              U64 buf_nbytes,
                              U64 *cursor,
                              U64 table_end,
                              I8 *out_value)
{
    U64 cur;
    U8 raw_u8;
    I32 err;

    if (!buf || !cursor || !out_value)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Stage decode through the canonical checked U8 reader so signed-I8 keeps
    // identical bounds/overflow and no-partial-write contracts.
    err = GGUFMetadataReadU8Checked(buf,
                                    buf_nbytes,
                                    &cur,
                                    table_end,
                                    &raw_u8);
    if (err != GGUF_META_TABLE_OK)
        return err;

    // Two's-complement reinterpretation is a plain cast in HolyC.
    *out_value = (I8)raw_u8;
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked two-byte tuple reader for compact metadata fields.
// Contract: decodes sequential bytes through GGUFMetadataReadU8Checked and
// commits both outputs + cursor only after both reads succeed.
I32 GGUFMetadataReadU8PairChecked(U8 *buf,
                                  U64 buf_nbytes,
                                  U64 *cursor,
                                  U64 table_end,
                                  U8 *out_first,
                                  U8 *out_second)
{
    U64 cur;
    U8 first;
    U8 second;
    I32 err;

    if (!buf || !cursor || !out_first || !out_second)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Stage both reads through the canonical checked U8 reader so bounds and
    // no-partial-write behavior exactly matches scalar decoders.
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &first);
    if (err != GGUF_META_TABLE_OK)
        return err;
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &second);
    if (err != GGUF_META_TABLE_OK)
        return err;

    *out_first = first;
    *out_second = second;
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked little-endian U16 metadata reader.
// Contract: writes `*out_value` and advances `*cursor` only on success.
I32 GGUFMetadataReadU16LEChecked(U8 *buf,
                                 U64 buf_nbytes,
                                 U64 *cursor,
                                 U64 table_end,
                                 U16 *out_value)
{
    U64 cur;
    U8 b0;
    U8 b1;
    I32 err;

    if (!buf || !cursor || !out_value)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Reuse the canonical checked 1-byte reader so 16-bit scalar decode keeps
    // identical cursor/bounds/no-partial-write behavior.
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &b0);
    if (err != GGUF_META_TABLE_OK)
        return err;
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &b1);
    if (err != GGUF_META_TABLE_OK)
        return err;

    *out_value = ((U16)b0)
               | (((U16)b1) << 8);
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked little-endian U16 tuple reader for compact metadata pair fields.
// Contract: stages both U16 decodes through GGUFMetadataReadU16LEChecked and
// commits outputs + cursor only when both lanes succeed.
I32 GGUFMetadataReadU16PairChecked(U8 *buf,
                                   U64 buf_nbytes,
                                   U64 *cursor,
                                   U64 table_end,
                                   U16 *out_first,
                                   U16 *out_second)
{
    U64 cur;
    U16 first;
    U16 second;
    I32 err;

    if (!buf || !cursor || !out_first || !out_second)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Stage both scalar reads through the canonical U16 helper so pair reads
    // preserve identical bounds/overflow and no-partial-write behavior.
    err = GGUFMetadataReadU16LEChecked(buf,
                                       buf_nbytes,
                                       &cur,
                                       table_end,
                                       &first);
    if (err != GGUF_META_TABLE_OK)
        return err;

    err = GGUFMetadataReadU16LEChecked(buf,
                                       buf_nbytes,
                                       &cur,
                                       table_end,
                                       &second);
    if (err != GGUF_META_TABLE_OK)
        return err;

    *out_first = first;
    *out_second = second;
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked little-endian I16 metadata reader.
// Contract: writes `*out_value` and advances `*cursor` only on success.
I32 GGUFMetadataReadI16LEChecked(U8 *buf,
                                 U64 buf_nbytes,
                                 U64 *cursor,
                                 U64 table_end,
                                 I16 *out_value)
{
    U64 cur;
    U16 raw_u16;
    I32 err;

    if (!buf || !cursor || !out_value)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Decode through the shared U16 checked reader so signed-I16 decode keeps
    // identical cursor/bounds/no-partial-write behavior.
    err = GGUFMetadataReadU16LEChecked(buf,
                                       buf_nbytes,
                                       &cur,
                                       table_end,
                                       &raw_u16);
    if (err != GGUF_META_TABLE_OK)
        return err;

    // Two's-complement reinterpretation is a plain cast in HolyC.
    *out_value = (I16)raw_u16;
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked little-endian U32 metadata reader.
// Contract: writes `*out_value` and advances `*cursor` only on success.
I32 GGUFMetadataReadU32LEChecked(U8 *buf,
                                 U64 buf_nbytes,
                                 U64 *cursor,
                                 U64 table_end,
                                 U32 *out_value)
{
    U64 cur;
    U8 b0;
    U8 b1;
    U8 b2;
    U8 b3;
    I32 err;

    if (!buf || !cursor || !out_value)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Reuse the canonical checked 1-byte reader so cursor/bounds semantics are
    // identical across scalar-width metadata decoders.
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &b0);
    if (err != GGUF_META_TABLE_OK)
        return err;
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &b1);
    if (err != GGUF_META_TABLE_OK)
        return err;
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &b2);
    if (err != GGUF_META_TABLE_OK)
        return err;
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &b3);
    if (err != GGUF_META_TABLE_OK)
        return err;

    *out_value = ((U32)b0)
               | (((U32)b1) << 8)
               | (((U32)b2) << 16)
               | (((U32)b3) << 24);
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked little-endian U32 tuple reader for compact metadata pair fields.
// Contract: stages both U32 decodes through GGUFMetadataReadU32LEChecked and
// commits outputs + cursor only when both lanes succeed.
I32 GGUFMetadataReadU32PairChecked(U8 *buf,
                                   U64 buf_nbytes,
                                   U64 *cursor,
                                   U64 table_end,
                                   U32 *out_first,
                                   U32 *out_second)
{
    U64 cur;
    U32 first;
    U32 second;
    I32 err;

    if (!buf || !cursor || !out_first || !out_second)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Stage both scalar reads through the canonical U32 helper so pair reads
    // preserve identical bounds/overflow and no-partial-write behavior.
    err = GGUFMetadataReadU32LEChecked(buf,
                                       buf_nbytes,
                                       &cur,
                                       table_end,
                                       &first);
    if (err != GGUF_META_TABLE_OK)
        return err;

    err = GGUFMetadataReadU32LEChecked(buf,
                                       buf_nbytes,
                                       &cur,
                                       table_end,
                                       &second);
    if (err != GGUF_META_TABLE_OK)
        return err;

    *out_first = first;
    *out_second = second;
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked little-endian U64 metadata reader.
// Contract: writes `*out_value` and advances `*cursor` only on success.
I32 GGUFMetadataReadU64LEChecked(U8 *buf,
                                 U64 buf_nbytes,
                                 U64 *cursor,
                                 U64 table_end,
                                 U64 *out_value)
{
    // Keep legacy symbol as the canonical parser entrypoint while routing
    // decode through the explicit no-partial-write helper.
    return GGUFMetadataReadU64LECheckedNoPartial(buf,
                                                 buf_nbytes,
                                                 cursor,
                                                 table_end,
                                                 out_value);
}

// Checked little-endian U64 metadata reader with explicit all-or-nothing
// staging contract. Cursor/output are committed only after all 8 byte reads
// succeed through the shared checked U8 reader.
I32 GGUFMetadataReadU64LECheckedNoPartial(U8 *buf,
                                          U64 buf_nbytes,
                                          U64 *cursor,
                                          U64 table_end,
                                          U64 *out_value)
{
    U64 cur;
    U8 b0;
    U8 b1;
    U8 b2;
    U8 b3;
    U8 b4;
    U8 b5;
    U8 b6;
    U8 b7;
    I32 err;

    if (!buf || !cursor || !out_value)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Reuse the canonical checked 1-byte reader so all scalar-width readers
    // inherit identical cursor/bounds behavior and overflow surfaces.
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &b0);
    if (err != GGUF_META_TABLE_OK)
        return err;
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &b1);
    if (err != GGUF_META_TABLE_OK)
        return err;
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &b2);
    if (err != GGUF_META_TABLE_OK)
        return err;
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &b3);
    if (err != GGUF_META_TABLE_OK)
        return err;
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &b4);
    if (err != GGUF_META_TABLE_OK)
        return err;
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &b5);
    if (err != GGUF_META_TABLE_OK)
        return err;
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &b6);
    if (err != GGUF_META_TABLE_OK)
        return err;
    err = GGUFMetadataReadU8Checked(buf, buf_nbytes, &cur, table_end, &b7);
    if (err != GGUF_META_TABLE_OK)
        return err;

    *out_value = ((U64)b0)
               | (((U64)b1) << 8)
               | (((U64)b2) << 16)
               | (((U64)b3) << 24)
               | (((U64)b4) << 32)
               | (((U64)b5) << 40)
               | (((U64)b6) << 48)
               | (((U64)b7) << 56);
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked little-endian I32 metadata reader.
// Contract: writes `*out_value` and advances `*cursor` only on success.
I32 GGUFMetadataReadI32LEChecked(U8 *buf,
                                 U64 buf_nbytes,
                                 U64 *cursor,
                                 U64 table_end,
                                 I32 *out_value)
{
    U64 cur;
    U32 raw_u32;
    I32 err;

    if (!buf || !cursor || !out_value)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Decode through the shared U32 checked reader so signed-I32 decode keeps
    // identical cursor/bounds/no-partial-write behavior.
    err = GGUFMetadataReadU32LEChecked(buf,
                                       buf_nbytes,
                                       &cur,
                                       table_end,
                                       &raw_u32);
    if (err != GGUF_META_TABLE_OK)
        return err;

    // Two's-complement reinterpretation is a plain cast in HolyC.
    *out_value = (I32)raw_u32;
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked little-endian I64 metadata reader.
// Contract: writes `*out_value` and advances `*cursor` only on success.
I32 GGUFMetadataReadI64LEChecked(U8 *buf,
                                 U64 buf_nbytes,
                                 U64 *cursor,
                                 U64 table_end,
                                 I64 *out_value)
{
    // Keep legacy symbol as the canonical parser entrypoint while routing
    // decode through the explicit no-partial-write helper.
    return GGUFMetadataReadI64LECheckedNoPartial(buf,
                                                 buf_nbytes,
                                                 cursor,
                                                 table_end,
                                                 out_value);
}

// Checked little-endian I64 metadata reader with explicit all-or-nothing
// staging contract. Cursor/output are committed only after full U64 decode
// succeeds through the shared checked U64 reader.
I32 GGUFMetadataReadI64LECheckedNoPartial(U8 *buf,
                                          U64 buf_nbytes,
                                          U64 *cursor,
                                          U64 table_end,
                                          I64 *out_value)
{
    U64 cur;
    U64 raw_u64;
    I32 err;

    if (!buf || !cursor || !out_value)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Keep no-partial-write semantics by decoding through a staged cursor and
    // only committing cursor/output after the shared U64 checked reader passes.
    err = GGUFMetadataReadU64LEChecked(buf,
                                       buf_nbytes,
                                       &cur,
                                       table_end,
                                       &raw_u64);
    if (err != GGUF_META_TABLE_OK)
        return err;

    *out_value = (I64)raw_u64;
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked little-endian FLOAT32 bit-pattern metadata reader.
// Runtime remains integer-only: the 32-bit payload is returned as raw bits
// so callers can audit exact IEEE754 encodings without float math.
// Contract: writes `*out_bits` and advances `*cursor` only on success.
I32 GGUFMetadataReadF32BitsLEChecked(U8 *buf,
                                     U64 buf_nbytes,
                                     U64 *cursor,
                                     U64 table_end,
                                     U32 *out_bits)
{
    U64 cur;
    U32 raw_bits;
    I32 err;

    if (!buf || !cursor || !out_bits)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Decode through the shared U32 checked reader so float32 payload bit
    // extraction inherits identical cursor/bounds/no-partial-write behavior.
    err = GGUFMetadataReadU32LEChecked(buf,
                                       buf_nbytes,
                                       &cur,
                                       table_end,
                                       &raw_bits);
    if (err != GGUF_META_TABLE_OK)
        return err;

    *out_bits = raw_bits;
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked little-endian FLOAT64 bit-pattern metadata reader.
// Runtime remains integer-only: the 64-bit payload is returned as raw bits
// so callers can audit exact IEEE754 encodings without float math.
// Contract: writes `*out_bits` and advances `*cursor` only on success.
I32 GGUFMetadataReadF64BitsLEChecked(U8 *buf,
                                     U64 buf_nbytes,
                                     U64 *cursor,
                                     U64 table_end,
                                     U64 *out_bits)
{
    U64 cur;
    U64 raw_bits;
    I32 err;

    if (!buf || !cursor || !out_bits)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Decode through the shared U64 checked reader so float64 payload bit
    // extraction inherits identical cursor/bounds/no-partial-write behavior.
    err = GGUFMetadataReadU64LEChecked(buf,
                                       buf_nbytes,
                                       &cur,
                                       table_end,
                                       &raw_bits);
    if (err != GGUF_META_TABLE_OK)
        return err;

    *out_bits = raw_bits;
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked metadata BOOL reader (`U8 in {0,1}`).
// Contract: writes `*out_value` and advances `*cursor` only on success.
I32 GGUFMetadataReadBoolChecked(U8 *buf,
                                U64 buf_nbytes,
                                U64 *cursor,
                                U64 table_end,
                                Bool *out_value)
{
    U64 cur;
    U8 raw_u8;
    I32 err;

    if (!buf || !cursor || !out_value)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Decode through the shared checked byte reader so bool decode inherits
    // canonical cursor/bounds/no-partial-write behavior.
    err = GGUFMetadataReadU8Checked(buf,
                                    buf_nbytes,
                                    &cur,
                                    table_end,
                                    &raw_u8);
    if (err != GGUF_META_TABLE_OK)
        return err;

    // GGUF bool payloads are strict 0/1 in one byte.
    if (raw_u8 > 1)
        return GGUF_META_TABLE_ERR_BAD_PARAM;

    *out_value = (raw_u8 != 0);
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked metadata BOOL pair reader (`U8 in {0,1}` for each lane).
// Contract: decodes two sequential BOOL payloads through shared checked bool
// reader staging and commits both outputs + cursor only after both lanes pass
// canonical validation (strict no-partial-write tuple semantics).
I32 GGUFMetadataReadBoolPairChecked(U8 *buf,
                                    U64 buf_nbytes,
                                    U64 *cursor,
                                    U64 table_end,
                                    Bool *out_first,
                                    Bool *out_second)
{
    U64 cur;
    Bool first;
    Bool second;
    I32 err;

    if (!buf || !cursor || !out_first || !out_second)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Stage both lane decodes through the canonical BOOL reader so pair decode
    // preserves identical bounds/canonical-value/no-partial-write semantics.
    err = GGUFMetadataReadBoolChecked(buf,
                                      buf_nbytes,
                                      &cur,
                                      table_end,
                                      &first);
    if (err != GGUF_META_TABLE_OK)
        return err;

    err = GGUFMetadataReadBoolChecked(buf,
                                      buf_nbytes,
                                      &cur,
                                      table_end,
                                      &second);
    if (err != GGUF_META_TABLE_OK)
        return err;

    *out_first = first;
    *out_second = second;
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked GGUF string header reader (`U64 len`) plus payload-window bounds
// validation. This helper stages cursor advancement to payload start so string
// payload readers can compose strict no-partial-write decode flows.
// Contract: writes `*out_len`, `*out_payload_end`, and advances `*cursor`
// to payload start only on success.
I32 GGUFMetadataReadStringU64HeaderAndBoundsChecked(U8 *buf,
                                                    U64 buf_nbytes,
                                                    U64 *cursor,
                                                    U64 table_end,
                                                    U64 *out_len,
                                                    U64 *out_payload_end)
{
    U64 cur;
    U64 str_len;
    U64 payload_end;
    I32 err;

    if (!buf || !cursor || !out_len || !out_payload_end)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Decode through the shared U64 reader so the length-prefix path keeps
    // identical checked bounds and no-partial-write semantics.
    err = GGUFMetadataReadU64LEChecked(buf,
                                       buf_nbytes,
                                       &cur,
                                       table_end,
                                       &str_len);
    if (err != GGUF_META_TABLE_OK)
        return err;

    // Validate `payload_start + str_len` against metadata table bounds.
    err = GGUFMetadataCursorCanAdvanceChecked(cur,
                                              str_len,
                                              table_end,
                                              &payload_end);
    if (err != GGUF_META_TABLE_OK)
        return err;

    // Metadata table span must also fit inside the backing input buffer.
    if (payload_end > buf_nbytes)
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS;

    if (str_len > GGUF_MAX_STRING_BYTES)
        return GGUF_META_TABLE_ERR_BAD_PARAM;

    *out_len = str_len;
    *out_payload_end = payload_end;
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked GGUF string-length reader (`U64 len` prefix). Decodes the length
// field and validates that the subsequent payload byte span stays inside the
// metadata table and backing buffer before any payload copy is attempted.
// Contract: writes `*out_len` and advances `*cursor` to payload start only on
// success; caller performs payload read/alloc separately.
I32 GGUFMetadataReadStringLenU64Checked(U8 *buf,
                                        U64 buf_nbytes,
                                        U64 *cursor,
                                        U64 table_end,
                                        U64 *out_len)
{
    U64 cur;
    U64 str_len;
    U64 payload_end;
    I32 err;

    if (!buf || !cursor || !out_len)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Route through the shared string U64-header + bounds helper so all string
    // prefix decoders keep one canonical checked cursor/bounds contract.
    err = GGUFMetadataReadStringU64HeaderAndBoundsChecked(buf,
                                                          buf_nbytes,
                                                          &cur,
                                                          table_end,
                                                          &str_len,
                                                          &payload_end);
    if (err != GGUF_META_TABLE_OK)
        return err;

    *out_len = str_len;
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked metadata string payload copier.
// Contract: copies exactly `str_len` bytes, advances `*cursor` by `str_len`, and
// writes no output bytes on any error path.
I32 GGUFMetadataReadStringBytesChecked(U8 *buf,
                                       U64 buf_nbytes,
                                       U64 *cursor,
                                       U64 table_end,
                                       U64 str_len,
                                       U8 *out_bytes,
                                       U64 out_nbytes)
{
    U64 cur;
    U64 next_cursor;
    U64 i;
    I32 err;

    if (!buf || !cursor || !out_bytes)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    // Keep all read-path bounds and cursor math in positive I64 space.
    if (buf_nbytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_META_TABLE_ERR_OVERFLOW;

    // String payload widths must stay within parser policy bounds.
    if (str_len > GGUF_MAX_STRING_BYTES)
        return GGUF_META_TABLE_ERR_BAD_PARAM;

    // Caller destination must be large enough for the checked payload width.
    if (out_nbytes < str_len)
        return GGUF_META_TABLE_ERR_BAD_PARAM;

    cur = *cursor;

    // Validate source window against table bounds via canonical helper.
    err = GGUFMetadataCursorCanAdvanceChecked(cur,
                                              str_len,
                                              table_end,
                                              &next_cursor);
    if (err != GGUF_META_TABLE_OK)
        return err;

    // Backing buffer must also contain the validated source window.
    if (next_cursor > buf_nbytes)
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS;

    // All checks passed: safe to copy and then commit cursor.
    for (i = 0; i < str_len; ++i)
        out_bytes[i] = buf[cur + i];

    *cursor = next_cursor;
    return GGUF_META_TABLE_OK;
}

// Checked GGUF metadata string reader (`len:U64` + payload bytes, no alloc).
// Contract: writes `*out_len`, copies payload bytes, and advances `*cursor`
// only when the full length+payload path succeeds.
I32 GGUFMetadataReadStringChecked(U8 *buf,
                                  U64 buf_nbytes,
                                  U64 *cursor,
                                  U64 table_end,
                                  U8 *out_bytes,
                                  U64 out_nbytes,
                                  U64 *out_len)
{
    U64 cur;
    U64 str_len;
    I32 err;

    if (!buf || !cursor || !out_bytes || !out_len)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Stage `len` decode through the shared helper; this commits only on
    // success and leaves `cur` at payload start.
    err = GGUFMetadataReadStringLenU64Checked(buf,
                                              buf_nbytes,
                                              &cur,
                                              table_end,
                                              &str_len);
    if (err != GGUF_META_TABLE_OK)
        return err;

    // Destination capacity is validated before payload copy.
    if (out_nbytes < str_len)
        return GGUF_META_TABLE_ERR_BAD_PARAM;

    // Copy payload bytes through the shared checked copier so this composed
    // reader inherits canonical no-partial-write cursor/bytes semantics.
    err = GGUFMetadataReadStringBytesChecked(buf,
                                             buf_nbytes,
                                             &cur,
                                             table_end,
                                             str_len,
                                             out_bytes,
                                             out_nbytes);
    if (err != GGUF_META_TABLE_OK)
        return err;

    *out_len = str_len;
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

// Checked GGUF metadata ARRAY header reader (`elem_type:U32`, `elem_count:U64`).
// This helper enforces strict element-type/count contracts before payload
// iteration so callers can safely dispatch to scalar/value readers.
// Contract: writes outputs and advances `*cursor` only on success.
I32 GGUFMetadataReadArrayHeaderChecked(U8 *buf,
                                       U64 buf_nbytes,
                                       U64 *cursor,
                                       U64 table_end,
                                       U32 *out_elem_type,
                                       U64 *out_elem_count)
{
    U64 cur;
    U32 elem_type;
    U64 elem_count;
    I32 err;

    if (!buf || !cursor || !out_elem_type || !out_elem_count)
        return GGUF_META_TABLE_ERR_NULL_PTR;

    cur = *cursor;

    // Decode element type via shared checked U32 reader so the cursor/bounds
    // behavior matches all other metadata scalar-width decoders.
    err = GGUFMetadataReadU32LEChecked(buf,
                                       buf_nbytes,
                                       &cur,
                                       table_end,
                                       &elem_type);
    if (err != GGUF_META_TABLE_OK)
        return err;

    // GGUF forbids nested arrays; additionally reject unsupported element tags
    // up front before payload-width dependent iteration begins.
    if (elem_type == GGUF_TYPE_ARRAY)
        return GGUF_META_TABLE_ERR_BAD_PARAM;
    if (!GGUFMetaTypeSupported(elem_type))
        return GGUF_META_TABLE_ERR_BAD_PARAM;

    // Decode element count through shared checked U64 reader so count parsing
    // inherits identical no-partial-write and overflow semantics.
    err = GGUFMetadataReadU64LEChecked(buf,
                                       buf_nbytes,
                                       &cur,
                                       table_end,
                                       &elem_count);
    if (err != GGUF_META_TABLE_OK)
        return err;

    // Keep array fan-out bounded before any payload byte math/iteration.
    if (elem_count > GGUF_MAX_ARRAY_ELEMS)
        return GGUF_META_TABLE_ERR_BAD_PARAM;

    *out_elem_type = elem_type;
    *out_elem_count = elem_count;
    *cursor = cur;
    return GGUF_META_TABLE_OK;
}

Bool GGUFMetaCanRead(U8 *buf, I64 size, I64 cursor, I64 need)
{
    if (!buf)
        return FALSE;
    if (size < 0 || cursor < 0 || need < 0)
        return FALSE;
    if (cursor > size)
        return FALSE;
    if (need > size - cursor)
        return FALSE;
    return TRUE;
}

Bool GGUFMetaReadU8(U8 *buf, I64 size, I64 *cursor, U8 *out)
{
    I64 c;

    if (!cursor || !out)
        return FALSE;

    c = *cursor;
    if (!GGUFMetaCanRead(buf, size, c, 1))
        return FALSE;

    *out = buf[c];
    *cursor = c + 1;
    return TRUE;
}

Bool GGUFMetaReadU16LE(U8 *buf, I64 size, I64 *cursor, U16 *out)
{
    I64 c;

    if (!cursor || !out)
        return FALSE;

    c = *cursor;
    if (!GGUFMetaCanRead(buf, size, c, 2))
        return FALSE;

    *out = (U16)buf[c]
         | ((U16)buf[c + 1] << 8);

    *cursor = c + 2;
    return TRUE;
}

Bool GGUFMetaReadU32LE(U8 *buf, I64 size, I64 *cursor, U32 *out)
{
    I64 c;

    if (!cursor || !out)
        return FALSE;

    c = *cursor;
    if (!GGUFMetaCanRead(buf, size, c, 4))
        return FALSE;

    *out = (U32)buf[c]
         | ((U32)buf[c + 1] << 8)
         | ((U32)buf[c + 2] << 16)
         | ((U32)buf[c + 3] << 24);

    *cursor = c + 4;
    return TRUE;
}

Bool GGUFMetaReadU64LE(U8 *buf, I64 size, I64 *cursor, U64 *out)
{
    I64 c;

    if (!cursor || !out)
        return FALSE;

    c = *cursor;
    if (!GGUFMetaCanRead(buf, size, c, 8))
        return FALSE;

    *out = (U64)buf[c]
         | ((U64)buf[c + 1] << 8)
         | ((U64)buf[c + 2] << 16)
         | ((U64)buf[c + 3] << 24)
         | ((U64)buf[c + 4] << 32)
         | ((U64)buf[c + 5] << 40)
         | ((U64)buf[c + 6] << 48)
         | ((U64)buf[c + 7] << 56);

    *cursor = c + 8;
    return TRUE;
}

Bool GGUFMetaTypeSupported(U32 type)
{
    switch (type)
    {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
        case GGUF_TYPE_BOOL:
        case GGUF_TYPE_STRING:
        case GGUF_TYPE_ARRAY:
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return TRUE;
    }

    return FALSE;
}

U0 GGUFMetaZeroString(GGUFString *s)
{
    if (!s)
        return;
    s->len = 0;
    s->data = NULL;
}

U0 GGUFMetaZeroValue(GGUFMetaValue *v)
{
    if (!v)
        return;

    v->type = 0;
    v->u64 = 0;
    v->i64 = 0;
    v->b = FALSE;
    GGUFMetaZeroString(&v->str);
    v->array_elem_type = 0;
    v->array_len = 0;
    v->array_payload_off = 0;
    v->array_payload_bytes = 0;
}

U0 GGUFMetaZeroKV(GGUFMetadataKV *kv)
{
    if (!kv)
        return;
    GGUFMetaZeroString(&kv->key);
    GGUFMetaZeroValue(&kv->value);
}

U0 GGUFMetaFreeString(GGUFString *s)
{
    if (!s)
        return;

    if (s->data)
        Free(s->data);

    s->data = NULL;
    s->len = 0;
}

I64 GGUFMetaReadString(U8 *buf, I64 size, I64 *cursor, GGUFString *out)
{
    U64 len;
    I64 c;
    U8 *dst;
    I64 i;

    if (!cursor || !out)
        return GGUF_META_PARSE_ERR_NULL_PTR;

    GGUFMetaZeroString(out);

    if (!GGUFMetaReadU64LE(buf, size, cursor, &len))
        return GGUF_META_PARSE_ERR_TRUNCATED;
    if (len > GGUF_MAX_STRING_BYTES)
        return GGUF_META_PARSE_ERR_BAD_STRING_LEN;

    c = *cursor;
    if (!GGUFMetaCanRead(buf, size, c, len))
        return GGUF_META_PARSE_ERR_TRUNCATED;

    dst = MAlloc(len + 1);
    if (!dst)
        return GGUF_META_PARSE_ERR_ALLOC;

    for (i = 0; i < len; ++i)
        dst[i] = buf[c + i];
    dst[len] = 0;

    *cursor = c + len;

    out->len = len;
    out->data = dst;
    return GGUF_META_PARSE_OK;
}

I64 GGUFMetaSkipValueByType(U8 *buf, I64 size, I64 *cursor, U32 type)
{
    U8 u8v;
    U16 u16v;
    U32 u32v;
    U64 u64v;
    U32 elem_type;
    U64 elem_count;
    U64 i;
    I64 err;

    if (!GGUFMetaTypeSupported(type))
        return GGUF_META_PARSE_ERR_BAD_TYPE;

    switch (type)
    {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            if (!GGUFMetaReadU8(buf, size, cursor, &u8v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            if (!GGUFMetaReadU16LE(buf, size, cursor, &u16v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            if (!GGUFMetaReadU32LE(buf, size, cursor, &u32v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            if (!GGUFMetaReadU64LE(buf, size, cursor, &u64v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_STRING:
            {
                GGUFString tmp;

                err = GGUFMetaReadString(buf, size, cursor, &tmp);
                if (err != GGUF_META_PARSE_OK)
                    return err;
                GGUFMetaFreeString(&tmp);
                return GGUF_META_PARSE_OK;
            }

        case GGUF_TYPE_ARRAY:
            if (!GGUFMetaReadU32LE(buf, size, cursor, &elem_type))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            if (elem_type == GGUF_TYPE_ARRAY)
                return GGUF_META_PARSE_ERR_NESTED_ARRAY;
            if (!GGUFMetaTypeSupported(elem_type))
                return GGUF_META_PARSE_ERR_BAD_TYPE;
            if (!GGUFMetaReadU64LE(buf, size, cursor, &elem_count))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            if (elem_count > GGUF_MAX_ARRAY_ELEMS)
                return GGUF_META_PARSE_ERR_BAD_ARRAY_LEN;

            for (i = 0; i < elem_count; ++i)
            {
                err = GGUFMetaSkipValueByType(buf, size, cursor, elem_type);
                if (err != GGUF_META_PARSE_OK)
                    return err;
            }

            return GGUF_META_PARSE_OK;
    }

    return GGUF_META_PARSE_ERR_BAD_TYPE;
}

I64 GGUFMetaParseScalar(U8 *buf, I64 size, I64 *cursor, U32 type, GGUFMetaValue *out)
{
    U8 u8v;
    U16 u16v;
    U32 u32v;
    U64 u64v;
    I64 err;

    if (!out)
        return GGUF_META_PARSE_ERR_NULL_PTR;

    if (!GGUFMetaTypeSupported(type))
        return GGUF_META_PARSE_ERR_BAD_TYPE;

    out->type = type;

    switch (type)
    {
        case GGUF_TYPE_UINT8:
            if (!GGUFMetaReadU8(buf, size, cursor, &u8v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            out->u64 = u8v;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_INT8:
            if (!GGUFMetaReadU8(buf, size, cursor, &u8v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            out->i64 = (I8)u8v;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_UINT16:
            if (!GGUFMetaReadU16LE(buf, size, cursor, &u16v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            out->u64 = u16v;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_INT16:
            if (!GGUFMetaReadU16LE(buf, size, cursor, &u16v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            out->i64 = (I16)u16v;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_UINT32:
            if (!GGUFMetaReadU32LE(buf, size, cursor, &u32v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            out->u64 = u32v;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_INT32:
            if (!GGUFMetaReadU32LE(buf, size, cursor, &u32v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            out->i64 = (I32)u32v;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_UINT64:
            if (!GGUFMetaReadU64LE(buf, size, cursor, &u64v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            out->u64 = u64v;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_INT64:
            if (!GGUFMetaReadU64LE(buf, size, cursor, &u64v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            out->i64 = (I64)u64v;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_FLOAT32:
            if (!GGUFMetaReadU32LE(buf, size, cursor, &u32v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            out->u64 = u32v;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_FLOAT64:
            if (!GGUFMetaReadU64LE(buf, size, cursor, &u64v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            out->u64 = u64v;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_BOOL:
            if (!GGUFMetaReadU8(buf, size, cursor, &u8v))
                return GGUF_META_PARSE_ERR_TRUNCATED;
            out->b = (u8v != 0);
            out->u64 = out->b;
            return GGUF_META_PARSE_OK;

        case GGUF_TYPE_STRING:
            err = GGUFMetaReadString(buf, size, cursor, &out->str);
            if (err != GGUF_META_PARSE_OK)
                return err;
            return GGUF_META_PARSE_OK;
    }

    return GGUF_META_PARSE_ERR_BAD_TYPE;
}

I64 GGUFMetaParseValue(U8 *buf, I64 size, I64 *cursor, GGUFMetaValue *out)
{
    U32 type;
    U32 elem_type;
    U64 elem_count;
    I64 payload_begin;
    I64 err;
    U64 i;

    if (!cursor || !out)
        return GGUF_META_PARSE_ERR_NULL_PTR;

    if (!GGUFMetaReadU32LE(buf, size, cursor, &type))
        return GGUF_META_PARSE_ERR_TRUNCATED;
    if (!GGUFMetaTypeSupported(type))
        return GGUF_META_PARSE_ERR_BAD_TYPE;

    GGUFMetaZeroValue(out);
    out->type = type;

    if (type != GGUF_TYPE_ARRAY)
        return GGUFMetaParseScalar(buf, size, cursor, type, out);

    if (!GGUFMetaReadU32LE(buf, size, cursor, &elem_type))
        return GGUF_META_PARSE_ERR_TRUNCATED;
    if (elem_type == GGUF_TYPE_ARRAY)
        return GGUF_META_PARSE_ERR_NESTED_ARRAY;
    if (!GGUFMetaTypeSupported(elem_type))
        return GGUF_META_PARSE_ERR_BAD_TYPE;

    if (!GGUFMetaReadU64LE(buf, size, cursor, &elem_count))
        return GGUF_META_PARSE_ERR_TRUNCATED;
    if (elem_count > GGUF_MAX_ARRAY_ELEMS)
        return GGUF_META_PARSE_ERR_BAD_ARRAY_LEN;

    payload_begin = *cursor;
    for (i = 0; i < elem_count; ++i)
    {
        err = GGUFMetaSkipValueByType(buf, size, cursor, elem_type);
        if (err != GGUF_META_PARSE_OK)
            return err;
    }

    out->array_elem_type = elem_type;
    out->array_len = elem_count;
    out->array_payload_off = payload_begin;
    out->array_payload_bytes = *cursor - payload_begin;
    return GGUF_META_PARSE_OK;
}

U0 GGUFFreeMetadataTable(GGUFMetadataTable *table)
{
    U64 i;

    if (!table)
        return;

    if (!table->items)
    {
        table->count = 0;
        return;
    }

    for (i = 0; i < table->count; ++i)
    {
        GGUFMetaFreeString(&table->items[i].key);
        if (table->items[i].value.type == GGUF_TYPE_STRING)
            GGUFMetaFreeString(&table->items[i].value.str);
    }

    Free(table->items);
    table->items = NULL;
    table->count = 0;
}

I64 GGUFParseMetadata(U8 *buf,
                      I64 size,
                      U64 metadata_kv_count,
                      GGUFMetadataTable *out,
                      I64 *cursor)
{
    GGUFMetadataKV *items;
    U64 i;
    I64 err;

    if (!buf || !out || !cursor)
        return GGUF_META_PARSE_ERR_NULL_PTR;
    if (metadata_kv_count > GGUF_MAX_METADATA_COUNT)
        return GGUF_META_PARSE_ERR_BAD_COUNT;

    out->count = 0;
    out->items = NULL;

    if (!metadata_kv_count)
        return GGUF_META_PARSE_OK;

    items = MAlloc(sizeof(GGUFMetadataKV) * metadata_kv_count);
    if (!items)
        return GGUF_META_PARSE_ERR_ALLOC;

    for (i = 0; i < metadata_kv_count; ++i)
        GGUFMetaZeroKV(&items[i]);

    out->items = items;

    for (i = 0; i < metadata_kv_count; ++i)
    {
        err = GGUFMetaReadString(buf, size, cursor, &items[i].key);
        if (err != GGUF_META_PARSE_OK)
        {
            out->count = metadata_kv_count;
            GGUFFreeMetadataTable(out);
            return err;
        }

        err = GGUFMetaParseValue(buf, size, cursor, &items[i].value);
        if (err != GGUF_META_PARSE_OK)
        {
            out->count = metadata_kv_count;
            GGUFFreeMetadataTable(out);
            return err;
        }
    }

    out->count = metadata_kv_count;
    return GGUF_META_PARSE_OK;
}

U64 GGUFMetaCStrLen(U8 *s)
{
    U64 n;

    if (!s)
        return 0;

    n = 0;
    while (s[n])
        ++n;
    return n;
}

Bool GGUFMetaStringEqCStr(GGUFString *lhs, U8 *rhs)
{
    U64 rhs_len;
    U64 i;

    if (!lhs || !rhs)
        return FALSE;
    if (!lhs->data)
        return FALSE;

    rhs_len = GGUFMetaCStrLen(rhs);
    if (lhs->len != rhs_len)
        return FALSE;

    // Byte-wise compare keeps lookup deterministic and avoids libc helpers.
    for (i = 0; i < lhs->len; ++i)
    {
        if (lhs->data[i] != rhs[i])
            return FALSE;
    }

    return TRUE;
}

GGUFMetadataKV *GGUFMetaFindByKey(GGUFMetadataTable *table, U8 *key)
{
    U64 i;

    if (!table || !key)
        return NULL;
    if (!table->items)
        return NULL;

    // GGUF metadata order is stable and small enough for a linear scan.
    for (i = 0; i < table->count; ++i)
    {
        if (GGUFMetaStringEqCStr(&table->items[i].key, key))
            return &table->items[i];
    }

    return NULL;
}

I64 GGUFMetaGetU64ByKey(GGUFMetadataTable *table, U8 *key, U64 *out)
{
    GGUFMetadataKV *kv;

    if (!table || !key || !out)
        return GGUF_META_PARSE_ERR_NULL_PTR;

    kv = GGUFMetaFindByKey(table, key);
    if (!kv)
        return GGUF_META_PARSE_ERR_NOT_FOUND;

    switch (kv->value.type)
    {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_UINT64:
            *out = kv->value.u64;
            return GGUF_META_PARSE_OK;
    }

    return GGUF_META_PARSE_ERR_TYPE_MISMATCH;
}

I64 GGUFMetaGetI64ByKey(GGUFMetadataTable *table, U8 *key, I64 *out)
{
    GGUFMetadataKV *kv;

    if (!table || !key || !out)
        return GGUF_META_PARSE_ERR_NULL_PTR;

    kv = GGUFMetaFindByKey(table, key);
    if (!kv)
        return GGUF_META_PARSE_ERR_NOT_FOUND;

    switch (kv->value.type)
    {
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_INT16:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_INT64:
            *out = kv->value.i64;
            return GGUF_META_PARSE_OK;
    }

    return GGUF_META_PARSE_ERR_TYPE_MISMATCH;
}

I64 GGUFMetaGetBoolByKey(GGUFMetadataTable *table, U8 *key, Bool *out)
{
    GGUFMetadataKV *kv;

    if (!table || !key || !out)
        return GGUF_META_PARSE_ERR_NULL_PTR;

    kv = GGUFMetaFindByKey(table, key);
    if (!kv)
        return GGUF_META_PARSE_ERR_NOT_FOUND;
    if (kv->value.type != GGUF_TYPE_BOOL)
        return GGUF_META_PARSE_ERR_TYPE_MISMATCH;

    *out = kv->value.b;
    return GGUF_META_PARSE_OK;
}

I64 GGUFMetaGetStringByKey(GGUFMetadataTable *table, U8 *key, GGUFString **out)
{
    GGUFMetadataKV *kv;

    if (!table || !key || !out)
        return GGUF_META_PARSE_ERR_NULL_PTR;

    kv = GGUFMetaFindByKey(table, key);
    if (!kv)
        return GGUF_META_PARSE_ERR_NOT_FOUND;
    if (kv->value.type != GGUF_TYPE_STRING)
        return GGUF_META_PARSE_ERR_TYPE_MISMATCH;

    *out = &kv->value.str;
    return GGUF_META_PARSE_OK;
}

I64 GGUFMetaGetF32BitsByKey(GGUFMetadataTable *table, U8 *key, U32 *out_bits)
{
    GGUFMetadataKV *kv;

    if (!table || !key || !out_bits)
        return GGUF_META_PARSE_ERR_NULL_PTR;

    kv = GGUFMetaFindByKey(table, key);
    if (!kv)
        return GGUF_META_PARSE_ERR_NOT_FOUND;
    if (kv->value.type != GGUF_TYPE_FLOAT32)
        return GGUF_META_PARSE_ERR_TYPE_MISMATCH;

    *out_bits = kv->value.u64;
    return GGUF_META_PARSE_OK;
}

I64 GGUFMetaGetF64BitsByKey(GGUFMetadataTable *table, U8 *key, U64 *out_bits)
{
    GGUFMetadataKV *kv;

    if (!table || !key || !out_bits)
        return GGUF_META_PARSE_ERR_NULL_PTR;

    kv = GGUFMetaFindByKey(table, key);
    if (!kv)
        return GGUF_META_PARSE_ERR_NOT_FOUND;
    if (kv->value.type != GGUF_TYPE_FLOAT64)
        return GGUF_META_PARSE_ERR_TYPE_MISMATCH;

    *out_bits = kv->value.u64;
    return GGUF_META_PARSE_OK;
}
