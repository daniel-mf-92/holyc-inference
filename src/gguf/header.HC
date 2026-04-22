// header.HC
// GGUF header parser for HolyC inference runtime.

#define GGUF_MAGIC_U32 0x46554747
#define GGUF_VERSION_V2 2
#define GGUF_VERSION_V3 3

// GGUF v2/v3 fixed header layout (little-endian):
//   byte  0..3   : magic "GGUF" (U32)
//   byte  4..7   : version (U32)
//   byte  8..15  : tensor_count (U64)
//   byte 16..23  : metadata_kv_count (U64)
// Keep offsets centralized so all parser entrypoints share one canonical layout.
#define GGUF_HEADER_MAGIC_OFFSET             0
#define GGUF_HEADER_VERSION_OFFSET           4
#define GGUF_HEADER_TENSOR_COUNT_OFFSET      8
#define GGUF_HEADER_METADATA_COUNT_OFFSET   16

#define GGUF_HEADER_MAGIC_BYTES              4
#define GGUF_HEADER_VERSION_BYTES            4
#define GGUF_HEADER_TENSOR_COUNT_BYTES       8
#define GGUF_HEADER_METADATA_COUNT_BYTES     8

#define GGUF_PARSE_OK 0
#define GGUF_PARSE_ERR_NULL_PTR 1
#define GGUF_PARSE_ERR_TRUNCATED 2
#define GGUF_PARSE_ERR_BAD_MAGIC 3
#define GGUF_PARSE_ERR_BAD_VERSION 4
#define GGUF_PARSE_ERR_BAD_COUNT 5

class GGUFHeader
{
    U32 magic;
    U32 version;
    U64 tensor_count;
    U64 metadata_kv_count;
};

#define GGUF_HEADER_OK              0
#define GGUF_HEADER_ERR_NULL_PTR    1
#define GGUF_HEADER_ERR_BAD_PARAM   2
#define GGUF_HEADER_ERR_OVERFLOW    3
#define GGUF_HEADER_ERR_TRUNCATED   4
#define GGUF_HEADER_ERR_BAD_MAGIC   5
#define GGUF_HEADER_ERR_BAD_VERSION 6

#define GGUF_HEADER_BYTES (GGUF_HEADER_MAGIC_BYTES         \
                         + GGUF_HEADER_VERSION_BYTES       \
                         + GGUF_HEADER_TENSOR_COUNT_BYTES  \
                         + GGUF_HEADER_METADATA_COUNT_BYTES)

Bool GGUFHeaderVersionSupported(U32 version)
{
    return (version == GGUF_VERSION_V2 || version == GGUF_VERSION_V3);
}

Bool GGUFHeaderCanReadSpan(U64 total_bytes, U64 offset, U64 need_bytes)
{
    U64 end_offset;

    if (offset > total_bytes)
        return FALSE;
    if (need_bytes > (0xFFFFFFFFFFFFFFFF - offset))
        return FALSE;

    end_offset = offset + need_bytes;
    if (end_offset > total_bytes)
        return FALSE;

    return TRUE;
}

I32 GGUFReadU32LEChecked(U8 *buf, U64 buf_nbytes, U64 offset, U32 *out_value)
{
    if (!buf || !out_value)
        return GGUF_HEADER_ERR_NULL_PTR;
    if (!GGUFHeaderCanReadSpan(buf_nbytes, offset, 4))
        return GGUF_HEADER_ERR_TRUNCATED;

    *out_value = (U32)buf[offset]
               | ((U32)buf[offset + 1] << 8)
               | ((U32)buf[offset + 2] << 16)
               | ((U32)buf[offset + 3] << 24);

    return GGUF_HEADER_OK;
}

I32 GGUFReadU16LEChecked(U8 *buf, U64 buf_nbytes, U64 offset, U16 *out_value)
{
    if (!buf || !out_value)
        return GGUF_HEADER_ERR_NULL_PTR;
    if (!GGUFHeaderCanReadSpan(buf_nbytes, offset, 2))
        return GGUF_HEADER_ERR_TRUNCATED;

    *out_value = (U16)buf[offset]
               | ((U16)buf[offset + 1] << 8);

    return GGUF_HEADER_OK;
}

I32 GGUFReadU64LEChecked(U8 *buf, U64 buf_nbytes, U64 offset, U64 *out_value)
{
    if (!buf || !out_value)
        return GGUF_HEADER_ERR_NULL_PTR;
    if (!GGUFHeaderCanReadSpan(buf_nbytes, offset, 8))
        return GGUF_HEADER_ERR_TRUNCATED;

    *out_value = (U64)buf[offset]
               | ((U64)buf[offset + 1] << 8)
               | ((U64)buf[offset + 2] << 16)
               | ((U64)buf[offset + 3] << 24)
               | ((U64)buf[offset + 4] << 32)
               | ((U64)buf[offset + 5] << 40)
               | ((U64)buf[offset + 6] << 48)
               | ((U64)buf[offset + 7] << 56);

    return GGUF_HEADER_OK;
}

I32 GGUFReadI32LEChecked(U8 *buf, U64 buf_nbytes, U64 offset, I32 *out_value)
{
    U32 raw;
    I32 status;

    if (!buf || !out_value)
        return GGUF_HEADER_ERR_NULL_PTR;

    status = GGUFReadU32LEChecked(buf, buf_nbytes, offset, &raw);
    if (status)
        return status;

    // GGUF stores all integers little-endian. Convert to signed I32 using
    // explicit two's-complement math so behavior stays deterministic.
    if (raw & 0x80000000)
        *out_value = -(I32)((~raw + 1) & 0xFFFFFFFF);
    else
        *out_value = (I32)raw;

    return GGUF_HEADER_OK;
}

I32 GGUFReadI64LEChecked(U8 *buf, U64 buf_nbytes, U64 offset, I64 *out_value)
{
    U64 raw;
    I32 status;

    if (!buf || !out_value)
        return GGUF_HEADER_ERR_NULL_PTR;

    status = GGUFReadU64LEChecked(buf, buf_nbytes, offset, &raw);
    if (status)
        return status;

    // Same explicit two's-complement conversion for I64 domain.
    if (raw & 0x8000000000000000)
        *out_value = -(I64)((~raw + 1) & 0xFFFFFFFFFFFFFFFF);
    else
        *out_value = (I64)raw;

    return GGUF_HEADER_OK;
}

// Validate parsed header fields and return the canonical byte width consumed by
// the fixed GGUF v2/v3 header prefix. Downstream metadata/tensor parsers should
// use this helper as the single source of truth for header-size math.
I32 GGUFHeaderValidateAndSizeChecked(GGUFHeader *header, U64 *out_header_bytes)
{
    if (!header || !out_header_bytes)
        return GGUF_HEADER_ERR_NULL_PTR;

    if (header->magic != GGUF_MAGIC_U32)
        return GGUF_HEADER_ERR_BAD_MAGIC;

    if (!GGUFHeaderVersionSupported(header->version))
        return GGUF_HEADER_ERR_BAD_VERSION;

    // HolyC parser loops index with signed offsets in multiple call sites.
    // Reject count fields that cannot be represented in positive I64 space.
    if (header->tensor_count > 0x7FFFFFFFFFFFFFFF)
        return GGUF_HEADER_ERR_OVERFLOW;
    if (header->metadata_kv_count > 0x7FFFFFFFFFFFFFFF)
        return GGUF_HEADER_ERR_OVERFLOW;

    // Guard future header-layout edits that could accidentally exceed the
    // signed index domain expected by downstream offset arithmetic.
    if ((U64)GGUF_HEADER_BYTES > 0x7FFFFFFFFFFFFFFF)
        return GGUF_HEADER_ERR_OVERFLOW;

    *out_header_bytes = GGUF_HEADER_BYTES;
    return GGUF_HEADER_OK;
}

// Checked GGUF header decoder for parser front-ends that consume a fixed
// byte span and need explicit status codes (NULL_PTR/BAD_PARAM/OVERFLOW).
// Decodes fields exactly as little-endian U32/U64 values from bytes[0..23].
I32 GGUFHeaderParseChecked(U8 *header_bytes,
                           U64 header_nbytes,
                           GGUFHeader *out_header,
                           U64 *out_header_bytes)
{
    U32 magic_u32;
    U32 version_u32;
    U64 tensor_count;
    U64 metadata_kv_count;
    I32 status;

    if (!header_bytes || !out_header || !out_header_bytes)
        return GGUF_HEADER_ERR_NULL_PTR;

    // HolyC indexed addressing uses signed offsets; reject spans that cannot
    // be represented in positive I64 index space to avoid signed-wrap hazards.
    if (header_nbytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_HEADER_ERR_OVERFLOW;

    if (header_nbytes < GGUF_HEADER_BYTES)
        return GGUF_HEADER_ERR_TRUNCATED;

    status = GGUFReadU32LEChecked(header_bytes,
                                  header_nbytes,
                                  GGUF_HEADER_MAGIC_OFFSET,
                                  &magic_u32);
    if (status)
        return status;
    if (magic_u32 != GGUF_MAGIC_U32)
        return GGUF_HEADER_ERR_BAD_MAGIC;

    status = GGUFReadU32LEChecked(header_bytes,
                                  header_nbytes,
                                  GGUF_HEADER_VERSION_OFFSET,
                                  &version_u32);
    if (status)
        return status;
    if (!GGUFHeaderVersionSupported(version_u32))
        return GGUF_HEADER_ERR_BAD_VERSION;

    status = GGUFReadU64LEChecked(header_bytes,
                                  header_nbytes,
                                  GGUF_HEADER_TENSOR_COUNT_OFFSET,
                                  &tensor_count);
    if (status)
        return status;

    status = GGUFReadU64LEChecked(header_bytes,
                                  header_nbytes,
                                  GGUF_HEADER_METADATA_COUNT_OFFSET,
                                  &metadata_kv_count);
    if (status)
        return status;

    out_header->magic = magic_u32;
    out_header->version = version_u32;
    out_header->tensor_count = tensor_count;
    out_header->metadata_kv_count = metadata_kv_count;

    return GGUFHeaderValidateAndSizeChecked(out_header, out_header_bytes);
}

// Checked GGUF header reader with caller-owned cursor advancement.
//
// Contract:
// - Decodes one GGUF fixed header at `*cursor` (little-endian v2/v3 only).
// - Advances cursor by canonical consumed byte width on success.
// - Leaves `*cursor` and `out_header` unchanged on failure.
I32 GGUFHeaderReadVersionedChecked(U8 *header_bytes,
                                   I64 header_nbytes,
                                   I64 *cursor,
                                   GGUFHeader *out_header)
{
    I64 snapshot_cursor;
    I64 staged_cursor;
    U64 staged_header_bytes;
    I32 status;
    GGUFHeader staged_header;

    if (!header_bytes || !cursor || !out_header)
        return GGUF_HEADER_ERR_NULL_PTR;

    if (header_nbytes < 0)
        return GGUF_HEADER_ERR_BAD_PARAM;

    snapshot_cursor = *cursor;
    if (snapshot_cursor < 0)
        return GGUF_HEADER_ERR_BAD_PARAM;
    if (snapshot_cursor > header_nbytes)
        return GGUF_HEADER_ERR_TRUNCATED;

    if (!GGUFHeaderCanReadSpan((U64)header_nbytes,
                               (U64)snapshot_cursor,
                               GGUF_HEADER_BYTES))
        return GGUF_HEADER_ERR_TRUNCATED;

    staged_cursor = snapshot_cursor;

    status = GGUFHeaderParseChecked(header_bytes + staged_cursor,
                                    (U64)(header_nbytes - staged_cursor),
                                    &staged_header,
                                    &staged_header_bytes);
    if (status)
        return status;

    if (staged_header_bytes != GGUF_HEADER_BYTES)
        return GGUF_HEADER_ERR_OVERFLOW;
    if (staged_cursor > (0x7FFFFFFFFFFFFFFF - (I64)staged_header_bytes))
        return GGUF_HEADER_ERR_OVERFLOW;

    staged_cursor += (I64)staged_header_bytes;

    out_header->magic = staged_header.magic;
    out_header->version = staged_header.version;
    out_header->tensor_count = staged_header.tensor_count;
    out_header->metadata_kv_count = staged_header.metadata_kv_count;
    *cursor = staged_cursor;

    return GGUF_HEADER_OK;
}

// Checked no-partial GGUF header reader.
//
// Contract:
// - Parses canonical GGUF fixed header fields from immutable byte span.
// - Uses checked little-endian integer reads with strict bound validation.
// - Performs zero writes to out_header/out_header_bytes on any failure path.
// - Commits parsed header + consumed-bytes exactly once on success.
I32 GGUFHeaderReadCheckedNoPartial(U8 *header_bytes,
                                   U64 header_nbytes,
                                   GGUFHeader *out_header,
                                   U64 *out_header_bytes)
{
    U8 *snapshot_header_bytes;
    U64 snapshot_header_nbytes;
    U32 staged_magic;
    U32 staged_version;
    U64 staged_tensor_count;
    U64 staged_metadata_kv_count;
    U64 staged_header_bytes;
    GGUFHeader staged_header;
    I32 status;

    if (!header_bytes || !out_header || !out_header_bytes)
        return GGUF_HEADER_ERR_NULL_PTR;

    if (header_nbytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_HEADER_ERR_OVERFLOW;
    if (header_nbytes < GGUF_HEADER_BYTES)
        return GGUF_HEADER_ERR_TRUNCATED;

    snapshot_header_bytes = header_bytes;
    snapshot_header_nbytes = header_nbytes;

    status = GGUFReadU32LEChecked(header_bytes,
                                  header_nbytes,
                                  GGUF_HEADER_MAGIC_OFFSET,
                                  &staged_magic);
    if (status)
        return status;
    if (staged_magic != GGUF_MAGIC_U32)
        return GGUF_HEADER_ERR_BAD_MAGIC;

    status = GGUFReadU32LEChecked(header_bytes,
                                  header_nbytes,
                                  GGUF_HEADER_VERSION_OFFSET,
                                  &staged_version);
    if (status)
        return status;
    if (!GGUFHeaderVersionSupported(staged_version))
        return GGUF_HEADER_ERR_BAD_VERSION;

    status = GGUFReadU64LEChecked(header_bytes,
                                  header_nbytes,
                                  GGUF_HEADER_TENSOR_COUNT_OFFSET,
                                  &staged_tensor_count);
    if (status)
        return status;

    status = GGUFReadU64LEChecked(header_bytes,
                                  header_nbytes,
                                  GGUF_HEADER_METADATA_COUNT_OFFSET,
                                  &staged_metadata_kv_count);
    if (status)
        return status;

    // Revalidate immutable snapshot before publishing staged outputs.
    if (snapshot_header_bytes != header_bytes)
        return GGUF_HEADER_ERR_BAD_PARAM;
    if (snapshot_header_nbytes != header_nbytes)
        return GGUF_HEADER_ERR_BAD_PARAM;

    staged_header.magic = staged_magic;
    staged_header.version = staged_version;
    staged_header.tensor_count = staged_tensor_count;
    staged_header.metadata_kv_count = staged_metadata_kv_count;

    status = GGUFHeaderValidateAndSizeChecked(&staged_header,
                                              &staged_header_bytes);
    if (status)
        return status;

    if (snapshot_header_bytes != header_bytes)
        return GGUF_HEADER_ERR_BAD_PARAM;
    if (snapshot_header_nbytes != header_nbytes)
        return GGUF_HEADER_ERR_BAD_PARAM;

    out_header->magic = staged_header.magic;
    out_header->version = staged_header.version;
    out_header->tensor_count = staged_header.tensor_count;
    out_header->metadata_kv_count = staged_header.metadata_kv_count;
    *out_header_bytes = staged_header_bytes;
    return GGUF_HEADER_OK;
}

// Version-aware GGUF header reader for stream parsers that track an explicit
// cursor. Decodes v2/v3 header fields from `buf[*cursor..)` and advances
// `*cursor` by exactly GGUF_HEADER_BYTES on success.
// Contract:
// - Input span decode is little-endian and fully bounds checked.
// - `*cursor`, `*out_header`, and `*out_header_bytes` are committed only on
//   full success (no partial writes on failure paths).
I32 GGUFHeaderReadVersionedChecked(U8 *buf,
                                   U64 buf_nbytes,
                                   U64 *cursor,
                                   GGUFHeader *out_header,
                                   U64 *out_header_bytes)
{
    U64 cur;
    U64 next_cursor;
    U64 header_bytes;
    U32 magic_u32;
    U32 version_u32;
    U64 tensor_count;
    U64 metadata_kv_count;
    GGUFHeader staged_header;
    I32 status;

    if (!buf || !cursor || !out_header || !out_header_bytes)
        return GGUF_HEADER_ERR_NULL_PTR;

    // Keep all parser indexing in positive I64 range.
    if (buf_nbytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_HEADER_ERR_OVERFLOW;

    cur = *cursor;
    if (cur > 0x7FFFFFFFFFFFFFFF)
        return GGUF_HEADER_ERR_OVERFLOW;
    if (cur > buf_nbytes)
        return GGUF_HEADER_ERR_BAD_PARAM;

    if (!GGUFHeaderCanReadSpan(buf_nbytes, cur, GGUF_HEADER_BYTES))
        return GGUF_HEADER_ERR_TRUNCATED;

    status = GGUFReadU32LEChecked(buf, buf_nbytes, cur, &magic_u32);
    if (status)
        return status;

    if (cur > (0xFFFFFFFFFFFFFFFF - 4))
        return GGUF_HEADER_ERR_OVERFLOW;
    cur += 4;

    status = GGUFReadU32LEChecked(buf, buf_nbytes, cur, &version_u32);
    if (status)
        return status;

    if (cur > (0xFFFFFFFFFFFFFFFF - 4))
        return GGUF_HEADER_ERR_OVERFLOW;
    cur += 4;

    status = GGUFReadU64LEChecked(buf, buf_nbytes, cur, &tensor_count);
    if (status)
        return status;

    if (cur > (0xFFFFFFFFFFFFFFFF - 8))
        return GGUF_HEADER_ERR_OVERFLOW;
    cur += 8;

    status = GGUFReadU64LEChecked(buf, buf_nbytes, cur, &metadata_kv_count);
    if (status)
        return status;

    if (cur > (0xFFFFFFFFFFFFFFFF - 8))
        return GGUF_HEADER_ERR_OVERFLOW;
    next_cursor = cur + 8;

    staged_header.magic = magic_u32;
    staged_header.version = version_u32;
    staged_header.tensor_count = tensor_count;
    staged_header.metadata_kv_count = metadata_kv_count;

    status = GGUFHeaderValidateAndSizeChecked(&staged_header, &header_bytes);
    if (status)
        return status;

    if (header_bytes != GGUF_HEADER_BYTES)
        return GGUF_HEADER_ERR_BAD_PARAM;
    if (next_cursor > buf_nbytes)
        return GGUF_HEADER_ERR_TRUNCATED;

    out_header->magic = staged_header.magic;
    out_header->version = staged_header.version;
    out_header->tensor_count = staged_header.tensor_count;
    out_header->metadata_kv_count = staged_header.metadata_kv_count;
    *out_header_bytes = header_bytes;
    *cursor = next_cursor;
    return GGUF_HEADER_OK;
}

Bool GGUFCanRead(U8 *buf, I64 size, I64 cursor, I64 need)
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

Bool GGUFReadU32LE(U8 *buf, I64 size, I64 *cursor, U32 *out)
{
    I64 c;

    if (!cursor || !out)
        return FALSE;

    c = *cursor;
    if (!GGUFCanRead(buf, size, c, 4))
        return FALSE;

    *out = (U32)buf[c]
         | ((U32)buf[c + 1] << 8)
         | ((U32)buf[c + 2] << 16)
         | ((U32)buf[c + 3] << 24);

    *cursor = c + 4;
    return TRUE;
}

Bool GGUFReadU64LE(U8 *buf, I64 size, I64 *cursor, U64 *out)
{
    I64 c;

    if (!cursor || !out)
        return FALSE;

    c = *cursor;
    if (!GGUFCanRead(buf, size, c, 8))
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

I64 GGUFParseHeader(U8 *buf, I64 size, GGUFHeader *out, I64 *cursor)
{
    I64 c;
    U32 magic;
    U32 version;
    U64 tensor_count;
    U64 metadata_kv_count;

    if (!buf || !out || !cursor)
        return GGUF_PARSE_ERR_NULL_PTR;

    c = *cursor;

    if (!GGUFCanRead(buf, size, c, GGUF_HEADER_BYTES))
        return GGUF_PARSE_ERR_TRUNCATED;

    if (!GGUFReadU32LE(buf, size, &c, &magic))
        return GGUF_PARSE_ERR_TRUNCATED;
    if (magic != GGUF_MAGIC_U32)
        return GGUF_PARSE_ERR_BAD_MAGIC;

    if (!GGUFReadU32LE(buf, size, &c, &version))
        return GGUF_PARSE_ERR_TRUNCATED;
    if (!GGUFHeaderVersionSupported(version))
        return GGUF_PARSE_ERR_BAD_VERSION;

    if (!GGUFReadU64LE(buf, size, &c, &tensor_count))
        return GGUF_PARSE_ERR_TRUNCATED;
    if (!GGUFReadU64LE(buf, size, &c, &metadata_kv_count))
        return GGUF_PARSE_ERR_TRUNCATED;

    if (!tensor_count)
        return GGUF_PARSE_ERR_BAD_COUNT;

    out->magic = magic;
    out->version = version;
    out->tensor_count = tensor_count;
    out->metadata_kv_count = metadata_kv_count;

    *cursor = c;
    return GGUF_PARSE_OK;
}
