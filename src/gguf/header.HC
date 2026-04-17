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
