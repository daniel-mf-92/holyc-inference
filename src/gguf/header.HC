// header.HC
// GGUF header parser for HolyC inference runtime.

#define GGUF_MAGIC_U32 0x46554747
#define GGUF_VERSION_V2 2
#define GGUF_VERSION_V3 3

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

    if (!GGUFReadU32LE(buf, size, &c, &magic))
        return GGUF_PARSE_ERR_TRUNCATED;
    if (magic != GGUF_MAGIC_U32)
        return GGUF_PARSE_ERR_BAD_MAGIC;

    if (!GGUFReadU32LE(buf, size, &c, &version))
        return GGUF_PARSE_ERR_TRUNCATED;
    if (version != GGUF_VERSION_V2 && version != GGUF_VERSION_V3)
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
