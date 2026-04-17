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
