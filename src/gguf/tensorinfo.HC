// tensorinfo.HC
// GGUF tensor info table parser for HolyC inference runtime.

#define GGUF_TENSOR_PARSE_OK 0
#define GGUF_TENSOR_PARSE_ERR_NULL_PTR 1
#define GGUF_TENSOR_PARSE_ERR_TRUNCATED 2
#define GGUF_TENSOR_PARSE_ERR_BAD_COUNT 3
#define GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN 4
#define GGUF_TENSOR_PARSE_ERR_BAD_DIMS 5
#define GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE 6
#define GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW 7
#define GGUF_TENSOR_PARSE_ERR_BAD_TYPE 8
#define GGUF_TENSOR_PARSE_ERR_ALLOC 9

#define GGUF_TENSOR_MAX_COUNT     (1 << 20)
#define GGUF_TENSOR_MAX_NAME_BYTES (1 << 20)
#define GGUF_TENSOR_MAX_DIMS      8

// Common ggml tensor types used in GGUF today. Parser is strict so bad type tags
// are rejected early and never reach tensor byte-loader math.
#define GGML_TYPE_F32   0
#define GGML_TYPE_F16   1
#define GGML_TYPE_Q4_0  2
#define GGML_TYPE_Q4_1  3
#define GGML_TYPE_Q5_0  6
#define GGML_TYPE_Q5_1  7
#define GGML_TYPE_Q8_0  8
#define GGML_TYPE_Q8_1  9
#define GGML_TYPE_Q2_K  10
#define GGML_TYPE_Q3_K  11
#define GGML_TYPE_Q4_K  12
#define GGML_TYPE_Q5_K  13
#define GGML_TYPE_Q6_K  14
#define GGML_TYPE_Q8_K  15
#define GGML_TYPE_IQ2_XXS 16
#define GGML_TYPE_IQ2_XS  17
#define GGML_TYPE_IQ3_XXS 18
#define GGML_TYPE_IQ1_S   19
#define GGML_TYPE_IQ4_NL  20
#define GGML_TYPE_IQ3_S   21
#define GGML_TYPE_IQ2_S   22
#define GGML_TYPE_IQ4_XS  23
#define GGML_TYPE_I8    24
#define GGML_TYPE_I16   25
#define GGML_TYPE_I32   26
#define GGML_TYPE_I64   27
#define GGML_TYPE_F64   28
#define GGML_TYPE_IQ1_M 29
#define GGML_TYPE_BF16  30
#define GGML_TYPE_Q4_0_4_4 31
#define GGML_TYPE_Q4_0_4_8 32
#define GGML_TYPE_Q4_0_8_8 33
#define GGML_TYPE_TQ1_0 34
#define GGML_TYPE_TQ2_0 35

class GGUFTensorString
{
    U64 len;
    U8 *data;
};

class GGUFTensorInfo
{
    GGUFTensorString name;

    U32 n_dims;
    U64 dims[GGUF_TENSOR_MAX_DIMS];
    U64 n_elements;

    U32 ggml_type;
    U64 offset;
};

class GGUFTensorTable
{
    U64 count;
    GGUFTensorInfo *items;
};

Bool GGUFTensorCanRead(U8 *buf, I64 size, I64 cursor, I64 need)
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

Bool GGUFTensorReadU32LE(U8 *buf, I64 size, I64 *cursor, U32 *out)
{
    I64 c;

    if (!cursor || !out)
        return FALSE;

    c = *cursor;
    if (!GGUFTensorCanRead(buf, size, c, 4))
        return FALSE;

    *out = (U32)buf[c]
         | ((U32)buf[c + 1] << 8)
         | ((U32)buf[c + 2] << 16)
         | ((U32)buf[c + 3] << 24);

    *cursor = c + 4;
    return TRUE;
}

Bool GGUFTensorReadU64LE(U8 *buf, I64 size, I64 *cursor, U64 *out)
{
    I64 c;

    if (!cursor || !out)
        return FALSE;

    c = *cursor;
    if (!GGUFTensorCanRead(buf, size, c, 8))
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

Bool GGUFTensorTypeKnown(U32 t)
{
    switch (t)
    {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_I8:
        case GGML_TYPE_I16:
        case GGML_TYPE_I32:
        case GGML_TYPE_I64:
        case GGML_TYPE_F64:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0_4_4:
        case GGML_TYPE_Q4_0_4_8:
        case GGML_TYPE_Q4_0_8_8:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            return TRUE;
    }

    return FALSE;
}

Bool GGUFTensorTryAddU64(U64 a, U64 b, U64 *out)
{
    if (!out)
        return FALSE;
    if (a > 0xFFFFFFFFFFFFFFFF - b)
        return FALSE;

    *out = a + b;
    return TRUE;
}

Bool GGUFTensorTryMulU64(U64 a, U64 b, U64 *out)
{
    if (!out)
        return FALSE;
    if (a && b > 0xFFFFFFFFFFFFFFFF / a)
        return FALSE;

    *out = a * b;
    return TRUE;
}

U0 GGUFTensorZeroString(GGUFTensorString *s)
{
    if (!s)
        return;
    s->len = 0;
    s->data = NULL;
}

U0 GGUFTensorZeroInfo(GGUFTensorInfo *ti)
{
    I64 i;

    if (!ti)
        return;

    GGUFTensorZeroString(&ti->name);
    ti->n_dims = 0;
    for (i = 0; i < GGUF_TENSOR_MAX_DIMS; ++i)
        ti->dims[i] = 0;
    ti->n_elements = 0;
    ti->ggml_type = 0;
    ti->offset = 0;
}

U0 GGUFTensorFreeString(GGUFTensorString *s)
{
    if (!s)
        return;

    if (s->data)
        Free(s->data);

    s->data = NULL;
    s->len = 0;
}

U0 GGUFFreeTensorTable(GGUFTensorTable *table)
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
        GGUFTensorFreeString(&table->items[i].name);

    Free(table->items);
    table->items = NULL;
    table->count = 0;
}

I64 GGUFTensorReadString(U8 *buf, I64 size, I64 *cursor, GGUFTensorString *out)
{
    U64 len;
    I64 c;
    U8 *dst;
    I64 i;

    if (!cursor || !out)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    GGUFTensorZeroString(out);

    if (!GGUFTensorReadU64LE(buf, size, cursor, &len))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (len > GGUF_TENSOR_MAX_NAME_BYTES)
        return GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN;

    c = *cursor;
    if (!GGUFTensorCanRead(buf, size, c, len))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    dst = MAlloc(len + 1);
    if (!dst)
        return GGUF_TENSOR_PARSE_ERR_ALLOC;

    for (i = 0; i < len; ++i)
        dst[i] = buf[c + i];
    dst[len] = 0;

    *cursor = c + len;

    out->len = len;
    out->data = dst;
    return GGUF_TENSOR_PARSE_OK;
}

I64 GGUFTensorParseOne(U8 *buf, I64 size, I64 *cursor, GGUFTensorInfo *out)
{
    U32 n_dims;
    U64 dim;
    U64 prod;
    U32 ggml_type;
    U64 offset;
    I64 err;
    U32 i;

    if (!cursor || !out)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    GGUFTensorZeroInfo(out);

    err = GGUFTensorReadString(buf, size, cursor, &out->name);
    if (err != GGUF_TENSOR_PARSE_OK)
        return err;

    if (!GGUFTensorReadU32LE(buf, size, cursor, &n_dims))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!n_dims || n_dims > GGUF_TENSOR_MAX_DIMS)
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS;

    // The tensor element count is the exact product of dims. Keep this explicit
    // and overflow-checked so later offset math is auditable and safe.
    prod = 1;
    for (i = 0; i < n_dims; ++i)
    {
        if (!GGUFTensorReadU64LE(buf, size, cursor, &dim))
            return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
        if (!dim)
            return GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE;
        if (prod > 0xFFFFFFFFFFFFFFFF / dim)
            return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;

        out->dims[i] = dim;
        prod *= dim;
    }

    if (!GGUFTensorReadU32LE(buf, size, cursor, &ggml_type))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTypeKnown(ggml_type))
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE;

    if (!GGUFTensorReadU64LE(buf, size, cursor, &offset))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    out->n_dims = n_dims;
    out->n_elements = prod;
    out->ggml_type = ggml_type;
    out->offset = offset;

    return GGUF_TENSOR_PARSE_OK;
}

// Checked no-partial parser for one GGUF tensor-info tuple.
// Parses {name, n_dims, dims[], ggml_type, offset} using U64 cursor/span math
// and only publishes to caller outputs after full tuple validation succeeds.
I64 GGUFTensorInfoReadCheckedNoPartial(U8 *buf,
                                       U64 size,
                                       U64 cursor,
                                       GGUFTensorInfo *out,
                                       U64 *out_next_cursor)
{
    GGUFTensorInfo staged;
    I64 staged_cursor;
    I64 status;
    U64 checked_end;
    U64 name_end;
    U64 dims_bytes;
    U64 dims_end;
    U64 type_end;
    U64 offset_end;
    U64 i;

    if (!buf || !out || !out_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    GGUFTensorZeroInfo(&staged);

    staged_cursor = cursor;
    status = GGUFTensorParseOne(buf,
                                (I64)size,
                                &staged_cursor,
                                &staged);
    if (status != GGUF_TENSOR_PARSE_OK)
    {
        GGUFTensorFreeString(&staged.name);
        return status;
    }

    if (!GGUFTensorTryAddU64(cursor, 8, &checked_end))
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else if (!GGUFTensorTryAddU64(checked_end, staged.name.len, &name_end))
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else if (!GGUFTensorTryAddU64(name_end, 4, &checked_end))
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else if (!GGUFTensorTryMulU64(staged.n_dims, 8, &dims_bytes))
        status = GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;
    else if (!GGUFTensorTryAddU64(checked_end, dims_bytes, &dims_end))
        status = GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;
    else if (!GGUFTensorTryAddU64(dims_end, 4, &type_end))
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else if (!GGUFTensorTryAddU64(type_end, 8, &offset_end))
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else if (offset_end > size)
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else if ((U64)staged_cursor != offset_end)
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else
        status = GGUF_TENSOR_PARSE_OK;

    if (status != GGUF_TENSOR_PARSE_OK)
    {
        GGUFTensorFreeString(&staged.name);
        return status;
    }

    for (i = staged.n_dims; i < GGUF_TENSOR_MAX_DIMS; ++i)
        staged.dims[i] = 0;

    *out = staged;
    *out_next_cursor = staged_cursor;
    return GGUF_TENSOR_PARSE_OK;
}

I64 GGUFParseTensorInfo(U8 *buf,
                        I64 size,
                        U64 tensor_count,
                        GGUFTensorTable *out,
                        I64 *cursor)
{
    GGUFTensorInfo *items;
    U64 i;
    I64 err;
    U64 next_cursor;

    if (!buf || !out || !cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (!tensor_count || tensor_count > GGUF_TENSOR_MAX_COUNT)
        return GGUF_TENSOR_PARSE_ERR_BAD_COUNT;

    out->count = 0;
    out->items = NULL;

    items = MAlloc(sizeof(GGUFTensorInfo) * tensor_count);
    if (!items)
        return GGUF_TENSOR_PARSE_ERR_ALLOC;

    for (i = 0; i < tensor_count; ++i)
        GGUFTensorZeroInfo(&items[i]);

    out->items = items;

    next_cursor = *cursor;
    for (i = 0; i < tensor_count; ++i)
    {
        err = GGUFTensorInfoReadCheckedNoPartial(buf,
                                                 size,
                                                 next_cursor,
                                                 &items[i],
                                                 &next_cursor);
        if (err != GGUF_TENSOR_PARSE_OK)
        {
            out->count = tensor_count;
            GGUFFreeTensorTable(out);
            return err;
        }
    }

    *cursor = next_cursor;

    out->count = tensor_count;
    return GGUF_TENSOR_PARSE_OK;
}
