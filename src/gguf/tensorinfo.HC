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

// Checked no-partial parser for one GGUF tensor name string.
// Reads `{u64 name_len, name_len bytes}` from `cursor` and atomically publishes
// `{name,next_cursor}` only after full span validation succeeds.
I64 GGUFTensorInfoReadNameCheckedNoPartial(U8 *buf,
                                           U64 size,
                                           U64 cursor,
                                           GGUFTensorString *out_name,
                                           U64 *out_next_cursor)
{
    U64 snapshot_size;
    U64 snapshot_cursor;
    GGUFTensorString staged;
    I64 staged_cursor;
    I64 status;
    U64 checked_end;

    if (!buf || !out_name || !out_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_size = size;
    snapshot_cursor = cursor;

    GGUFTensorZeroString(&staged);

    staged_cursor = cursor;
    status = GGUFTensorReadString(buf,
                                  (I64)size,
                                  &staged_cursor,
                                  &staged);
    if (status != GGUF_TENSOR_PARSE_OK)
    {
        GGUFTensorFreeString(&staged);
        return status;
    }

    // Keep cursor arithmetic explicit and overflow-checked in U64 so bounds are
    // auditable before publishing caller-visible outputs.
    if (!GGUFTensorTryAddU64(cursor, 8, &checked_end))
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else if (!GGUFTensorTryAddU64(checked_end, staged.len, &checked_end))
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else if (checked_end > size)
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else if ((U64)staged_cursor != checked_end)
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else
        status = GGUF_TENSOR_PARSE_OK;

    if (status != GGUF_TENSOR_PARSE_OK)
    {
        GGUFTensorFreeString(&staged);
        return status;
    }

    // Commit-only publish guard: keep diagnostics/output publication tied to
    // immutable snapshot values so no-partial contracts stay explicit.
    if (snapshot_size != size || snapshot_cursor != cursor)
    {
        GGUFTensorFreeString(&staged);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }

    *out_name = staged;
    *out_next_cursor = checked_end;
    return GGUF_TENSOR_PARSE_OK;
}

// Commit-only diagnostics wrapper for tensor name parsing.
//
// Composes GGUFTensorInfoReadNameCheckedNoPartial and stages canonical
// diagnostics tuple {name_len, required_bytes, next_cursor}. Caller receives
// diagnostics only; staged name storage is always released before return.
I64 GGUFTensorInfoReadNameCheckedNoPartialCommitOnly(U8 *buf,
                                                     U64 size,
                                                     U64 cursor,
                                                     U64 *out_name_len,
                                                     U64 *out_required_bytes,
                                                     U64 *out_next_cursor)
{
    U8 *snapshot_buf;
    U64 snapshot_size;
    U64 snapshot_cursor;
    GGUFTensorString staged_name;
    U64 staged_name_len;
    U64 staged_required_bytes;
    U64 staged_next_cursor;
    U64 computed_end;
    I64 status;

    if (!buf || !out_name_len || !out_required_bytes || !out_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (out_name_len == out_required_bytes ||
        out_name_len == out_next_cursor ||
        out_required_bytes == out_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    GGUFTensorZeroString(&staged_name);
    staged_name_len = 0;
    staged_required_bytes = 0;
    staged_next_cursor = 0;

    status = GGUFTensorInfoReadNameCheckedNoPartial(buf,
                                                    size,
                                                    cursor,
                                                    &staged_name,
                                                    &staged_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    staged_name_len = staged_name.len;

    if (!GGUFTensorTryAddU64(8, staged_name_len, &staged_required_bytes))
    {
        GGUFTensorFreeString(&staged_name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }
    if (!GGUFTensorTryAddU64(cursor, staged_required_bytes, &computed_end))
    {
        GGUFTensorFreeString(&staged_name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }
    if (computed_end != staged_next_cursor)
    {
        GGUFTensorFreeString(&staged_name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
    {
        GGUFTensorFreeString(&staged_name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }

    GGUFTensorFreeString(&staged_name);

    *out_name_len = staged_name_len;
    *out_required_bytes = staged_required_bytes;
    *out_next_cursor = staged_next_cursor;
    return GGUF_TENSOR_PARSE_OK;
}

// Checked no-partial parser for `{n_dims, dims[], ggml_type, offset}`.
// Uses explicit U64 staging math and publishes the tuple atomically only
// after full span parity and type/dim validation succeed.
I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartial(U8 *buf,
                                                     U64 size,
                                                     U64 cursor,
                                                     U32 *out_n_dims,
                                                     U64 *out_dims,
                                                     U64 *out_n_elements,
                                                     U32 *out_ggml_type,
                                                     U64 *out_offset,
                                                     U64 *out_next_cursor)
{
    U64 snapshot_size;
    U64 snapshot_cursor;
    I64 staged_cursor;
    U32 staged_n_dims;
    U64 staged_dims[GGUF_TENSOR_MAX_DIMS];
    U64 staged_n_elements;
    U32 staged_ggml_type;
    U64 staged_offset;
    I64 status;
    U64 dims_bytes;
    U64 checked_end;
    U64 i;

    if (!buf || !out_n_dims || !out_dims || !out_n_elements || !out_ggml_type || !out_offset || !out_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_cursor = cursor;
    staged_n_dims = 0;
    staged_n_elements = 1;
    staged_ggml_type = 0;
    staged_offset = 0;
    for (i = 0; i < GGUF_TENSOR_MAX_DIMS; ++i)
        staged_dims[i] = 0;

    if (!GGUFTensorReadU32LE(buf, (I64)size, &staged_cursor, &staged_n_dims))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!staged_n_dims || staged_n_dims > GGUF_TENSOR_MAX_DIMS)
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS;

    // Decode each dim explicitly so every element-count multiply is visible
    // and overflow-checked before any caller-visible publication.
    for (i = 0; i < staged_n_dims; ++i)
    {
        if (!GGUFTensorReadU64LE(buf, (I64)size, &staged_cursor, &staged_dims[i]))
            return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
        if (!staged_dims[i])
            return GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE;
        if (staged_n_elements > 0xFFFFFFFFFFFFFFFF / staged_dims[i])
            return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;

        staged_n_elements *= staged_dims[i];
    }

    if (!GGUFTensorReadU32LE(buf, (I64)size, &staged_cursor, &staged_ggml_type))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTypeKnown(staged_ggml_type))
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE;

    if (!GGUFTensorReadU64LE(buf, (I64)size, &staged_cursor, &staged_offset))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor, 4, &checked_end))
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else if (!GGUFTensorTryMulU64(staged_n_dims, 8, &dims_bytes))
        status = GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;
    else if (!GGUFTensorTryAddU64(checked_end, dims_bytes, &checked_end))
        status = GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;
    else if (!GGUFTensorTryAddU64(checked_end, 4, &checked_end))
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else if (!GGUFTensorTryAddU64(checked_end, 8, &checked_end))
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else if (checked_end > size)
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else if ((U64)staged_cursor != checked_end)
        status = GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    else
        status = GGUF_TENSOR_PARSE_OK;

    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_n_dims = staged_n_dims;
    for (i = 0; i < GGUF_TENSOR_MAX_DIMS; ++i)
        out_dims[i] = staged_dims[i];
    *out_n_elements = staged_n_elements;
    *out_ggml_type = staged_ggml_type;
    *out_offset = staged_offset;
    *out_next_cursor = checked_end;

    return GGUF_TENSOR_PARSE_OK;
}

// Commit-only diagnostics wrapper for dims/type/offset tuple decoding.
//
// The helper composes `GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartial` and
// stages canonical diagnostics:
//   {dim_count, dims_cells, required_bytes, type_value, tensor_offset, last_dim_index}
// then atomically publishes only after immutable cursor/span snapshots match.
I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnly(
    U8 *buf,
    U64 size,
    U64 cursor,
    U32 *out_dim_count,
    U64 *out_dims_cells,
    U64 *out_required_bytes,
    U32 *out_type_value,
    U64 *out_tensor_offset,
    U64 *out_last_dim_index,
    U64 *out_next_cursor)
{
    U8 *snapshot_buf;
    U64 snapshot_size;
    U64 snapshot_cursor;
    U32 staged_dim_count;
    U64 staged_dims[GGUF_TENSOR_MAX_DIMS];
    U64 staged_n_elements;
    U32 staged_type_value;
    U64 staged_tensor_offset;
    U64 staged_next_cursor;
    U64 staged_dims_cells;
    U64 staged_required_bytes;
    U64 staged_last_dim_index;
    U64 staged_dims_bytes;
    U64 computed_end;
    I64 status;

    if (!buf || !out_dim_count || !out_dims_cells || !out_required_bytes)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (!out_type_value || !out_tensor_offset || !out_last_dim_index || !out_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_dim_count = 0;
    staged_n_elements = 0;
    staged_type_value = 0;
    staged_tensor_offset = 0;
    staged_next_cursor = 0;
    staged_dims_cells = 0;
    staged_required_bytes = 0;
    staged_last_dim_index = 0;

    status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartial(
        buf,
        size,
        cursor,
        &staged_dim_count,
        staged_dims,
        &staged_n_elements,
        &staged_type_value,
        &staged_tensor_offset,
        &staged_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    staged_dims_cells = staged_dim_count;
    staged_last_dim_index = staged_dim_count - 1;

    if (!GGUFTensorTryMulU64(staged_dims_cells, 8, &staged_dims_bytes))
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;

    // required_bytes = 4 (n_dims) + n_dims*8 (dims) + 4 (type) + 8 (offset)
    if (!GGUFTensorTryAddU64(4, staged_dims_bytes, &staged_required_bytes))
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;
    if (!GGUFTensorTryAddU64(staged_required_bytes, 4, &staged_required_bytes))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTryAddU64(staged_required_bytes, 8, &staged_required_bytes))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor, staged_required_bytes, &computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (computed_end != staged_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_dim_count = staged_dim_count;
    *out_dims_cells = staged_dims_cells;
    *out_required_bytes = staged_required_bytes;
    *out_type_value = staged_type_value;
    *out_tensor_offset = staged_tensor_offset;
    *out_last_dim_index = staged_last_dim_index;
    *out_next_cursor = staged_next_cursor;
    return GGUF_TENSOR_PARSE_OK;
}

// Diagnostics-only no-write companion for dims/type/offset tuple decoding.
//
// Composes `...CommitOnly` then recomputes canonical tuple from immutable
// {buf,size,cursor} snapshots via `...CheckedNoPartial`. Caller-visible outputs
// are published only after exact tuple parity succeeds.
I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnly(
    U8 *buf,
    U64 size,
    U64 cursor,
    U32 *out_dim_count,
    U64 *out_dims_cells,
    U64 *out_required_bytes,
    U32 *out_type_value,
    U64 *out_tensor_offset,
    U64 *out_last_dim_index)
{
    U8 *snapshot_buf;
    U64 snapshot_size;
    U64 snapshot_cursor;

    U32 staged_dim_count;
    U64 staged_dims_cells;
    U64 staged_required_bytes;
    U32 staged_type_value;
    U64 staged_tensor_offset;
    U64 staged_last_dim_index;
    U64 staged_next_cursor;

    U32 canonical_dim_count;
    U64 canonical_dims[GGUF_TENSOR_MAX_DIMS];
    U64 canonical_n_elements;
    U32 canonical_type_value;
    U64 canonical_tensor_offset;
    U64 canonical_next_cursor;
    U64 canonical_dims_cells;
    U64 canonical_required_bytes;
    U64 canonical_last_dim_index;
    U64 canonical_dims_bytes;
    U64 computed_end;
    I64 status;

    if (!buf || !out_dim_count || !out_dims_cells || !out_required_bytes)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (!out_type_value || !out_tensor_offset || !out_last_dim_index)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (out_dim_count == (U32 *)out_dims_cells ||
        out_dim_count == (U32 *)out_required_bytes ||
        out_dim_count == out_type_value ||
        out_dim_count == (U32 *)out_tensor_offset ||
        out_dim_count == (U32 *)out_last_dim_index ||
        out_dims_cells == out_required_bytes ||
        out_dims_cells == (U64 *)out_type_value ||
        out_dims_cells == out_tensor_offset ||
        out_dims_cells == out_last_dim_index ||
        out_required_bytes == (U64 *)out_type_value ||
        out_required_bytes == out_tensor_offset ||
        out_required_bytes == out_last_dim_index ||
        out_type_value == (U32 *)out_tensor_offset ||
        out_type_value == (U32 *)out_last_dim_index ||
        out_tensor_offset == out_last_dim_index)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_dim_count = 0;
    staged_dims_cells = 0;
    staged_required_bytes = 0;
    staged_type_value = 0;
    staged_tensor_offset = 0;
    staged_last_dim_index = 0;
    staged_next_cursor = 0;

    canonical_dim_count = 0;
    canonical_n_elements = 0;
    canonical_type_value = 0;
    canonical_tensor_offset = 0;
    canonical_next_cursor = 0;
    canonical_dims_cells = 0;
    canonical_required_bytes = 0;
    canonical_last_dim_index = 0;

    status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnly(
        buf,
        size,
        cursor,
        &staged_dim_count,
        &staged_dims_cells,
        &staged_required_bytes,
        &staged_type_value,
        &staged_tensor_offset,
        &staged_last_dim_index,
        &staged_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartial(
        buf,
        size,
        cursor,
        &canonical_dim_count,
        canonical_dims,
        &canonical_n_elements,
        &canonical_type_value,
        &canonical_tensor_offset,
        &canonical_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    canonical_dims_cells = canonical_dim_count;
    canonical_last_dim_index = canonical_dim_count - 1;

    if (!GGUFTensorTryMulU64(canonical_dims_cells, 8, &canonical_dims_bytes))
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;

    if (!GGUFTensorTryAddU64(4, canonical_dims_bytes, &canonical_required_bytes))
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;
    if (!GGUFTensorTryAddU64(canonical_required_bytes, 4, &canonical_required_bytes))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTryAddU64(canonical_required_bytes, 8, &canonical_required_bytes))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor, canonical_required_bytes, &computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (computed_end != canonical_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_dim_count != canonical_dim_count ||
        staged_dims_cells != canonical_dims_cells ||
        staged_required_bytes != canonical_required_bytes ||
        staged_type_value != canonical_type_value ||
        staged_tensor_offset != canonical_tensor_offset ||
        staged_last_dim_index != canonical_last_dim_index ||
        staged_next_cursor != canonical_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_dim_count = staged_dim_count;
    *out_dims_cells = staged_dims_cells;
    *out_required_bytes = staged_required_bytes;
    *out_type_value = staged_type_value;
    *out_tensor_offset = staged_tensor_offset;
    *out_last_dim_index = staged_last_dim_index;
    return GGUF_TENSOR_PARSE_OK;
}

I64 GGUFTensorParseOne(U8 *buf, I64 size, I64 *cursor, GGUFTensorInfo *out)
{
    U32 n_dims;
    U64 n_elements;
    U32 ggml_type;
    U64 offset;
    U64 next_cursor;
    I64 err;

    if (!cursor || !out)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    GGUFTensorZeroInfo(out);

    err = GGUFTensorInfoReadNameCheckedNoPartial(buf,
                                                 size,
                                                 *cursor,
                                                 &out->name,
                                                 &next_cursor);
    if (err != GGUF_TENSOR_PARSE_OK)
        return err;
    *cursor = next_cursor;

    err = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartial(buf,
                                                           size,
                                                           *cursor,
                                                           &n_dims,
                                                           out->dims,
                                                           &n_elements,
                                                           &ggml_type,
                                                           &offset,
                                                           &next_cursor);
    if (err != GGUF_TENSOR_PARSE_OK)
        return err;
    *cursor = next_cursor;

    out->n_dims = n_dims;
    out->n_elements = n_elements;
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

// Commit-only diagnostics wrapper for one GGUF tensor-info tuple.
//
// Composes `GGUFTensorInfoReadCheckedNoPartial` and stages canonical
// diagnostics tuple:
//   {name_len, dim_count, dims_cells, required_bytes, type_value,
//    tensor_offset, next_cursor}
// Outputs are published atomically only after immutable {buf,size,cursor}
// snapshot parity and exact span math validation.
I64 GGUFTensorInfoReadCheckedNoPartialCommitOnly(U8 *buf,
                                                 U64 size,
                                                 U64 cursor,
                                                 U64 *out_name_len,
                                                 U32 *out_dim_count,
                                                 U64 *out_dims_cells,
                                                 U64 *out_required_bytes,
                                                 U32 *out_type_value,
                                                 U64 *out_tensor_offset,
                                                 U64 *out_next_cursor)
{
    U8 *snapshot_buf;
    U64 snapshot_size;
    U64 snapshot_cursor;

    GGUFTensorInfo staged;
    U64 staged_next_cursor;
    U64 staged_name_len;
    U32 staged_dim_count;
    U64 staged_dims_cells;
    U64 staged_required_bytes;
    U32 staged_type_value;
    U64 staged_tensor_offset;

    U64 dims_bytes;
    U64 computed_end;
    I64 status;
    U8 *out_name_len_ptr;
    U8 *out_dim_count_ptr;
    U8 *out_dims_cells_ptr;
    U8 *out_required_bytes_ptr;
    U8 *out_type_value_ptr;
    U8 *out_tensor_offset_ptr;
    U8 *out_next_cursor_ptr;

    if (!buf || !out_name_len || !out_dim_count || !out_dims_cells ||
        !out_required_bytes || !out_type_value || !out_tensor_offset ||
        !out_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    out_name_len_ptr = (U8 *)out_name_len;
    out_dim_count_ptr = (U8 *)out_dim_count;
    out_dims_cells_ptr = (U8 *)out_dims_cells;
    out_required_bytes_ptr = (U8 *)out_required_bytes;
    out_type_value_ptr = (U8 *)out_type_value;
    out_tensor_offset_ptr = (U8 *)out_tensor_offset;
    out_next_cursor_ptr = (U8 *)out_next_cursor;

    if (out_name_len_ptr == out_dim_count_ptr ||
        out_name_len_ptr == out_dims_cells_ptr ||
        out_name_len_ptr == out_required_bytes_ptr ||
        out_name_len_ptr == out_type_value_ptr ||
        out_name_len_ptr == out_tensor_offset_ptr ||
        out_name_len_ptr == out_next_cursor_ptr ||
        out_dim_count_ptr == out_dims_cells_ptr ||
        out_dim_count_ptr == out_required_bytes_ptr ||
        out_dim_count_ptr == out_type_value_ptr ||
        out_dim_count_ptr == out_tensor_offset_ptr ||
        out_dim_count_ptr == out_next_cursor_ptr ||
        out_dims_cells_ptr == out_required_bytes_ptr ||
        out_dims_cells_ptr == out_type_value_ptr ||
        out_dims_cells_ptr == out_tensor_offset_ptr ||
        out_dims_cells_ptr == out_next_cursor_ptr ||
        out_required_bytes_ptr == out_type_value_ptr ||
        out_required_bytes_ptr == out_tensor_offset_ptr ||
        out_required_bytes_ptr == out_next_cursor_ptr ||
        out_type_value_ptr == out_tensor_offset_ptr ||
        out_type_value_ptr == out_next_cursor_ptr ||
        out_tensor_offset_ptr == out_next_cursor_ptr)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    GGUFTensorZeroInfo(&staged);
    staged_next_cursor = 0;

    status = GGUFTensorInfoReadCheckedNoPartial(buf,
                                                size,
                                                cursor,
                                                &staged,
                                                &staged_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
    {
        GGUFTensorFreeString(&staged.name);
        return status;
    }

    staged_name_len = staged.name.len;
    staged_dim_count = staged.n_dims;
    staged_dims_cells = staged_dim_count;
    staged_type_value = staged.ggml_type;
    staged_tensor_offset = staged.offset;

    if (!GGUFTensorTryAddU64(8, staged_name_len, &staged_required_bytes))
    {
        GGUFTensorFreeString(&staged.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }
    if (!GGUFTensorTryAddU64(staged_required_bytes, 4, &staged_required_bytes))
    {
        GGUFTensorFreeString(&staged.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }
    if (!GGUFTensorTryMulU64(staged_dims_cells, 8, &dims_bytes))
    {
        GGUFTensorFreeString(&staged.name);
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;
    }
    if (!GGUFTensorTryAddU64(staged_required_bytes, dims_bytes, &staged_required_bytes))
    {
        GGUFTensorFreeString(&staged.name);
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;
    }
    if (!GGUFTensorTryAddU64(staged_required_bytes, 4, &staged_required_bytes))
    {
        GGUFTensorFreeString(&staged.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }
    if (!GGUFTensorTryAddU64(staged_required_bytes, 8, &staged_required_bytes))
    {
        GGUFTensorFreeString(&staged.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }

    if (!GGUFTensorTryAddU64(cursor, staged_required_bytes, &computed_end))
    {
        GGUFTensorFreeString(&staged.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }
    if (computed_end != staged_next_cursor)
    {
        GGUFTensorFreeString(&staged.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
    {
        GGUFTensorFreeString(&staged.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }

    *out_name_len = staged_name_len;
    *out_dim_count = staged_dim_count;
    *out_dims_cells = staged_dims_cells;
    *out_required_bytes = staged_required_bytes;
    *out_type_value = staged_type_value;
    *out_tensor_offset = staged_tensor_offset;
    *out_next_cursor = staged_next_cursor;

    GGUFTensorFreeString(&staged.name);
    return GGUF_TENSOR_PARSE_OK;
}

// Diagnostics-only no-write companion for full tensor-info tuple decoding.
//
// Composes `...CommitOnly`, recomputes canonical tuple from immutable
// {buf,size,cursor} snapshots via `...CheckedNoPartial`, and publishes only
// when all tuple fields match exactly under strict no-partial semantics.
I64 GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnly(
    U8 *buf,
    U64 size,
    U64 cursor,
    U64 *out_name_len,
    U32 *out_dim_count,
    U64 *out_dims_cells,
    U64 *out_required_bytes,
    U32 *out_type_value,
    U64 *out_tensor_offset,
    U64 *out_next_cursor)
{
    U8 *snapshot_buf;
    U64 snapshot_size;
    U64 snapshot_cursor;

    U64 staged_name_len;
    U32 staged_dim_count;
    U64 staged_dims_cells;
    U64 staged_required_bytes;
    U32 staged_type_value;
    U64 staged_tensor_offset;
    U64 staged_next_cursor;

    GGUFTensorInfo canonical;
    U64 canonical_next_cursor;
    U64 canonical_name_len;
    U32 canonical_dim_count;
    U64 canonical_dims_cells;
    U64 canonical_required_bytes;
    U32 canonical_type_value;
    U64 canonical_tensor_offset;
    U64 dims_bytes;
    U64 computed_end;
    I64 status;
    U8 *out_name_len_ptr;
    U8 *out_dim_count_ptr;
    U8 *out_dims_cells_ptr;
    U8 *out_required_bytes_ptr;
    U8 *out_type_value_ptr;
    U8 *out_tensor_offset_ptr;
    U8 *out_next_cursor_ptr;

    if (!buf || !out_name_len || !out_dim_count || !out_dims_cells)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (!out_required_bytes || !out_type_value || !out_tensor_offset || !out_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    out_name_len_ptr = (U8 *)out_name_len;
    out_dim_count_ptr = (U8 *)out_dim_count;
    out_dims_cells_ptr = (U8 *)out_dims_cells;
    out_required_bytes_ptr = (U8 *)out_required_bytes;
    out_type_value_ptr = (U8 *)out_type_value;
    out_tensor_offset_ptr = (U8 *)out_tensor_offset;
    out_next_cursor_ptr = (U8 *)out_next_cursor;

    if (out_name_len_ptr == out_dim_count_ptr ||
        out_name_len_ptr == out_dims_cells_ptr ||
        out_name_len_ptr == out_required_bytes_ptr ||
        out_name_len_ptr == out_type_value_ptr ||
        out_name_len_ptr == out_tensor_offset_ptr ||
        out_name_len_ptr == out_next_cursor_ptr ||
        out_dim_count_ptr == out_dims_cells_ptr ||
        out_dim_count_ptr == out_required_bytes_ptr ||
        out_dim_count_ptr == out_type_value_ptr ||
        out_dim_count_ptr == out_tensor_offset_ptr ||
        out_dim_count_ptr == out_next_cursor_ptr ||
        out_dims_cells_ptr == out_required_bytes_ptr ||
        out_dims_cells_ptr == out_type_value_ptr ||
        out_dims_cells_ptr == out_tensor_offset_ptr ||
        out_dims_cells_ptr == out_next_cursor_ptr ||
        out_required_bytes_ptr == out_type_value_ptr ||
        out_required_bytes_ptr == out_tensor_offset_ptr ||
        out_required_bytes_ptr == out_next_cursor_ptr ||
        out_type_value_ptr == out_tensor_offset_ptr ||
        out_type_value_ptr == out_next_cursor_ptr ||
        out_tensor_offset_ptr == out_next_cursor_ptr)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_name_len = 0;
    staged_dim_count = 0;
    staged_dims_cells = 0;
    staged_required_bytes = 0;
    staged_type_value = 0;
    staged_tensor_offset = 0;
    staged_next_cursor = 0;

    GGUFTensorZeroInfo(&canonical);
    canonical_next_cursor = 0;
    canonical_name_len = 0;
    canonical_dim_count = 0;
    canonical_dims_cells = 0;
    canonical_required_bytes = 0;
    canonical_type_value = 0;
    canonical_tensor_offset = 0;

    status = GGUFTensorInfoReadCheckedNoPartialCommitOnly(buf,
                                                           size,
                                                           cursor,
                                                           &staged_name_len,
                                                           &staged_dim_count,
                                                           &staged_dims_cells,
                                                           &staged_required_bytes,
                                                           &staged_type_value,
                                                           &staged_tensor_offset,
                                                           &staged_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadCheckedNoPartial(buf,
                                                size,
                                                cursor,
                                                &canonical,
                                                &canonical_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
    {
        GGUFTensorFreeString(&canonical.name);
        return status;
    }

    canonical_name_len = canonical.name.len;
    canonical_dim_count = canonical.n_dims;
    canonical_dims_cells = canonical_dim_count;
    canonical_type_value = canonical.ggml_type;
    canonical_tensor_offset = canonical.offset;

    if (!GGUFTensorTryAddU64(8, canonical_name_len, &canonical_required_bytes))
    {
        GGUFTensorFreeString(&canonical.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }
    if (!GGUFTensorTryAddU64(canonical_required_bytes, 4, &canonical_required_bytes))
    {
        GGUFTensorFreeString(&canonical.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }
    if (!GGUFTensorTryMulU64(canonical_dims_cells, 8, &dims_bytes))
    {
        GGUFTensorFreeString(&canonical.name);
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;
    }
    if (!GGUFTensorTryAddU64(canonical_required_bytes, dims_bytes, &canonical_required_bytes))
    {
        GGUFTensorFreeString(&canonical.name);
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;
    }
    if (!GGUFTensorTryAddU64(canonical_required_bytes, 4, &canonical_required_bytes))
    {
        GGUFTensorFreeString(&canonical.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }
    if (!GGUFTensorTryAddU64(canonical_required_bytes, 8, &canonical_required_bytes))
    {
        GGUFTensorFreeString(&canonical.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }

    if (!GGUFTensorTryAddU64(cursor, canonical_required_bytes, &computed_end))
    {
        GGUFTensorFreeString(&canonical.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }
    if (computed_end != canonical_next_cursor)
    {
        GGUFTensorFreeString(&canonical.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
    {
        GGUFTensorFreeString(&canonical.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }

    if (staged_name_len != canonical_name_len ||
        staged_dim_count != canonical_dim_count ||
        staged_dims_cells != canonical_dims_cells ||
        staged_required_bytes != canonical_required_bytes ||
        staged_type_value != canonical_type_value ||
        staged_tensor_offset != canonical_tensor_offset ||
        staged_next_cursor != canonical_next_cursor)
    {
        GGUFTensorFreeString(&canonical.name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }

    *out_name_len = staged_name_len;
    *out_dim_count = staged_dim_count;
    *out_dims_cells = staged_dims_cells;
    *out_required_bytes = staged_required_bytes;
    *out_type_value = staged_type_value;
    *out_tensor_offset = staged_tensor_offset;
    *out_next_cursor = staged_next_cursor;

    GGUFTensorFreeString(&canonical.name);
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
