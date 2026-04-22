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

Bool GGUFTensorByteSpanEnd(U8 *base, U64 bytes, U8 **out_end)
{
    U64 begin_addr;
    U64 end_addr;

    if (!base || !out_end)
        return FALSE;

    begin_addr = (U64)base;
    if (!GGUFTensorTryAddU64(begin_addr, bytes, &end_addr))
        return FALSE;

    *out_end = (U8 *)end_addr;
    return TRUE;
}

Bool GGUFTensorByteSpansOverlap(U8 *a_base,
                                U64 a_bytes,
                                U8 *b_base,
                                U64 b_bytes)
{
    U8 *a_end;
    U8 *b_end;

    if (!a_base || !b_base)
        return FALSE;
    if (!a_bytes || !b_bytes)
        return FALSE;

    if (!GGUFTensorByteSpanEnd(a_base, a_bytes, &a_end))
        return TRUE;
    if (!GGUFTensorByteSpanEnd(b_base, b_bytes, &b_end))
        return TRUE;

    if (a_end <= b_base || b_end <= a_base)
        return FALSE;
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

// Diagnostics-only no-write companion for tensor name tuple decoding.
//
// Composes `...CommitOnly`, then recomputes canonical
// {name_len, required_bytes, next_cursor} from immutable {buf,size,cursor}
// snapshots. Outputs are published atomically only when staged and canonical
// tuples match exactly, preserving strict no-partial semantics.
I64 GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnly(
    U8 *buf,
    U64 size,
    U64 cursor,
    U64 *out_name_len,
    U64 *out_required_bytes,
    U64 *out_next_cursor)
{
    U8 *snapshot_buf;
    U64 snapshot_size;
    U64 snapshot_cursor;

    U64 staged_name_len;
    U64 staged_required_bytes;
    U64 staged_next_cursor;

    GGUFTensorString canonical_name;
    U64 canonical_name_len;
    U64 canonical_required_bytes;
    U64 canonical_next_cursor;
    U64 computed_end;
    I64 status;

    U8 *out_name_len_ptr;
    U8 *out_required_bytes_ptr;
    U8 *out_next_cursor_ptr;

    if (!buf || !out_name_len || !out_required_bytes || !out_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    out_name_len_ptr = (U8 *)out_name_len;
    out_required_bytes_ptr = (U8 *)out_required_bytes;
    out_next_cursor_ptr = (U8 *)out_next_cursor;

    if (out_name_len_ptr == out_required_bytes_ptr ||
        out_name_len_ptr == out_next_cursor_ptr ||
        out_required_bytes_ptr == out_next_cursor_ptr)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_name_len = 0;
    staged_required_bytes = 0;
    staged_next_cursor = 0;

    GGUFTensorZeroString(&canonical_name);
    canonical_name_len = 0;
    canonical_required_bytes = 0;
    canonical_next_cursor = 0;

    status = GGUFTensorInfoReadNameCheckedNoPartialCommitOnly(buf,
                                                               size,
                                                               cursor,
                                                               &staged_name_len,
                                                               &staged_required_bytes,
                                                               &staged_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadNameCheckedNoPartial(buf,
                                                    size,
                                                    cursor,
                                                    &canonical_name,
                                                    &canonical_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
    {
        GGUFTensorFreeString(&canonical_name);
        return status;
    }

    canonical_name_len = canonical_name.len;
    if (canonical_name_len > GGUF_TENSOR_MAX_NAME_BYTES)
    {
        GGUFTensorFreeString(&canonical_name);
        return GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN;
    }
    if (!GGUFTensorTryAddU64(8, canonical_name_len, &canonical_required_bytes))
    {
        GGUFTensorFreeString(&canonical_name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }

    // Keep explicit U64 span math so the byte window is auditable: name header
    // (8 bytes) + payload (name_len), then cursor + required_bytes.
    if (canonical_next_cursor > size)
    {
        GGUFTensorFreeString(&canonical_name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }
    if (!GGUFTensorTryAddU64(cursor, canonical_required_bytes, &computed_end))
    {
        GGUFTensorFreeString(&canonical_name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }
    if (computed_end > size)
    {
        GGUFTensorFreeString(&canonical_name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }
    if (computed_end != canonical_next_cursor)
    {
        GGUFTensorFreeString(&canonical_name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
    {
        GGUFTensorFreeString(&canonical_name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }

    if (staged_name_len != canonical_name_len ||
        staged_required_bytes != canonical_required_bytes ||
        staged_next_cursor != canonical_next_cursor)
    {
        GGUFTensorFreeString(&canonical_name);
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    }

    GGUFTensorFreeString(&canonical_name);

    *out_name_len = staged_name_len;
    *out_required_bytes = staged_required_bytes;
    *out_next_cursor = staged_next_cursor;
    return GGUF_TENSOR_PARSE_OK;
}

// IQ-1020 diagnostics-only parity gate for tensor name tuple decoding.
//
// Composes `...CommitOnlyPreflightOnly` and canonical `...CommitOnly`, then
// publishes atomically only when immutable snapshots and tuple parity match:
//   {name_len, required_bytes, next_cursor}
I64 GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParity(
    U8 *buf,
    U64 size,
    U64 cursor,
    U64 *out_name_len,
    U64 *out_required_bytes,
    U64 *out_next_cursor)
{
    U8 *snapshot_buf;
    U64 snapshot_size;
    U64 snapshot_cursor;

    U64 staged_pre_name_len;
    U64 staged_pre_required_bytes;
    U64 staged_pre_next_cursor;
    U64 staged_pre_computed_end;
    U64 staged_pre_required_check;
    U64 staged_pre_span_remaining;

    U64 staged_commit_name_len;
    U64 staged_commit_required_bytes;
    U64 staged_commit_next_cursor;
    U64 staged_commit_computed_end;
    U64 staged_commit_required_check;
    U64 staged_commit_span_remaining;

    I64 status;

    U8 *out_name_len_ptr;
    U8 *out_required_bytes_ptr;
    U8 *out_next_cursor_ptr;

    if (!buf || !out_name_len || !out_required_bytes || !out_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    out_name_len_ptr = (U8 *)out_name_len;
    out_required_bytes_ptr = (U8 *)out_required_bytes;
    out_next_cursor_ptr = (U8 *)out_next_cursor;

    if (out_name_len_ptr == out_required_bytes_ptr ||
        out_name_len_ptr == out_next_cursor_ptr ||
        out_required_bytes_ptr == out_next_cursor_ptr)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_pre_name_len = 0;
    staged_pre_required_bytes = 0;
    staged_pre_next_cursor = 0;
    staged_pre_computed_end = 0;
    staged_pre_required_check = 0;
    staged_pre_span_remaining = 0;

    staged_commit_name_len = 0;
    staged_commit_required_bytes = 0;
    staged_commit_next_cursor = 0;
    staged_commit_computed_end = 0;
    staged_commit_required_check = 0;
    staged_commit_span_remaining = 0;

    status = GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnly(
        buf,
        size,
        cursor,
        &staged_pre_name_len,
        &staged_pre_required_bytes,
        &staged_pre_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadNameCheckedNoPartialCommitOnly(
        buf,
        size,
        cursor,
        &staged_commit_name_len,
        &staged_commit_required_bytes,
        &staged_commit_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (!GGUFTensorTryAddU64(8, staged_pre_name_len, &staged_pre_required_check))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTryAddU64(8, staged_commit_name_len, &staged_commit_required_check))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_pre_name_len > GGUF_TENSOR_MAX_NAME_BYTES ||
        staged_commit_name_len > GGUF_TENSOR_MAX_NAME_BYTES)
        return GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN;

    // Keep explicit cursor span checks visible: cursor + required_bytes must
    // exactly match staged next_cursor in both composed paths.
    if (!GGUFTensorTryAddU64(cursor,
                             staged_pre_required_bytes,
                             &staged_pre_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTryAddU64(cursor,
                             staged_commit_required_bytes,
                             &staged_commit_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    staged_pre_span_remaining = size - cursor;
    staged_commit_span_remaining = size - cursor;
    if (staged_pre_required_bytes > staged_pre_span_remaining ||
        staged_commit_required_bytes > staged_commit_span_remaining)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_pre_computed_end > size ||
        staged_commit_computed_end > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_pre_next_cursor < cursor ||
        staged_commit_next_cursor < cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_pre_computed_end != staged_pre_next_cursor ||
        staged_commit_computed_end != staged_commit_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_pre_name_len != staged_commit_name_len ||
        staged_pre_required_bytes != staged_commit_required_bytes ||
        staged_pre_next_cursor != staged_commit_next_cursor ||
        staged_pre_required_bytes != staged_pre_required_check ||
        staged_commit_required_bytes != staged_commit_required_check)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_name_len = staged_commit_name_len;
    *out_required_bytes = staged_commit_required_bytes;
    *out_next_cursor = staged_commit_next_cursor;
    return GGUF_TENSOR_PARSE_OK;
}

// IQ-1028 commit-only diagnostics wrapper for tensor name parity preflight.
//
// Stages tuple diagnostics via
// `...CommitOnlyPreflightOnlyParity`, then revalidates immutable
// {buf,size,cursor} snapshots and canonical span math:
//   required_bytes = 8 + name_len
//   next_cursor = cursor + required_bytes
// and atomically publishes only on strict parity success.
I64 GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
    U8 *buf,
    U64 size,
    U64 cursor,
    U64 *out_name_len,
    U64 *out_required_bytes,
    U64 *out_next_cursor)
{
    U8 *snapshot_buf;
    U64 snapshot_size;
    U64 snapshot_cursor;

    U64 staged_name_len;
    U64 staged_required_bytes;
    U64 staged_next_cursor;

    U64 canonical_required_bytes;
    U64 canonical_next_cursor;
    U64 span_remaining;

    U8 *out_name_len_ptr;
    U8 *out_required_bytes_ptr;
    U8 *out_next_cursor_ptr;

    I64 status;

    if (!buf || !out_name_len || !out_required_bytes || !out_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    out_name_len_ptr = (U8 *)out_name_len;
    out_required_bytes_ptr = (U8 *)out_required_bytes;
    out_next_cursor_ptr = (U8 *)out_next_cursor;

    if (GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64),
                                   out_required_bytes_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64),
                                   out_next_cursor_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_required_bytes_ptr, sizeof(U64),
                                   out_next_cursor_ptr, sizeof(U64)))
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_name_len = 0;
    staged_required_bytes = 0;
    staged_next_cursor = 0;

    canonical_required_bytes = 0;
    canonical_next_cursor = 0;
    span_remaining = 0;

    status = GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParity(
        buf,
        size,
        cursor,
        &staged_name_len,
        &staged_required_bytes,
        &staged_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_name_len > GGUF_TENSOR_MAX_NAME_BYTES)
        return GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN;

    if (!GGUFTensorTryAddU64(8, staged_name_len, &canonical_required_bytes))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_required_bytes != canonical_required_bytes)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    span_remaining = size - cursor;
    if (staged_required_bytes > span_remaining)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor, staged_required_bytes, &canonical_next_cursor))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_next_cursor < cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (canonical_next_cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (canonical_next_cursor != staged_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_name_len = staged_name_len;
    *out_required_bytes = staged_required_bytes;
    *out_next_cursor = staged_next_cursor;
    return GGUF_TENSOR_PARSE_OK;
}

// IQ-1036 diagnostics-only no-write companion for tensor name parity commit.
//
// Composes `...ParityCommitOnly` and canonical
// `...CommitOnlyPreflightOnly`, recomputes canonical span tuple
// `{name_len, required_bytes, next_cursor}` from immutable snapshots,
// and atomically publishes outputs only on strict tuple parity.
I64 GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    U8 *buf,
    U64 size,
    U64 cursor,
    U64 *out_name_len,
    U64 *out_required_bytes,
    U64 *out_next_cursor)
{
    U8 *snapshot_buf;
    U64 snapshot_size;
    U64 snapshot_cursor;

    U64 staged_commit_name_len;
    U64 staged_commit_required_bytes;
    U64 staged_commit_next_cursor;
    U64 staged_commit_required_check;
    U64 staged_commit_computed_end;

    U64 canonical_name_len;
    U64 canonical_required_bytes;
    U64 canonical_next_cursor;
    U64 canonical_required_check;
    U64 canonical_computed_end;

    U64 raw_name_len;
    U64 raw_required_bytes;
    U64 raw_computed_end;
    U64 raw_span_remaining;

    U8 *out_name_len_ptr;
    U8 *out_required_bytes_ptr;
    U8 *out_next_cursor_ptr;

    I64 status;

    if (!buf || !out_name_len || !out_required_bytes || !out_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    out_name_len_ptr = (U8 *)out_name_len;
    out_required_bytes_ptr = (U8 *)out_required_bytes;
    out_next_cursor_ptr = (U8 *)out_next_cursor;

    if (out_name_len_ptr == out_required_bytes_ptr ||
        out_name_len_ptr == out_next_cursor_ptr ||
        out_required_bytes_ptr == out_next_cursor_ptr)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_commit_name_len = 0;
    staged_commit_required_bytes = 0;
    staged_commit_next_cursor = 0;
    staged_commit_required_check = 0;
    staged_commit_computed_end = 0;

    canonical_name_len = 0;
    canonical_required_bytes = 0;
    canonical_next_cursor = 0;
    canonical_required_check = 0;
    canonical_computed_end = 0;

    raw_name_len = 0;
    raw_required_bytes = 0;
    raw_computed_end = 0;
    raw_span_remaining = 0;

    status = GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        buf,
        size,
        cursor,
        &staged_commit_name_len,
        &staged_commit_required_bytes,
        &staged_commit_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnly(
        buf,
        size,
        cursor,
        &canonical_name_len,
        &canonical_required_bytes,
        &canonical_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (!GGUFTensorTryAddU64(8, staged_commit_name_len, &staged_commit_required_check))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTryAddU64(8, canonical_name_len, &canonical_required_check))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_commit_name_len > GGUF_TENSOR_MAX_NAME_BYTES ||
        canonical_name_len > GGUF_TENSOR_MAX_NAME_BYTES)
        return GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN;

    if (!GGUFTensorCanRead(buf, size, cursor, 8))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    // Reconstruct the on-wire name header directly from immutable bytes so
    // parity checks are auditable against raw GGUF payload, not only helper
    // outputs.
    raw_name_len = (U64)buf[cursor]
                 | ((U64)buf[cursor + 1] << 8)
                 | ((U64)buf[cursor + 2] << 16)
                 | ((U64)buf[cursor + 3] << 24)
                 | ((U64)buf[cursor + 4] << 32)
                 | ((U64)buf[cursor + 5] << 40)
                 | ((U64)buf[cursor + 6] << 48)
                 | ((U64)buf[cursor + 7] << 56);

    if (raw_name_len > GGUF_TENSOR_MAX_NAME_BYTES)
        return GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN;
    if (!GGUFTensorTryAddU64(8, raw_name_len, &raw_required_bytes))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    raw_span_remaining = size - cursor;
    if (raw_required_bytes > raw_span_remaining)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTryAddU64(cursor, raw_required_bytes, &raw_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (raw_computed_end > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_commit_required_bytes,
                             &staged_commit_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTryAddU64(cursor,
                             canonical_required_bytes,
                             &canonical_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_commit_computed_end > size ||
        canonical_computed_end > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_commit_computed_end != staged_commit_next_cursor ||
        canonical_computed_end != canonical_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_commit_name_len != canonical_name_len ||
        staged_commit_name_len != raw_name_len ||
        staged_commit_required_bytes != canonical_required_bytes ||
        staged_commit_required_bytes != raw_required_bytes ||
        staged_commit_next_cursor != canonical_next_cursor ||
        staged_commit_next_cursor != raw_computed_end ||
        staged_commit_required_bytes != staged_commit_required_check ||
        canonical_required_bytes != canonical_required_check)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_name_len = staged_commit_name_len;
    *out_required_bytes = staged_commit_required_bytes;
    *out_next_cursor = staged_commit_next_cursor;
    return GGUF_TENSOR_PARSE_OK;
}

// IQ-1030 diagnostics-only parity gate for tensor-name preflight parity wrappers.
//
// Composes `...ParityCommitOnlyPreflightOnly` and canonical
// `...ParityCommitOnly`, enforces immutable {buf,size,cursor} snapshots and
// exact tuple parity over `{name_len, required_bytes, next_cursor}` before
// atomic publish.
I64 GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
    U8 *buf,
    U64 size,
    U64 cursor,
    U64 *out_name_len,
    U64 *out_required_bytes,
    U64 *out_next_cursor)
{
    U8 *snapshot_buf;
    U64 snapshot_size;
    U64 snapshot_cursor;

    U64 staged_preflight_name_len;
    U64 staged_preflight_required_bytes;
    U64 staged_preflight_next_cursor;

    U64 staged_parity_name_len;
    U64 staged_parity_required_bytes;
    U64 staged_parity_next_cursor;

    U64 staged_preflight_required_check;
    U64 staged_parity_required_check;
    U64 staged_preflight_computed_end;
    U64 staged_parity_computed_end;
    U64 staged_preflight_span_remaining;
    U64 staged_parity_span_remaining;

    U8 *out_name_len_ptr;
    U8 *out_required_bytes_ptr;
    U8 *out_next_cursor_ptr;

    I64 status;

    if (!buf || !out_name_len || !out_required_bytes || !out_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    out_name_len_ptr = (U8 *)out_name_len;
    out_required_bytes_ptr = (U8 *)out_required_bytes;
    out_next_cursor_ptr = (U8 *)out_next_cursor;

    if (out_name_len_ptr == out_required_bytes_ptr ||
        out_name_len_ptr == out_next_cursor_ptr ||
        out_required_bytes_ptr == out_next_cursor_ptr)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_preflight_name_len = 0;
    staged_preflight_required_bytes = 0;
    staged_preflight_next_cursor = 0;

    staged_parity_name_len = 0;
    staged_parity_required_bytes = 0;
    staged_parity_next_cursor = 0;

    staged_preflight_required_check = 0;
    staged_parity_required_check = 0;
    staged_preflight_computed_end = 0;
    staged_parity_computed_end = 0;
    staged_preflight_span_remaining = 0;
    staged_parity_span_remaining = 0;

    status = GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        buf,
        size,
        cursor,
        &staged_preflight_name_len,
        &staged_preflight_required_bytes,
        &staged_preflight_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        buf,
        size,
        cursor,
        &staged_parity_name_len,
        &staged_parity_required_bytes,
        &staged_parity_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (!GGUFTensorTryAddU64(8,
                             staged_preflight_name_len,
                             &staged_preflight_required_check))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTryAddU64(8,
                             staged_parity_name_len,
                             &staged_parity_required_check))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_preflight_required_bytes,
                             &staged_preflight_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTryAddU64(cursor,
                             staged_parity_required_bytes,
                             &staged_parity_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    staged_preflight_span_remaining = size - cursor;
    staged_parity_span_remaining = size - cursor;

    if (staged_preflight_required_bytes > staged_preflight_span_remaining ||
        staged_parity_required_bytes > staged_parity_span_remaining)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_preflight_computed_end > size ||
        staged_parity_computed_end > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_preflight_computed_end != staged_preflight_next_cursor ||
        staged_parity_computed_end != staged_parity_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_preflight_name_len != staged_parity_name_len ||
        staged_preflight_name_len > GGUF_TENSOR_MAX_NAME_BYTES ||
        staged_parity_name_len > GGUF_TENSOR_MAX_NAME_BYTES ||
        staged_preflight_required_bytes != staged_parity_required_bytes ||
        staged_preflight_next_cursor != staged_parity_next_cursor ||
        staged_preflight_required_bytes != staged_preflight_required_check ||
        staged_parity_required_bytes != staged_parity_required_check)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_name_len = staged_parity_name_len;
    *out_required_bytes = staged_parity_required_bytes;
    *out_next_cursor = staged_parity_next_cursor;
    return GGUF_TENSOR_PARSE_OK;
}

// IQ-1031 commit-only diagnostics wrapper for tensor-name parity-preflight gate.
//
// Composes `...ParityCommitOnlyPreflightOnlyParity` and canonical
// `...ParityCommitOnlyPreflightOnly`, revalidates immutable
// `{buf,size,cursor}` snapshots and canonical tuple span math
// `{name_len,required_bytes,next_cursor}`, then atomically publishes only
// when strict parity holds.
I64 GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
    U8 *buf,
    U64 size,
    U64 cursor,
    U64 *out_name_len,
    U64 *out_required_bytes,
    U64 *out_next_cursor)
{
    U8 *snapshot_buf;
    U64 snapshot_size;
    U64 snapshot_cursor;

    U64 staged_parity_name_len;
    U64 staged_parity_required_bytes;
    U64 staged_parity_next_cursor;
    U64 staged_parity_required_check;
    U64 staged_parity_computed_end;
    U64 staged_parity_span_remaining;

    U64 canonical_name_len;
    U64 canonical_required_bytes;
    U64 canonical_next_cursor;
    U64 canonical_required_check;
    U64 canonical_computed_end;
    U64 canonical_span_remaining;

    U8 *out_name_len_ptr;
    U8 *out_required_bytes_ptr;
    U8 *out_next_cursor_ptr;

    I64 status;

    if (!buf || !out_name_len || !out_required_bytes || !out_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    out_name_len_ptr = (U8 *)out_name_len;
    out_required_bytes_ptr = (U8 *)out_required_bytes;
    out_next_cursor_ptr = (U8 *)out_next_cursor;

    if (out_name_len_ptr == out_required_bytes_ptr ||
        out_name_len_ptr == out_next_cursor_ptr ||
        out_required_bytes_ptr == out_next_cursor_ptr)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_parity_name_len = 0;
    staged_parity_required_bytes = 0;
    staged_parity_next_cursor = 0;
    staged_parity_required_check = 0;
    staged_parity_computed_end = 0;
    staged_parity_span_remaining = 0;

    canonical_name_len = 0;
    canonical_required_bytes = 0;
    canonical_next_cursor = 0;
    canonical_required_check = 0;
    canonical_computed_end = 0;
    canonical_span_remaining = 0;

    status = GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        buf,
        size,
        cursor,
        &staged_parity_name_len,
        &staged_parity_required_bytes,
        &staged_parity_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        buf,
        size,
        cursor,
        &canonical_name_len,
        &canonical_required_bytes,
        &canonical_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (!GGUFTensorTryAddU64(8,
                             staged_parity_name_len,
                             &staged_parity_required_check))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTryAddU64(8,
                             canonical_name_len,
                             &canonical_required_check))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_parity_required_bytes,
                             &staged_parity_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTryAddU64(cursor,
                             canonical_required_bytes,
                             &canonical_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    staged_parity_span_remaining = size - cursor;
    canonical_span_remaining = size - cursor;

    if (staged_parity_required_bytes > staged_parity_span_remaining ||
        canonical_required_bytes > canonical_span_remaining)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_parity_computed_end > size ||
        canonical_computed_end > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_parity_computed_end != staged_parity_next_cursor ||
        canonical_computed_end != canonical_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_parity_name_len != canonical_name_len ||
        staged_parity_name_len > GGUF_TENSOR_MAX_NAME_BYTES ||
        canonical_name_len > GGUF_TENSOR_MAX_NAME_BYTES ||
        staged_parity_required_bytes != canonical_required_bytes ||
        staged_parity_next_cursor != canonical_next_cursor ||
        staged_parity_required_bytes != staged_parity_required_check ||
        canonical_required_bytes != canonical_required_check)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_name_len = staged_parity_name_len;
    *out_required_bytes = staged_parity_required_bytes;
    *out_next_cursor = staged_parity_next_cursor;
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

// IQ-1017 diagnostics-only parity gate for dims/type/offset tuple diagnostics.
//
// Compose both:
//   1) preflight-only no-write diagnostics wrapper;
//   2) canonical commit-only diagnostics wrapper.
// Enforce immutable {buf,size,cursor} snapshots plus strict tuple parity over
// {dim_count, dims_cells, required_bytes, type_value, tensor_offset,
//  last_dim_index} before atomic output publish.
I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParity(
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

    U32 staged_pre_dim_count;
    U64 staged_pre_dims_cells;
    U64 staged_pre_required_bytes;
    U32 staged_pre_type_value;
    U64 staged_pre_tensor_offset;
    U64 staged_pre_last_dim_index;

    U32 staged_commit_dim_count;
    U64 staged_commit_dims_cells;
    U64 staged_commit_required_bytes;
    U32 staged_commit_type_value;
    U64 staged_commit_tensor_offset;
    U64 staged_commit_last_dim_index;
    U64 staged_commit_next_cursor;
    U64 staged_commit_computed_end;

    I64 status;
    U8 *out_dim_count_ptr;
    U8 *out_dims_cells_ptr;
    U8 *out_required_bytes_ptr;
    U8 *out_type_value_ptr;
    U8 *out_tensor_offset_ptr;
    U8 *out_last_dim_index_ptr;

    if (!buf || !out_dim_count || !out_dims_cells || !out_required_bytes)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (!out_type_value || !out_tensor_offset || !out_last_dim_index)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    out_dim_count_ptr = (U8 *)out_dim_count;
    out_dims_cells_ptr = (U8 *)out_dims_cells;
    out_required_bytes_ptr = (U8 *)out_required_bytes;
    out_type_value_ptr = (U8 *)out_type_value;
    out_tensor_offset_ptr = (U8 *)out_tensor_offset;
    out_last_dim_index_ptr = (U8 *)out_last_dim_index;

    if (out_dim_count_ptr == out_dims_cells_ptr ||
        out_dim_count_ptr == out_required_bytes_ptr ||
        out_dim_count_ptr == out_type_value_ptr ||
        out_dim_count_ptr == out_tensor_offset_ptr ||
        out_dim_count_ptr == out_last_dim_index_ptr ||
        out_dims_cells_ptr == out_required_bytes_ptr ||
        out_dims_cells_ptr == out_type_value_ptr ||
        out_dims_cells_ptr == out_tensor_offset_ptr ||
        out_dims_cells_ptr == out_last_dim_index_ptr ||
        out_required_bytes_ptr == out_type_value_ptr ||
        out_required_bytes_ptr == out_tensor_offset_ptr ||
        out_required_bytes_ptr == out_last_dim_index_ptr ||
        out_type_value_ptr == out_tensor_offset_ptr ||
        out_type_value_ptr == out_last_dim_index_ptr ||
        out_tensor_offset_ptr == out_last_dim_index_ptr)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_pre_dim_count = 0;
    staged_pre_dims_cells = 0;
    staged_pre_required_bytes = 0;
    staged_pre_type_value = 0;
    staged_pre_tensor_offset = 0;
    staged_pre_last_dim_index = 0;

    staged_commit_dim_count = 0;
    staged_commit_dims_cells = 0;
    staged_commit_required_bytes = 0;
    staged_commit_type_value = 0;
    staged_commit_tensor_offset = 0;
    staged_commit_last_dim_index = 0;
    staged_commit_next_cursor = 0;

    status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnly(
        buf,
        size,
        cursor,
        &staged_pre_dim_count,
        &staged_pre_dims_cells,
        &staged_pre_required_bytes,
        &staged_pre_type_value,
        &staged_pre_tensor_offset,
        &staged_pre_last_dim_index);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnly(
        buf,
        size,
        cursor,
        &staged_commit_dim_count,
        &staged_commit_dims_cells,
        &staged_commit_required_bytes,
        &staged_commit_type_value,
        &staged_commit_tensor_offset,
        &staged_commit_last_dim_index,
        &staged_commit_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_pre_dim_count != staged_commit_dim_count ||
        staged_pre_dims_cells != staged_commit_dims_cells ||
        staged_pre_required_bytes != staged_commit_required_bytes ||
        staged_pre_type_value != staged_commit_type_value ||
        staged_pre_tensor_offset != staged_commit_tensor_offset ||
        staged_pre_last_dim_index != staged_commit_last_dim_index)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_pre_dims_cells != staged_pre_dim_count ||
        staged_commit_dims_cells != staged_commit_dim_count)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_commit_required_bytes,
                             &staged_commit_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_commit_computed_end != staged_commit_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_dim_count = staged_commit_dim_count;
    *out_dims_cells = staged_commit_dims_cells;
    *out_required_bytes = staged_commit_required_bytes;
    *out_type_value = staged_commit_type_value;
    *out_tensor_offset = staged_commit_tensor_offset;
    *out_last_dim_index = staged_commit_last_dim_index;
    return GGUF_TENSOR_PARSE_OK;
}


// Commit-only diagnostics wrapper over dims/type/offset preflight parity.
//
// Stages tuple diagnostics through
// `...CommitOnlyPreflightOnlyParity`, then revalidates immutable
// {buf,size,cursor} snapshots and canonical span math
//   required_bytes = 4 + dim_count*8 + 4 + 8
//   next_cursor = cursor + required_bytes
// before atomically publishing outputs.
I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
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

    U64 canonical_dims_bytes;
    U64 canonical_required_bytes;
    U64 canonical_next_cursor;
    U64 span_remaining;

    I64 status;

    U8 *out_dim_count_ptr;
    U8 *out_dims_cells_ptr;
    U8 *out_required_bytes_ptr;
    U8 *out_type_value_ptr;
    U8 *out_tensor_offset_ptr;
    U8 *out_last_dim_index_ptr;

    if (!buf || !out_dim_count || !out_dims_cells || !out_required_bytes)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (!out_type_value || !out_tensor_offset || !out_last_dim_index)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    out_dim_count_ptr = (U8 *)out_dim_count;
    out_dims_cells_ptr = (U8 *)out_dims_cells;
    out_required_bytes_ptr = (U8 *)out_required_bytes;
    out_type_value_ptr = (U8 *)out_type_value;
    out_tensor_offset_ptr = (U8 *)out_tensor_offset;
    out_last_dim_index_ptr = (U8 *)out_last_dim_index;

    if (out_dim_count_ptr == out_dims_cells_ptr ||
        out_dim_count_ptr == out_required_bytes_ptr ||
        out_dim_count_ptr == out_type_value_ptr ||
        out_dim_count_ptr == out_tensor_offset_ptr ||
        out_dim_count_ptr == out_last_dim_index_ptr ||
        out_dims_cells_ptr == out_required_bytes_ptr ||
        out_dims_cells_ptr == out_type_value_ptr ||
        out_dims_cells_ptr == out_tensor_offset_ptr ||
        out_dims_cells_ptr == out_last_dim_index_ptr ||
        out_required_bytes_ptr == out_type_value_ptr ||
        out_required_bytes_ptr == out_tensor_offset_ptr ||
        out_required_bytes_ptr == out_last_dim_index_ptr ||
        out_type_value_ptr == out_tensor_offset_ptr ||
        out_type_value_ptr == out_last_dim_index_ptr ||
        out_tensor_offset_ptr == out_last_dim_index_ptr)
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

    canonical_dims_bytes = 0;
    canonical_required_bytes = 0;
    canonical_next_cursor = 0;
    span_remaining = 0;

    status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParity(
        buf,
        size,
        cursor,
        &staged_dim_count,
        &staged_dims_cells,
        &staged_required_bytes,
        &staged_type_value,
        &staged_tensor_offset,
        &staged_last_dim_index);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!staged_dim_count)
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS;
    if (staged_dim_count > GGUF_TENSOR_MAX_DIMS)
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS;

    if (staged_dims_cells != staged_dim_count)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_last_dim_index != (U64)(staged_dim_count - 1))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTypeKnown(staged_type_value))
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE;

    if (!GGUFTensorTryMulU64(staged_dims_cells, 8, &canonical_dims_bytes))
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;
    if (!GGUFTensorTryAddU64(4, canonical_dims_bytes, &canonical_required_bytes))
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW;
    if (!GGUFTensorTryAddU64(canonical_required_bytes, 4, &canonical_required_bytes))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTryAddU64(canonical_required_bytes, 8, &canonical_required_bytes))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_required_bytes != canonical_required_bytes)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    span_remaining = size - cursor;
    if (staged_required_bytes > span_remaining)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor, staged_required_bytes, &canonical_next_cursor))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (canonical_next_cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_dim_count = staged_dim_count;
    *out_dims_cells = staged_dims_cells;
    *out_required_bytes = staged_required_bytes;
    *out_type_value = staged_type_value;
    *out_tensor_offset = staged_tensor_offset;
    *out_last_dim_index = staged_last_dim_index;
    return GGUF_TENSOR_PARSE_OK;
}


// IQ-1032 diagnostics-only no-write companion for dims parity-commit wrappers.
//
// Compose `...PreflightOnlyParityCommitOnly` and canonical
// `...CommitOnlyPreflightOnly`, revalidate immutable `{buf,size,cursor}`
// snapshots, enforce exact tuple parity over
// `{dim_count,dims_cells,required_bytes,type_value,tensor_offset,last_dim_index}`,
// and atomically publish diagnostics only under strict parity success.
I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
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

    U32 staged_commit_dim_count;
    U64 staged_commit_dims_cells;
    U64 staged_commit_required_bytes;
    U32 staged_commit_type_value;
    U64 staged_commit_tensor_offset;
    U64 staged_commit_last_dim_index;

    U32 staged_pre_dim_count;
    U64 staged_pre_dims_cells;
    U64 staged_pre_required_bytes;
    U32 staged_pre_type_value;
    U64 staged_pre_tensor_offset;
    U64 staged_pre_last_dim_index;

    U64 staged_commit_computed_end;
    U64 staged_pre_computed_end;

    I64 status;

    U8 *out_dim_count_ptr;
    U8 *out_dims_cells_ptr;
    U8 *out_required_bytes_ptr;
    U8 *out_type_value_ptr;
    U8 *out_tensor_offset_ptr;
    U8 *out_last_dim_index_ptr;

    if (!buf || !out_dim_count || !out_dims_cells || !out_required_bytes)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (!out_type_value || !out_tensor_offset || !out_last_dim_index)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    out_dim_count_ptr = (U8 *)out_dim_count;
    out_dims_cells_ptr = (U8 *)out_dims_cells;
    out_required_bytes_ptr = (U8 *)out_required_bytes;
    out_type_value_ptr = (U8 *)out_type_value;
    out_tensor_offset_ptr = (U8 *)out_tensor_offset;
    out_last_dim_index_ptr = (U8 *)out_last_dim_index;

    if (out_dim_count_ptr == out_dims_cells_ptr ||
        out_dim_count_ptr == out_required_bytes_ptr ||
        out_dim_count_ptr == out_type_value_ptr ||
        out_dim_count_ptr == out_tensor_offset_ptr ||
        out_dim_count_ptr == out_last_dim_index_ptr ||
        out_dims_cells_ptr == out_required_bytes_ptr ||
        out_dims_cells_ptr == out_type_value_ptr ||
        out_dims_cells_ptr == out_tensor_offset_ptr ||
        out_dims_cells_ptr == out_last_dim_index_ptr ||
        out_required_bytes_ptr == out_type_value_ptr ||
        out_required_bytes_ptr == out_tensor_offset_ptr ||
        out_required_bytes_ptr == out_last_dim_index_ptr ||
        out_type_value_ptr == out_tensor_offset_ptr ||
        out_type_value_ptr == out_last_dim_index_ptr ||
        out_tensor_offset_ptr == out_last_dim_index_ptr)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_commit_dim_count = 0;
    staged_commit_dims_cells = 0;
    staged_commit_required_bytes = 0;
    staged_commit_type_value = 0;
    staged_commit_tensor_offset = 0;
    staged_commit_last_dim_index = 0;

    staged_pre_dim_count = 0;
    staged_pre_dims_cells = 0;
    staged_pre_required_bytes = 0;
    staged_pre_type_value = 0;
    staged_pre_tensor_offset = 0;
    staged_pre_last_dim_index = 0;

    staged_commit_computed_end = 0;
    staged_pre_computed_end = 0;

    status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        buf,
        size,
        cursor,
        &staged_commit_dim_count,
        &staged_commit_dims_cells,
        &staged_commit_required_bytes,
        &staged_commit_type_value,
        &staged_commit_tensor_offset,
        &staged_commit_last_dim_index);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnly(
        buf,
        size,
        cursor,
        &staged_pre_dim_count,
        &staged_pre_dims_cells,
        &staged_pre_required_bytes,
        &staged_pre_type_value,
        &staged_pre_tensor_offset,
        &staged_pre_last_dim_index);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_commit_dim_count != staged_pre_dim_count ||
        staged_commit_dims_cells != staged_pre_dims_cells ||
        staged_commit_required_bytes != staged_pre_required_bytes ||
        staged_commit_type_value != staged_pre_type_value ||
        staged_commit_tensor_offset != staged_pre_tensor_offset ||
        staged_commit_last_dim_index != staged_pre_last_dim_index)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!staged_commit_dim_count || !staged_pre_dim_count)
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS;
    if (staged_commit_dim_count > GGUF_TENSOR_MAX_DIMS ||
        staged_pre_dim_count > GGUF_TENSOR_MAX_DIMS)
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS;

    if (staged_commit_dims_cells != staged_commit_dim_count ||
        staged_pre_dims_cells != staged_pre_dim_count)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_commit_last_dim_index != (U64)(staged_commit_dim_count - 1) ||
        staged_pre_last_dim_index != (U64)(staged_pre_dim_count - 1))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTypeKnown(staged_commit_type_value) ||
        !GGUFTensorTypeKnown(staged_pre_type_value))
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE;

    if (staged_commit_required_bytes > size - cursor ||
        staged_pre_required_bytes > size - cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_commit_required_bytes,
                             &staged_commit_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_commit_computed_end > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_pre_required_bytes,
                             &staged_pre_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_pre_computed_end > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_commit_computed_end != staged_pre_computed_end)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_dim_count = staged_commit_dim_count;
    *out_dims_cells = staged_commit_dims_cells;
    *out_required_bytes = staged_commit_required_bytes;
    *out_type_value = staged_commit_type_value;
    *out_tensor_offset = staged_commit_tensor_offset;
    *out_last_dim_index = staged_commit_last_dim_index;
    return GGUF_TENSOR_PARSE_OK;
}


// IQ-1037 diagnostics-only parity gate for dims commit+preflight wrappers.
//
// Compose `...PreflightOnlyParityCommitOnlyPreflightOnly` and canonical
// `...PreflightOnlyParityCommitOnly`, enforce immutable `{buf,size,cursor}`
// snapshots plus exact tuple parity over
// `{dim_count,dims_cells,required_bytes,type_value,tensor_offset,last_dim_index}`
// before atomic publish.
I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
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

    U32 staged_preflight_dim_count;
    U64 staged_preflight_dims_cells;
    U64 staged_preflight_required_bytes;
    U32 staged_preflight_type_value;
    U64 staged_preflight_tensor_offset;
    U64 staged_preflight_last_dim_index;

    U32 staged_commit_dim_count;
    U64 staged_commit_dims_cells;
    U64 staged_commit_required_bytes;
    U32 staged_commit_type_value;
    U64 staged_commit_tensor_offset;
    U64 staged_commit_last_dim_index;

    U64 staged_preflight_computed_end;
    U64 staged_commit_computed_end;

    I64 status;

    U8 *out_dim_count_ptr;
    U8 *out_dims_cells_ptr;
    U8 *out_required_bytes_ptr;
    U8 *out_type_value_ptr;
    U8 *out_tensor_offset_ptr;
    U8 *out_last_dim_index_ptr;

    if (!buf || !out_dim_count || !out_dims_cells || !out_required_bytes)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (!out_type_value || !out_tensor_offset || !out_last_dim_index)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    out_dim_count_ptr = (U8 *)out_dim_count;
    out_dims_cells_ptr = (U8 *)out_dims_cells;
    out_required_bytes_ptr = (U8 *)out_required_bytes;
    out_type_value_ptr = (U8 *)out_type_value;
    out_tensor_offset_ptr = (U8 *)out_tensor_offset;
    out_last_dim_index_ptr = (U8 *)out_last_dim_index;

    if (out_dim_count_ptr == out_dims_cells_ptr ||
        out_dim_count_ptr == out_required_bytes_ptr ||
        out_dim_count_ptr == out_type_value_ptr ||
        out_dim_count_ptr == out_tensor_offset_ptr ||
        out_dim_count_ptr == out_last_dim_index_ptr ||
        out_dims_cells_ptr == out_required_bytes_ptr ||
        out_dims_cells_ptr == out_type_value_ptr ||
        out_dims_cells_ptr == out_tensor_offset_ptr ||
        out_dims_cells_ptr == out_last_dim_index_ptr ||
        out_required_bytes_ptr == out_type_value_ptr ||
        out_required_bytes_ptr == out_tensor_offset_ptr ||
        out_required_bytes_ptr == out_last_dim_index_ptr ||
        out_type_value_ptr == out_tensor_offset_ptr ||
        out_type_value_ptr == out_last_dim_index_ptr ||
        out_tensor_offset_ptr == out_last_dim_index_ptr)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_preflight_dim_count = 0;
    staged_preflight_dims_cells = 0;
    staged_preflight_required_bytes = 0;
    staged_preflight_type_value = 0;
    staged_preflight_tensor_offset = 0;
    staged_preflight_last_dim_index = 0;

    staged_commit_dim_count = 0;
    staged_commit_dims_cells = 0;
    staged_commit_required_bytes = 0;
    staged_commit_type_value = 0;
    staged_commit_tensor_offset = 0;
    staged_commit_last_dim_index = 0;

    staged_preflight_computed_end = 0;
    staged_commit_computed_end = 0;

    status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        buf,
        size,
        cursor,
        &staged_preflight_dim_count,
        &staged_preflight_dims_cells,
        &staged_preflight_required_bytes,
        &staged_preflight_type_value,
        &staged_preflight_tensor_offset,
        &staged_preflight_last_dim_index);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        buf,
        size,
        cursor,
        &staged_commit_dim_count,
        &staged_commit_dims_cells,
        &staged_commit_required_bytes,
        &staged_commit_type_value,
        &staged_commit_tensor_offset,
        &staged_commit_last_dim_index);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_preflight_dim_count != staged_commit_dim_count ||
        staged_preflight_dims_cells != staged_commit_dims_cells ||
        staged_preflight_required_bytes != staged_commit_required_bytes ||
        staged_preflight_type_value != staged_commit_type_value ||
        staged_preflight_tensor_offset != staged_commit_tensor_offset ||
        staged_preflight_last_dim_index != staged_commit_last_dim_index)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!staged_preflight_dim_count || !staged_commit_dim_count)
        return GGUF_TENSOR_PARSE_ERR_BAD_DIM_COUNT;
    if (staged_preflight_dim_count > GGUF_TENSOR_MAX_DIMS ||
        staged_commit_dim_count > GGUF_TENSOR_MAX_DIMS)
        return GGUF_TENSOR_PARSE_ERR_BAD_DIM_COUNT;

    if (staged_preflight_dims_cells != staged_preflight_dim_count ||
        staged_commit_dims_cells != staged_commit_dim_count)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_preflight_last_dim_index != (U64)(staged_preflight_dim_count - 1) ||
        staged_commit_last_dim_index != (U64)(staged_commit_dim_count - 1))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTypeKnown(staged_preflight_type_value) ||
        !GGUFTensorTypeKnown(staged_commit_type_value))
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_preflight_required_bytes,
                             &staged_preflight_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_preflight_computed_end > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_commit_required_bytes,
                             &staged_commit_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_commit_computed_end > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_preflight_computed_end != staged_commit_computed_end)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_dim_count = staged_commit_dim_count;
    *out_dims_cells = staged_commit_dims_cells;
    *out_required_bytes = staged_commit_required_bytes;
    *out_type_value = staged_commit_type_value;
    *out_tensor_offset = staged_commit_tensor_offset;
    *out_last_dim_index = staged_commit_last_dim_index;
    return GGUF_TENSOR_PARSE_OK;
}

// IQ-1038 commit-only diagnostics wrapper for dims parity-preflight gate.
//
// Compose `...PreflightOnlyParityCommitOnlyPreflightOnlyParity` and canonical
// `...PreflightOnlyParityCommitOnlyPreflightOnly`, revalidate immutable
// `{buf,size,cursor}` snapshots and required-bytes span parity, then
// atomically publish only when strict tuple parity holds for
// `{dim_count,dims_cells,required_bytes,type_value,tensor_offset,last_dim_index}`.
I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
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

    U32 staged_parity_dim_count;
    U64 staged_parity_dims_cells;
    U64 staged_parity_required_bytes;
    U32 staged_parity_type_value;
    U64 staged_parity_tensor_offset;
    U64 staged_parity_last_dim_index;

    U32 canonical_dim_count;
    U64 canonical_dims_cells;
    U64 canonical_required_bytes;
    U32 canonical_type_value;
    U64 canonical_tensor_offset;
    U64 canonical_last_dim_index;

    U64 staged_parity_computed_end;
    U64 canonical_computed_end;

    I64 status;

    U8 *out_dim_count_ptr;
    U8 *out_dims_cells_ptr;
    U8 *out_required_bytes_ptr;
    U8 *out_type_value_ptr;
    U8 *out_tensor_offset_ptr;
    U8 *out_last_dim_index_ptr;

    if (!buf || !out_dim_count || !out_dims_cells || !out_required_bytes)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (!out_type_value || !out_tensor_offset || !out_last_dim_index)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    out_dim_count_ptr = (U8 *)out_dim_count;
    out_dims_cells_ptr = (U8 *)out_dims_cells;
    out_required_bytes_ptr = (U8 *)out_required_bytes;
    out_type_value_ptr = (U8 *)out_type_value;
    out_tensor_offset_ptr = (U8 *)out_tensor_offset;
    out_last_dim_index_ptr = (U8 *)out_last_dim_index;

    if (out_dim_count_ptr == out_dims_cells_ptr ||
        out_dim_count_ptr == out_required_bytes_ptr ||
        out_dim_count_ptr == out_type_value_ptr ||
        out_dim_count_ptr == out_tensor_offset_ptr ||
        out_dim_count_ptr == out_last_dim_index_ptr ||
        out_dims_cells_ptr == out_required_bytes_ptr ||
        out_dims_cells_ptr == out_type_value_ptr ||
        out_dims_cells_ptr == out_tensor_offset_ptr ||
        out_dims_cells_ptr == out_last_dim_index_ptr ||
        out_required_bytes_ptr == out_type_value_ptr ||
        out_required_bytes_ptr == out_tensor_offset_ptr ||
        out_required_bytes_ptr == out_last_dim_index_ptr ||
        out_type_value_ptr == out_tensor_offset_ptr ||
        out_type_value_ptr == out_last_dim_index_ptr ||
        out_tensor_offset_ptr == out_last_dim_index_ptr)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_parity_dim_count = 0;
    staged_parity_dims_cells = 0;
    staged_parity_required_bytes = 0;
    staged_parity_type_value = 0;
    staged_parity_tensor_offset = 0;
    staged_parity_last_dim_index = 0;

    canonical_dim_count = 0;
    canonical_dims_cells = 0;
    canonical_required_bytes = 0;
    canonical_type_value = 0;
    canonical_tensor_offset = 0;
    canonical_last_dim_index = 0;

    staged_parity_computed_end = 0;
    canonical_computed_end = 0;

    status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        buf,
        size,
        cursor,
        &staged_parity_dim_count,
        &staged_parity_dims_cells,
        &staged_parity_required_bytes,
        &staged_parity_type_value,
        &staged_parity_tensor_offset,
        &staged_parity_last_dim_index);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        buf,
        size,
        cursor,
        &canonical_dim_count,
        &canonical_dims_cells,
        &canonical_required_bytes,
        &canonical_type_value,
        &canonical_tensor_offset,
        &canonical_last_dim_index);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_parity_required_bytes,
                             &staged_parity_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTryAddU64(cursor,
                             canonical_required_bytes,
                             &canonical_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_parity_computed_end > size ||
        canonical_computed_end > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_parity_dim_count != canonical_dim_count ||
        staged_parity_dims_cells != canonical_dims_cells ||
        staged_parity_required_bytes != canonical_required_bytes ||
        staged_parity_type_value != canonical_type_value ||
        staged_parity_tensor_offset != canonical_tensor_offset ||
        staged_parity_last_dim_index != canonical_last_dim_index)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTypeKnown(staged_parity_type_value) ||
        !GGUFTensorTypeKnown(canonical_type_value))
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE;

    *out_dim_count = staged_parity_dim_count;
    *out_dims_cells = staged_parity_dims_cells;
    *out_required_bytes = staged_parity_required_bytes;
    *out_type_value = staged_parity_type_value;
    *out_tensor_offset = staged_parity_tensor_offset;
    *out_last_dim_index = staged_parity_last_dim_index;
    return GGUF_TENSOR_PARSE_OK;
}

// IQ-1052 diagnostics-only zero-write companion for dims/type/offset parity.
//
// Compose `...ParityCommitOnly` and canonical
// `...ParityCommitOnlyPreflightOnly`, then enforce immutable
// `{buf,size,cursor}` snapshots and strict tuple parity before atomic publish.
I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
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

    U32 staged_commit_dim_count;
    U64 staged_commit_dims_cells;
    U64 staged_commit_required_bytes;
    U32 staged_commit_type_value;
    U64 staged_commit_tensor_offset;
    U64 staged_commit_last_dim_index;

    U32 canonical_dim_count;
    U64 canonical_dims_cells;
    U64 canonical_required_bytes;
    U32 canonical_type_value;
    U64 canonical_tensor_offset;
    U64 canonical_last_dim_index;

    U64 staged_commit_computed_end;
    U64 canonical_computed_end;

    I64 status;

    U8 *out_dim_count_ptr;
    U8 *out_dims_cells_ptr;
    U8 *out_required_bytes_ptr;
    U8 *out_type_value_ptr;
    U8 *out_tensor_offset_ptr;
    U8 *out_last_dim_index_ptr;

    if (!buf || !out_dim_count || !out_dims_cells || !out_required_bytes)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;
    if (!out_type_value || !out_tensor_offset || !out_last_dim_index)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    out_dim_count_ptr = (U8 *)out_dim_count;
    out_dims_cells_ptr = (U8 *)out_dims_cells;
    out_required_bytes_ptr = (U8 *)out_required_bytes;
    out_type_value_ptr = (U8 *)out_type_value;
    out_tensor_offset_ptr = (U8 *)out_tensor_offset;
    out_last_dim_index_ptr = (U8 *)out_last_dim_index;

    if (out_dim_count_ptr == out_dims_cells_ptr ||
        out_dim_count_ptr == out_required_bytes_ptr ||
        out_dim_count_ptr == out_type_value_ptr ||
        out_dim_count_ptr == out_tensor_offset_ptr ||
        out_dim_count_ptr == out_last_dim_index_ptr ||
        out_dims_cells_ptr == out_required_bytes_ptr ||
        out_dims_cells_ptr == out_type_value_ptr ||
        out_dims_cells_ptr == out_tensor_offset_ptr ||
        out_dims_cells_ptr == out_last_dim_index_ptr ||
        out_required_bytes_ptr == out_type_value_ptr ||
        out_required_bytes_ptr == out_tensor_offset_ptr ||
        out_required_bytes_ptr == out_last_dim_index_ptr ||
        out_type_value_ptr == out_tensor_offset_ptr ||
        out_type_value_ptr == out_last_dim_index_ptr ||
        out_tensor_offset_ptr == out_last_dim_index_ptr)
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_commit_dim_count = 0;
    staged_commit_dims_cells = 0;
    staged_commit_required_bytes = 0;
    staged_commit_type_value = 0;
    staged_commit_tensor_offset = 0;
    staged_commit_last_dim_index = 0;

    canonical_dim_count = 0;
    canonical_dims_cells = 0;
    canonical_required_bytes = 0;
    canonical_type_value = 0;
    canonical_tensor_offset = 0;
    canonical_last_dim_index = 0;

    staged_commit_computed_end = 0;
    canonical_computed_end = 0;

    status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
        buf,
        size,
        cursor,
        &staged_commit_dim_count,
        &staged_commit_dims_cells,
        &staged_commit_required_bytes,
        &staged_commit_type_value,
        &staged_commit_tensor_offset,
        &staged_commit_last_dim_index);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        buf,
        size,
        cursor,
        &canonical_dim_count,
        &canonical_dims_cells,
        &canonical_required_bytes,
        &canonical_type_value,
        &canonical_tensor_offset,
        &canonical_last_dim_index);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_commit_required_bytes,
                             &staged_commit_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (!GGUFTensorTryAddU64(cursor,
                             canonical_required_bytes,
                             &canonical_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_commit_computed_end > size ||
        canonical_computed_end > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_commit_dim_count != canonical_dim_count ||
        staged_commit_dims_cells != canonical_dims_cells ||
        staged_commit_required_bytes != canonical_required_bytes ||
        staged_commit_type_value != canonical_type_value ||
        staged_commit_tensor_offset != canonical_tensor_offset ||
        staged_commit_last_dim_index != canonical_last_dim_index)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTypeKnown(staged_commit_type_value) ||
        !GGUFTensorTypeKnown(canonical_type_value))
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE;

    *out_dim_count = staged_commit_dim_count;
    *out_dims_cells = staged_commit_dims_cells;
    *out_required_bytes = staged_commit_required_bytes;
    *out_type_value = staged_commit_type_value;
    *out_tensor_offset = staged_commit_tensor_offset;
    *out_last_dim_index = staged_commit_last_dim_index;
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


// IQ-1015 diagnostics-only parity gate for tensor-info tuple diagnostics.
//
// Compose both:
//   1) preflight-only no-write diagnostics wrapper;
//   2) canonical commit-only diagnostics wrapper.
// Enforce immutable {buf,size,cursor} snapshots and exact tuple parity over
// {name_len, dim_count, dims_cells, required_bytes, type_value,
//  tensor_offset, next_cursor} before atomic output publish.
I64 GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParity(
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

    U64 staged_pre_name_len;
    U32 staged_pre_dim_count;
    U64 staged_pre_dims_cells;
    U64 staged_pre_required_bytes;
    U32 staged_pre_type_value;
    U64 staged_pre_tensor_offset;
    U64 staged_pre_next_cursor;

    U64 staged_commit_name_len;
    U32 staged_commit_dim_count;
    U64 staged_commit_dims_cells;
    U64 staged_commit_required_bytes;
    U32 staged_commit_type_value;
    U64 staged_commit_tensor_offset;
    U64 staged_commit_next_cursor;
    U64 staged_pre_computed_end;
    U64 staged_commit_computed_end;

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

    staged_pre_name_len = 0;
    staged_pre_dim_count = 0;
    staged_pre_dims_cells = 0;
    staged_pre_required_bytes = 0;
    staged_pre_type_value = 0;
    staged_pre_tensor_offset = 0;
    staged_pre_next_cursor = 0;

    staged_commit_name_len = 0;
    staged_commit_dim_count = 0;
    staged_commit_dims_cells = 0;
    staged_commit_required_bytes = 0;
    staged_commit_type_value = 0;
    staged_commit_tensor_offset = 0;
    staged_commit_next_cursor = 0;

    status = GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnly(
        buf,
        size,
        cursor,
        &staged_pre_name_len,
        &staged_pre_dim_count,
        &staged_pre_dims_cells,
        &staged_pre_required_bytes,
        &staged_pre_type_value,
        &staged_pre_tensor_offset,
        &staged_pre_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadCheckedNoPartialCommitOnly(
        buf,
        size,
        cursor,
        &staged_commit_name_len,
        &staged_commit_dim_count,
        &staged_commit_dims_cells,
        &staged_commit_required_bytes,
        &staged_commit_type_value,
        &staged_commit_tensor_offset,
        &staged_commit_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_pre_name_len != staged_commit_name_len ||
        staged_pre_dim_count != staged_commit_dim_count ||
        staged_pre_dims_cells != staged_commit_dims_cells ||
        staged_pre_required_bytes != staged_commit_required_bytes ||
        staged_pre_type_value != staged_commit_type_value ||
        staged_pre_tensor_offset != staged_commit_tensor_offset ||
        staged_pre_next_cursor != staged_commit_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_pre_dims_cells != staged_pre_dim_count ||
        staged_commit_dims_cells != staged_commit_dim_count)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_pre_required_bytes,
                             &staged_pre_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_pre_computed_end != staged_pre_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_commit_required_bytes,
                             &staged_commit_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_commit_computed_end != staged_commit_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_name_len = staged_commit_name_len;
    *out_dim_count = staged_commit_dim_count;
    *out_dims_cells = staged_commit_dims_cells;
    *out_required_bytes = staged_commit_required_bytes;
    *out_type_value = staged_commit_type_value;
    *out_tensor_offset = staged_commit_tensor_offset;
    *out_next_cursor = staged_commit_next_cursor;
    return GGUF_TENSOR_PARSE_OK;
}

// IQ-1018 commit-only diagnostics wrapper for tensor-info parity tuple gates.
//
// Stage diagnostics via ...CommitOnlyPreflightOnlyParity, revalidate immutable
// {buf,size,cursor} snapshots, enforce exact tuple parity against canonical
// ...CommitOnly outputs, and atomically publish only under strict no-partial
// semantics.
I64 GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
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

    U64 staged_parity_name_len;
    U32 staged_parity_dim_count;
    U64 staged_parity_dims_cells;
    U64 staged_parity_required_bytes;
    U32 staged_parity_type_value;
    U64 staged_parity_tensor_offset;
    U64 staged_parity_next_cursor;

    U64 staged_commit_name_len;
    U32 staged_commit_dim_count;
    U64 staged_commit_dims_cells;
    U64 staged_commit_required_bytes;
    U32 staged_commit_type_value;
    U64 staged_commit_tensor_offset;
    U64 staged_commit_next_cursor;
    U64 staged_parity_computed_end;
    U64 staged_commit_computed_end;

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

    staged_parity_name_len = 0;
    staged_parity_dim_count = 0;
    staged_parity_dims_cells = 0;
    staged_parity_required_bytes = 0;
    staged_parity_type_value = 0;
    staged_parity_tensor_offset = 0;
    staged_parity_next_cursor = 0;
    staged_parity_computed_end = 0;

    staged_commit_name_len = 0;
    staged_commit_dim_count = 0;
    staged_commit_dims_cells = 0;
    staged_commit_required_bytes = 0;
    staged_commit_type_value = 0;
    staged_commit_tensor_offset = 0;
    staged_commit_next_cursor = 0;
    staged_commit_computed_end = 0;

    status = GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParity(
        buf,
        size,
        cursor,
        &staged_parity_name_len,
        &staged_parity_dim_count,
        &staged_parity_dims_cells,
        &staged_parity_required_bytes,
        &staged_parity_type_value,
        &staged_parity_tensor_offset,
        &staged_parity_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadCheckedNoPartialCommitOnly(
        buf,
        size,
        cursor,
        &staged_commit_name_len,
        &staged_commit_dim_count,
        &staged_commit_dims_cells,
        &staged_commit_required_bytes,
        &staged_commit_type_value,
        &staged_commit_tensor_offset,
        &staged_commit_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_parity_name_len != staged_commit_name_len ||
        staged_parity_dim_count != staged_commit_dim_count ||
        staged_parity_dims_cells != staged_commit_dims_cells ||
        staged_parity_required_bytes != staged_commit_required_bytes ||
        staged_parity_type_value != staged_commit_type_value ||
        staged_parity_tensor_offset != staged_commit_tensor_offset ||
        staged_parity_next_cursor != staged_commit_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_parity_dims_cells != staged_parity_dim_count ||
        staged_commit_dims_cells != staged_commit_dim_count)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTypeKnown(staged_parity_type_value) ||
        !GGUFTensorTypeKnown(staged_commit_type_value))
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_parity_required_bytes,
                             &staged_parity_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_parity_computed_end != staged_parity_next_cursor ||
        staged_parity_computed_end > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_commit_required_bytes,
                             &staged_commit_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_commit_computed_end != staged_commit_next_cursor ||
        staged_commit_computed_end > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_name_len = staged_commit_name_len;
    *out_dim_count = staged_commit_dim_count;
    *out_dims_cells = staged_commit_dims_cells;
    *out_required_bytes = staged_commit_required_bytes;
    *out_type_value = staged_commit_type_value;
    *out_tensor_offset = staged_commit_tensor_offset;
    *out_next_cursor = staged_commit_next_cursor;
    return GGUF_TENSOR_PARSE_OK;
}

// IQ-1019 diagnostics-only no-write companion for tensor-info commit parity wrappers.
//
// Compose ...PreflightOnlyParityCommitOnly and canonical ...CommitOnlyPreflightOnly,
// revalidate immutable {buf,size,cursor} snapshots, enforce exact tuple parity, and
// atomically publish diagnostics only on strict parity success.
I64 GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
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

    U64 staged_commit_name_len;
    U32 staged_commit_dim_count;
    U64 staged_commit_dims_cells;
    U64 staged_commit_required_bytes;
    U32 staged_commit_type_value;
    U64 staged_commit_tensor_offset;
    U64 staged_commit_next_cursor;

    U64 staged_preflight_name_len;
    U32 staged_preflight_dim_count;
    U64 staged_preflight_dims_cells;
    U64 staged_preflight_required_bytes;
    U32 staged_preflight_type_value;
    U64 staged_preflight_tensor_offset;
    U64 staged_preflight_next_cursor;

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

    if (GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64), out_dim_count_ptr, sizeof(U32)) ||
        GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64), out_dims_cells_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64), out_required_bytes_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64), out_type_value_ptr, sizeof(U32)) ||
        GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64), out_tensor_offset_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64), out_next_cursor_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_dim_count_ptr, sizeof(U32), out_dims_cells_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_dim_count_ptr, sizeof(U32), out_required_bytes_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_dim_count_ptr, sizeof(U32), out_type_value_ptr, sizeof(U32)) ||
        GGUFTensorByteSpansOverlap(out_dim_count_ptr, sizeof(U32), out_tensor_offset_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_dim_count_ptr, sizeof(U32), out_next_cursor_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_dims_cells_ptr, sizeof(U64), out_required_bytes_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_dims_cells_ptr, sizeof(U64), out_type_value_ptr, sizeof(U32)) ||
        GGUFTensorByteSpansOverlap(out_dims_cells_ptr, sizeof(U64), out_tensor_offset_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_dims_cells_ptr, sizeof(U64), out_next_cursor_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_required_bytes_ptr, sizeof(U64), out_type_value_ptr, sizeof(U32)) ||
        GGUFTensorByteSpansOverlap(out_required_bytes_ptr, sizeof(U64), out_tensor_offset_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_required_bytes_ptr, sizeof(U64), out_next_cursor_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_type_value_ptr, sizeof(U32), out_tensor_offset_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_type_value_ptr, sizeof(U32), out_next_cursor_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_tensor_offset_ptr, sizeof(U64), out_next_cursor_ptr, sizeof(U64)))
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_commit_name_len = 0;
    staged_commit_dim_count = 0;
    staged_commit_dims_cells = 0;
    staged_commit_required_bytes = 0;
    staged_commit_type_value = 0;
    staged_commit_tensor_offset = 0;
    staged_commit_next_cursor = 0;

    staged_preflight_name_len = 0;
    staged_preflight_dim_count = 0;
    staged_preflight_dims_cells = 0;
    staged_preflight_required_bytes = 0;
    staged_preflight_type_value = 0;
    staged_preflight_tensor_offset = 0;
    staged_preflight_next_cursor = 0;

    status = GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        buf,
        size,
        cursor,
        &staged_commit_name_len,
        &staged_commit_dim_count,
        &staged_commit_dims_cells,
        &staged_commit_required_bytes,
        &staged_commit_type_value,
        &staged_commit_tensor_offset,
        &staged_commit_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnly(
        buf,
        size,
        cursor,
        &staged_preflight_name_len,
        &staged_preflight_dim_count,
        &staged_preflight_dims_cells,
        &staged_preflight_required_bytes,
        &staged_preflight_type_value,
        &staged_preflight_tensor_offset,
        &staged_preflight_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_commit_name_len != staged_preflight_name_len ||
        staged_commit_dim_count != staged_preflight_dim_count ||
        staged_commit_dims_cells != staged_preflight_dims_cells ||
        staged_commit_required_bytes != staged_preflight_required_bytes ||
        staged_commit_type_value != staged_preflight_type_value ||
        staged_commit_tensor_offset != staged_preflight_tensor_offset ||
        staged_commit_next_cursor != staged_preflight_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_name_len = staged_preflight_name_len;
    *out_dim_count = staged_preflight_dim_count;
    *out_dims_cells = staged_preflight_dims_cells;
    *out_required_bytes = staged_preflight_required_bytes;
    *out_type_value = staged_preflight_type_value;
    *out_tensor_offset = staged_preflight_tensor_offset;
    *out_next_cursor = staged_preflight_next_cursor;
    return GGUF_TENSOR_PARSE_OK;
}

// IQ-1027 diagnostics-only parity gate for tensor-info commit+preflight wrappers.
//
// Compose `...ParityCommitOnlyPreflightOnly` and canonical
// `...ParityCommitOnly`, enforce immutable `{buf,size,cursor}` snapshots,
// revalidate canonical required-span math, and atomically publish only when
// strict tuple parity holds for
// `{name_len,dim_count,dims_cells,required_bytes,type_value,tensor_offset,next_cursor}`.
I64 GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
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

    U64 staged_preflight_name_len;
    U32 staged_preflight_dim_count;
    U64 staged_preflight_dims_cells;
    U64 staged_preflight_required_bytes;
    U32 staged_preflight_type_value;
    U64 staged_preflight_tensor_offset;
    U64 staged_preflight_next_cursor;
    U64 staged_preflight_computed_end;

    U64 staged_commit_name_len;
    U32 staged_commit_dim_count;
    U64 staged_commit_dims_cells;
    U64 staged_commit_required_bytes;
    U32 staged_commit_type_value;
    U64 staged_commit_tensor_offset;
    U64 staged_commit_next_cursor;
    U64 staged_commit_computed_end;

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

    if (GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64), out_dim_count_ptr, sizeof(U32)) ||
        GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64), out_dims_cells_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64), out_required_bytes_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64), out_type_value_ptr, sizeof(U32)) ||
        GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64), out_tensor_offset_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64), out_next_cursor_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_dim_count_ptr, sizeof(U32), out_dims_cells_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_dim_count_ptr, sizeof(U32), out_required_bytes_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_dim_count_ptr, sizeof(U32), out_type_value_ptr, sizeof(U32)) ||
        GGUFTensorByteSpansOverlap(out_dim_count_ptr, sizeof(U32), out_tensor_offset_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_dim_count_ptr, sizeof(U32), out_next_cursor_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_dims_cells_ptr, sizeof(U64), out_required_bytes_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_dims_cells_ptr, sizeof(U64), out_type_value_ptr, sizeof(U32)) ||
        GGUFTensorByteSpansOverlap(out_dims_cells_ptr, sizeof(U64), out_tensor_offset_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_dims_cells_ptr, sizeof(U64), out_next_cursor_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_required_bytes_ptr, sizeof(U64), out_type_value_ptr, sizeof(U32)) ||
        GGUFTensorByteSpansOverlap(out_required_bytes_ptr, sizeof(U64), out_tensor_offset_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_required_bytes_ptr, sizeof(U64), out_next_cursor_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_type_value_ptr, sizeof(U32), out_tensor_offset_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_type_value_ptr, sizeof(U32), out_next_cursor_ptr, sizeof(U64)) ||
        GGUFTensorByteSpansOverlap(out_tensor_offset_ptr, sizeof(U64), out_next_cursor_ptr, sizeof(U64)))
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR;

    if (cursor > size)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (size > 0x7FFFFFFFFFFFFFFF || cursor > 0x7FFFFFFFFFFFFFFF)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    snapshot_buf = buf;
    snapshot_size = size;
    snapshot_cursor = cursor;

    staged_preflight_name_len = 0;
    staged_preflight_dim_count = 0;
    staged_preflight_dims_cells = 0;
    staged_preflight_required_bytes = 0;
    staged_preflight_type_value = 0;
    staged_preflight_tensor_offset = 0;
    staged_preflight_next_cursor = 0;
    staged_preflight_computed_end = 0;

    staged_commit_name_len = 0;
    staged_commit_dim_count = 0;
    staged_commit_dims_cells = 0;
    staged_commit_required_bytes = 0;
    staged_commit_type_value = 0;
    staged_commit_tensor_offset = 0;
    staged_commit_next_cursor = 0;
    staged_commit_computed_end = 0;

    status = GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        buf,
        size,
        cursor,
        &staged_preflight_name_len,
        &staged_preflight_dim_count,
        &staged_preflight_dims_cells,
        &staged_preflight_required_bytes,
        &staged_preflight_type_value,
        &staged_preflight_tensor_offset,
        &staged_preflight_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    status = GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        buf,
        size,
        cursor,
        &staged_commit_name_len,
        &staged_commit_dim_count,
        &staged_commit_dims_cells,
        &staged_commit_required_bytes,
        &staged_commit_type_value,
        &staged_commit_tensor_offset,
        &staged_commit_next_cursor);
    if (status != GGUF_TENSOR_PARSE_OK)
        return status;

    if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_preflight_name_len != staged_commit_name_len ||
        staged_preflight_dim_count != staged_commit_dim_count ||
        staged_preflight_dims_cells != staged_commit_dims_cells ||
        staged_preflight_required_bytes != staged_commit_required_bytes ||
        staged_preflight_type_value != staged_commit_type_value ||
        staged_preflight_tensor_offset != staged_commit_tensor_offset ||
        staged_preflight_next_cursor != staged_commit_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (staged_preflight_dims_cells != staged_preflight_dim_count ||
        staged_commit_dims_cells != staged_commit_dim_count)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTypeKnown(staged_preflight_type_value) ||
        !GGUFTensorTypeKnown(staged_commit_type_value))
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_preflight_required_bytes,
                             &staged_preflight_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_preflight_computed_end > size ||
        staged_preflight_computed_end != staged_preflight_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    if (!GGUFTensorTryAddU64(cursor,
                             staged_commit_required_bytes,
                             &staged_commit_computed_end))
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;
    if (staged_commit_computed_end > size ||
        staged_commit_computed_end != staged_commit_next_cursor)
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED;

    *out_name_len = staged_commit_name_len;
    *out_dim_count = staged_commit_dim_count;
    *out_dims_cells = staged_commit_dims_cells;
    *out_required_bytes = staged_commit_required_bytes;
    *out_type_value = staged_commit_type_value;
    *out_tensor_offset = staged_commit_tensor_offset;
    *out_next_cursor = staged_commit_next_cursor;
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
// IQ1031_MARKER
