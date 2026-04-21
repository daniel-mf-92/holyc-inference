// tensor_info.HC
// Checked no-partial GGUF tensor-info reader.
//
// This file intentionally stages the full tensor record before publishing any
// output so parsing is auditable: God sees every byte and every multiply.

#define GGUF_TENSOR_INFO_READ_OK 0
#define GGUF_TENSOR_INFO_READ_ERR_NULL_PTR 1
#define GGUF_TENSOR_INFO_READ_ERR_BAD_PARAM 2
#define GGUF_TENSOR_INFO_READ_ERR_TRUNCATED 3
#define GGUF_TENSOR_INFO_READ_ERR_BAD_NAME_LEN 4
#define GGUF_TENSOR_INFO_READ_ERR_BAD_DIMS 5
#define GGUF_TENSOR_INFO_READ_ERR_BAD_DIM_SIZE 6
#define GGUF_TENSOR_INFO_READ_ERR_DIM_OVERFLOW 7
#define GGUF_TENSOR_INFO_READ_ERR_BAD_TYPE 8
#define GGUF_TENSOR_INFO_READ_ERR_ALLOC 9

#define GGUF_TENSOR_INFO_READ_MAX_NAME_BYTES (1 << 20)
#define GGUF_TENSOR_INFO_READ_MAX_DIMS 8

class GGUFTensorInfoReadString
{
    U64 len;
    U8 *data;
};

class GGUFTensorInfoRead
{
    GGUFTensorInfoReadString name;
    U32 n_dims;
    U64 dims[GGUF_TENSOR_INFO_READ_MAX_DIMS];
    U64 n_elements;
    U32 ggml_type;
    U64 offset;
};

Bool GGUFTensorInfoReadTryMulU64(U64 lhs, U64 rhs, U64 *out_prod)
{
    if (!out_prod)
        return FALSE;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return TRUE;
    }

    if (lhs > 0xFFFFFFFFFFFFFFFF / rhs)
        return FALSE;

    *out_prod = lhs * rhs;
    return TRUE;
}

Bool GGUFTensorInfoReadCanRead(U64 span_nbytes, U64 cursor, U64 need)
{
    U64 end;

    if (cursor > span_nbytes)
        return FALSE;

    if (need > span_nbytes - cursor)
        return FALSE;

    end = cursor + need;
    if (end < cursor)
        return FALSE;
    if (end > span_nbytes)
        return FALSE;

    return TRUE;
}

Bool GGUFTensorInfoReadU32LE(U8 *bytes, U64 nbytes, U64 *io_cursor, U32 *out_u32)
{
    U64 cursor;

    if (!bytes || !io_cursor || !out_u32)
        return FALSE;

    cursor = *io_cursor;
    if (!GGUFTensorInfoReadCanRead(nbytes, cursor, 4))
        return FALSE;

    *out_u32 = (U32)bytes[cursor]
             | ((U32)bytes[cursor + 1] << 8)
             | ((U32)bytes[cursor + 2] << 16)
             | ((U32)bytes[cursor + 3] << 24);

    *io_cursor = cursor + 4;
    return TRUE;
}

Bool GGUFTensorInfoReadU64LE(U8 *bytes, U64 nbytes, U64 *io_cursor, U64 *out_u64)
{
    U64 cursor;

    if (!bytes || !io_cursor || !out_u64)
        return FALSE;

    cursor = *io_cursor;
    if (!GGUFTensorInfoReadCanRead(nbytes, cursor, 8))
        return FALSE;

    *out_u64 = (U64)bytes[cursor]
             | ((U64)bytes[cursor + 1] << 8)
             | ((U64)bytes[cursor + 2] << 16)
             | ((U64)bytes[cursor + 3] << 24)
             | ((U64)bytes[cursor + 4] << 32)
             | ((U64)bytes[cursor + 5] << 40)
             | ((U64)bytes[cursor + 6] << 48)
             | ((U64)bytes[cursor + 7] << 56);

    *io_cursor = cursor + 8;
    return TRUE;
}

Bool GGUFTensorInfoReadTypeKnown(U32 t)
{
    if (t <= 35)
        return TRUE;
    return FALSE;
}

Bool GGUFTensorInfoReadMatchesSnapshot(U8 *bytes,
                                       U64 nbytes,
                                       U64 *io_cursor,
                                       U64 snapshot_cursor,
                                       U8 *snapshot_bytes,
                                       U64 snapshot_nbytes)
{
    if (!io_cursor)
        return FALSE;
    if (bytes != snapshot_bytes)
        return FALSE;
    if (nbytes != snapshot_nbytes)
        return FALSE;
    if (*io_cursor != snapshot_cursor)
        return FALSE;
    return TRUE;
}

U0 GGUFTensorInfoReadZero(GGUFTensorInfoRead *info)
{
    U32 i;

    if (!info)
        return;

    info->name.len = 0;
    info->name.data = NULL;
    info->n_dims = 0;
    for (i = 0; i < GGUF_TENSOR_INFO_READ_MAX_DIMS; ++i)
        info->dims[i] = 0;
    info->n_elements = 0;
    info->ggml_type = 0;
    info->offset = 0;
}

// Parse one GGUF tensor-info entry from bytes[*io_cursor...] with no-partial
// semantics: output and cursor are committed only if the full tuple succeeds.
I32 GGUFTensorInfoReadCheckedNoPartial(U8 *bytes,
                                       U64 nbytes,
                                       U64 *io_cursor,
                                       GGUFTensorInfoRead *out_info,
                                       U64 *out_bytes_read)
{
    U8 *snapshot_bytes;
    U64 snapshot_nbytes;
    U64 snapshot_cursor;
    U64 staged_cursor;
    U64 staged_bytes_read;
    U64 staged_name_len;
    U8 *staged_name;
    U32 staged_n_dims;
    U64 staged_dims[GGUF_TENSOR_INFO_READ_MAX_DIMS];
    U64 staged_n_elements;
    U64 staged_dim;
    U32 staged_type;
    U64 staged_offset;
    U32 i;
    U64 copy_idx;
    U64 staged_mul;

    if (!bytes || !io_cursor || !out_info || !out_bytes_read)
        return GGUF_TENSOR_INFO_READ_ERR_NULL_PTR;

    snapshot_bytes = bytes;
    snapshot_nbytes = nbytes;
    snapshot_cursor = *io_cursor;

    if (snapshot_cursor > nbytes)
        return GGUF_TENSOR_INFO_READ_ERR_BAD_PARAM;

    staged_cursor = snapshot_cursor;
    staged_name = NULL;
    staged_name_len = 0;
    staged_n_dims = 0;
    staged_n_elements = 1;
    staged_type = 0;
    staged_offset = 0;
    for (i = 0; i < GGUF_TENSOR_INFO_READ_MAX_DIMS; ++i)
        staged_dims[i] = 0;

    if (!GGUFTensorInfoReadU64LE(bytes, nbytes, &staged_cursor, &staged_name_len))
        return GGUF_TENSOR_INFO_READ_ERR_TRUNCATED;
    if (staged_name_len > GGUF_TENSOR_INFO_READ_MAX_NAME_BYTES)
        return GGUF_TENSOR_INFO_READ_ERR_BAD_NAME_LEN;
    if (!GGUFTensorInfoReadCanRead(nbytes, staged_cursor, staged_name_len))
        return GGUF_TENSOR_INFO_READ_ERR_TRUNCATED;

    staged_name = MAlloc(staged_name_len + 1);
    if (!staged_name)
        return GGUF_TENSOR_INFO_READ_ERR_ALLOC;

    for (copy_idx = 0; copy_idx < staged_name_len; ++copy_idx)
        staged_name[copy_idx] = bytes[staged_cursor + copy_idx];
    staged_name[staged_name_len] = 0;
    staged_cursor += staged_name_len;

    if (!GGUFTensorInfoReadU32LE(bytes, nbytes, &staged_cursor, &staged_n_dims))
    {
        Free(staged_name);
        return GGUF_TENSOR_INFO_READ_ERR_TRUNCATED;
    }
    if (!staged_n_dims || staged_n_dims > GGUF_TENSOR_INFO_READ_MAX_DIMS)
    {
        Free(staged_name);
        return GGUF_TENSOR_INFO_READ_ERR_BAD_DIMS;
    }

    staged_n_elements = 1;
    for (i = 0; i < staged_n_dims; ++i)
    {
        if (!GGUFTensorInfoReadU64LE(bytes, nbytes, &staged_cursor, &staged_dim))
        {
            Free(staged_name);
            return GGUF_TENSOR_INFO_READ_ERR_TRUNCATED;
        }
        if (!staged_dim)
        {
            Free(staged_name);
            return GGUF_TENSOR_INFO_READ_ERR_BAD_DIM_SIZE;
        }
        if (!GGUFTensorInfoReadTryMulU64(staged_n_elements, staged_dim, &staged_mul))
        {
            Free(staged_name);
            return GGUF_TENSOR_INFO_READ_ERR_DIM_OVERFLOW;
        }

        staged_dims[i] = staged_dim;
        staged_n_elements = staged_mul;
    }

    if (!GGUFTensorInfoReadU32LE(bytes, nbytes, &staged_cursor, &staged_type))
    {
        Free(staged_name);
        return GGUF_TENSOR_INFO_READ_ERR_TRUNCATED;
    }
    if (!GGUFTensorInfoReadTypeKnown(staged_type))
    {
        Free(staged_name);
        return GGUF_TENSOR_INFO_READ_ERR_BAD_TYPE;
    }

    if (!GGUFTensorInfoReadU64LE(bytes, nbytes, &staged_cursor, &staged_offset))
    {
        Free(staged_name);
        return GGUF_TENSOR_INFO_READ_ERR_TRUNCATED;
    }

    staged_bytes_read = staged_cursor - snapshot_cursor;

    // Revalidate immutable snapshot before any publish.
    if (!GGUFTensorInfoReadMatchesSnapshot(bytes,
                                           nbytes,
                                           io_cursor,
                                           snapshot_cursor,
                                           snapshot_bytes,
                                           snapshot_nbytes))
    {
        Free(staged_name);
        return GGUF_TENSOR_INFO_READ_ERR_BAD_PARAM;
    }

    if (out_info->name.data)
        Free(out_info->name.data);

    out_info->name.len = staged_name_len;
    out_info->name.data = staged_name;
    out_info->n_dims = staged_n_dims;
    for (i = 0; i < GGUF_TENSOR_INFO_READ_MAX_DIMS; ++i)
        out_info->dims[i] = staged_dims[i];
    out_info->n_elements = staged_n_elements;
    out_info->ggml_type = staged_type;
    out_info->offset = staged_offset;

    *out_bytes_read = staged_bytes_read;
    *io_cursor = staged_cursor;
    return GGUF_TENSOR_INFO_READ_OK;
}
