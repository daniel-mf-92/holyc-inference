// reader.HC
// Checked GGUF tensor payload reader helpers for HolyC inference runtime.

#define GGUF_READER_OK 0
#define GGUF_READER_ERR_NULL_PTR 1
#define GGUF_READER_ERR_BAD_PARAM 2
#define GGUF_READER_ERR_OVERFLOW 3
#define GGUF_READER_ERR_TRUNCATED 4
#define GGUF_READER_ERR_BAD_TYPE 5
#define GGUF_READER_ERR_BAD_DIM 6
#define GGUF_READER_ERR_BAD_CAPACITY 7

#define GGUF_READER_GGML_TYPE_Q4_0 2
#define GGUF_READER_GGML_TYPE_Q8_0 8
#define GGUF_READER_Q4_0_BLOCK_SIZE 32
#define GGUF_READER_Q4_0_BLOCK_BYTES 18
#define GGUF_READER_Q8_0_BLOCK_SIZE 32
#define GGUF_READER_Q8_0_BLOCK_BYTES 34

class GGUFReaderTensorInfoQ4_0
{
    U32 n_dims;
    U64 dims[4];
    U32 ggml_type;
    U64 offset;
};

Bool GGUFReaderTryAddU64(U64 lhs, U64 rhs, U64 *out_sum)
{
    if (!out_sum)
        return FALSE;
    if (lhs > 0xFFFFFFFFFFFFFFFF - rhs)
        return FALSE;
    *out_sum = lhs + rhs;
    return TRUE;
}

Bool GGUFReaderTryMulU64(U64 lhs, U64 rhs, U64 *out_prod)
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

Bool GGUFReaderTensorInfoQ4_0MatchesSnapshot(
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U32 staged_n_dims,
    U32 staged_ggml_type,
    U64 staged_offset,
    U64 *staged_dims)
{
    U32 i;

    if (!tensor_info || !staged_dims)
        return FALSE;

    if (tensor_info->n_dims != staged_n_dims)
        return FALSE;
    if (tensor_info->ggml_type != staged_ggml_type)
        return FALSE;
    if (tensor_info->offset != staged_offset)
        return FALSE;

    for (i = 0; i < 4; ++i)
    {
        if (tensor_info->dims[i] != staged_dims[i])
            return FALSE;
    }

    return TRUE;
}

Bool GGUFReaderTensorInfoQ8_0MatchesSnapshot(
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U32 staged_n_dims,
    U32 staged_ggml_type,
    U64 staged_offset,
    U64 *staged_dims)
{
    // Q8_0 and Q4_0 tensor metadata share the same shape/type container;
    // keep a dedicated helper name so Q8_0 readers remain explicit/auditable.
    return GGUFReaderTensorInfoQ4_0MatchesSnapshot(tensor_info,
                                                   staged_n_dims,
                                                   staged_ggml_type,
                                                   staged_offset,
                                                   staged_dims);
}

I32 GGUFReadTensorDataQ4_0CheckedNoPartial(
    U8 *gguf_bytes,
    U64 gguf_nbytes,
    U64 tensor_data_base,
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U8 *out_q4_0_blocks,
    U64 out_q4_0_blocks_capacity,
    I64 *out_blocks_read,
    I64 *out_bytes_read)
{
    U32 i;
    U32 staged_n_dims;
    U32 staged_ggml_type;
    U64 staged_dims[4];
    U64 staged_offset;
    U64 staged_element_count;
    U64 staged_block_count;
    U64 staged_required_bytes;
    U64 staged_payload_start;
    U64 staged_payload_end;
    U64 staged_out_end;
    I64 staged_blocks_read_i64;
    I64 staged_bytes_read_i64;
    U64 copy_idx;

    if (!gguf_bytes || !tensor_info || !out_q4_0_blocks ||
        !out_blocks_read || !out_bytes_read)
        return GGUF_READER_ERR_NULL_PTR;

    // Snapshot immutable tensor geometry before any math/copy so all publish
    // happens from one consistent view.
    staged_n_dims = tensor_info->n_dims;
    staged_ggml_type = tensor_info->ggml_type;
    staged_offset = tensor_info->offset;
    for (i = 0; i < 4; ++i)
        staged_dims[i] = tensor_info->dims[i];

    if (!staged_n_dims || staged_n_dims > 4)
        return GGUF_READER_ERR_BAD_DIM;
    if (staged_ggml_type != GGUF_READER_GGML_TYPE_Q4_0)
        return GGUF_READER_ERR_BAD_TYPE;

    staged_element_count = 1;
    for (i = 0; i < staged_n_dims; ++i)
    {
        if (!staged_dims[i])
            return GGUF_READER_ERR_BAD_DIM;
        if (!GGUFReaderTryMulU64(staged_element_count,
                                 staged_dims[i],
                                 &staged_element_count))
            return GGUF_READER_ERR_OVERFLOW;
    }

    if (staged_element_count % GGUF_READER_Q4_0_BLOCK_SIZE)
        return GGUF_READER_ERR_BAD_DIM;

    staged_block_count = staged_element_count / GGUF_READER_Q4_0_BLOCK_SIZE;
    if (!GGUFReaderTryMulU64(staged_block_count,
                             GGUF_READER_Q4_0_BLOCK_BYTES,
                             &staged_required_bytes))
        return GGUF_READER_ERR_OVERFLOW;

    if (!GGUFReaderTryAddU64(tensor_data_base,
                             staged_offset,
                             &staged_payload_start))
        return GGUF_READER_ERR_OVERFLOW;
    if (!GGUFReaderTryAddU64(staged_payload_start,
                             staged_required_bytes,
                             &staged_payload_end))
        return GGUF_READER_ERR_OVERFLOW;

    if (staged_payload_start > gguf_nbytes)
        return GGUF_READER_ERR_TRUNCATED;
    if (staged_payload_end > gguf_nbytes)
        return GGUF_READER_ERR_TRUNCATED;

    if (!GGUFReaderTryAddU64(0,
                             staged_required_bytes,
                             &staged_out_end))
        return GGUF_READER_ERR_OVERFLOW;
    if (staged_out_end > out_q4_0_blocks_capacity)
        return GGUF_READER_ERR_BAD_CAPACITY;

    if (!GGUFReaderTensorInfoQ4_0MatchesSnapshot(tensor_info,
                                                 staged_n_dims,
                                                 staged_ggml_type,
                                                 staged_offset,
                                                 staged_dims))
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_block_count > 0x7FFFFFFFFFFFFFFF)
        return GGUF_READER_ERR_OVERFLOW;
    if (staged_required_bytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_READER_ERR_OVERFLOW;

    staged_blocks_read_i64 = (I64)staged_block_count;
    staged_bytes_read_i64 = (I64)staged_required_bytes;

    // No-partial semantics: all bounds/capacity checks complete before copy.
    for (copy_idx = 0; copy_idx < staged_required_bytes; ++copy_idx)
        out_q4_0_blocks[copy_idx] = gguf_bytes[staged_payload_start + copy_idx];

    // Atomic diagnostics publish: commit both counters from staged locals.
    *out_blocks_read = staged_blocks_read_i64;
    *out_bytes_read = staged_bytes_read_i64;
    return GGUF_READER_OK;
}

I32 GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnly(
    U8 *gguf_bytes,
    U64 gguf_nbytes,
    U64 tensor_data_base,
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U8 *out_q4_0_blocks,
    U64 out_q4_0_blocks_capacity,
    I64 *out_blocks_read,
    I64 *out_bytes_read)
{
    U32 i;
    U32 staged_n_dims;
    U32 staged_ggml_type;
    U64 staged_dims[4];
    U64 staged_offset;
    I64 staged_blocks_read_i64;
    I64 staged_bytes_read_i64;
    I32 status;

    if (!gguf_bytes || !tensor_info || !out_q4_0_blocks ||
        !out_blocks_read || !out_bytes_read)
        return GGUF_READER_ERR_NULL_PTR;

    if (out_blocks_read == out_bytes_read)
        return GGUF_READER_ERR_BAD_PARAM;

    if ((U8 *)out_blocks_read == out_q4_0_blocks ||
        (U8 *)out_bytes_read == out_q4_0_blocks)
        return GGUF_READER_ERR_BAD_PARAM;

    // Keep diagnostics outputs disjoint from tensor metadata storage so
    // snapshot revalidation cannot be corrupted by aliasing writes.
    if ((U8 *)out_blocks_read == (U8 *)tensor_info ||
        (U8 *)out_bytes_read == (U8 *)tensor_info)
        return GGUF_READER_ERR_BAD_PARAM;

    // Snapshot immutable tensor metadata so diagnostics publish only if the
    // caller did not mutate geometry while the checked read ran.
    staged_n_dims = tensor_info->n_dims;
    staged_ggml_type = tensor_info->ggml_type;
    staged_offset = tensor_info->offset;
    for (i = 0; i < 4; ++i)
        staged_dims[i] = tensor_info->dims[i];

    status = GGUFReadTensorDataQ4_0CheckedNoPartial(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q4_0_blocks,
        out_q4_0_blocks_capacity,
        &staged_blocks_read_i64,
        &staged_bytes_read_i64);
    if (status != GGUF_READER_OK)
        return status;

    if (!GGUFReaderTensorInfoQ4_0MatchesSnapshot(tensor_info,
                                                 staged_n_dims,
                                                 staged_ggml_type,
                                                 staged_offset,
                                                 staged_dims))
        return GGUF_READER_ERR_BAD_PARAM;

    // Commit-only diagnostics publish.
    *out_blocks_read = staged_blocks_read_i64;
    *out_bytes_read = staged_bytes_read_i64;
    return GGUF_READER_OK;
}

I32 GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnly(
    U8 *gguf_bytes,
    U64 gguf_nbytes,
    U64 tensor_data_base,
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U8 *out_q4_0_blocks,
    U64 out_q4_0_blocks_capacity,
    I64 *out_blocks_read,
    I64 *out_bytes_read)
{
    U32 i;
    U32 staged_n_dims;
    U32 staged_ggml_type;
    U64 staged_dims[4];
    U64 staged_offset;
    U64 staged_element_count;
    U64 staged_block_count;
    U64 staged_required_bytes;
    U64 staged_payload_start;
    U64 staged_payload_end;
    U64 staged_out_end;
    I64 staged_blocks_read_i64;
    I64 staged_bytes_read_i64;

    if (!gguf_bytes || !tensor_info || !out_q4_0_blocks ||
        !out_blocks_read || !out_bytes_read)
        return GGUF_READER_ERR_NULL_PTR;

    // Snapshot immutable tensor geometry first, then derive diagnostics from
    // this single view so publish is atomic and deterministic.
    staged_n_dims = tensor_info->n_dims;
    staged_ggml_type = tensor_info->ggml_type;
    staged_offset = tensor_info->offset;
    for (i = 0; i < 4; ++i)
        staged_dims[i] = tensor_info->dims[i];

    if (!staged_n_dims || staged_n_dims > 4)
        return GGUF_READER_ERR_BAD_DIM;
    if (staged_ggml_type != GGUF_READER_GGML_TYPE_Q4_0)
        return GGUF_READER_ERR_BAD_TYPE;

    staged_element_count = 1;
    for (i = 0; i < staged_n_dims; ++i)
    {
        if (!staged_dims[i])
            return GGUF_READER_ERR_BAD_DIM;
        if (!GGUFReaderTryMulU64(staged_element_count,
                                 staged_dims[i],
                                 &staged_element_count))
            return GGUF_READER_ERR_OVERFLOW;
    }

    // Q4_0 stores 32 quantized elements per block.
    if (staged_element_count % GGUF_READER_Q4_0_BLOCK_SIZE)
        return GGUF_READER_ERR_BAD_DIM;

    staged_block_count = staged_element_count / GGUF_READER_Q4_0_BLOCK_SIZE;
    if (!GGUFReaderTryMulU64(staged_block_count,
                             GGUF_READER_Q4_0_BLOCK_BYTES,
                             &staged_required_bytes))
        return GGUF_READER_ERR_OVERFLOW;

    if (!GGUFReaderTryAddU64(tensor_data_base,
                             staged_offset,
                             &staged_payload_start))
        return GGUF_READER_ERR_OVERFLOW;
    if (!GGUFReaderTryAddU64(staged_payload_start,
                             staged_required_bytes,
                             &staged_payload_end))
        return GGUF_READER_ERR_OVERFLOW;

    if (staged_payload_start > gguf_nbytes)
        return GGUF_READER_ERR_TRUNCATED;
    if (staged_payload_end > gguf_nbytes)
        return GGUF_READER_ERR_TRUNCATED;

    if (!GGUFReaderTryAddU64(0,
                             staged_required_bytes,
                             &staged_out_end))
        return GGUF_READER_ERR_OVERFLOW;
    if (staged_out_end > out_q4_0_blocks_capacity)
        return GGUF_READER_ERR_BAD_CAPACITY;

    if (!GGUFReaderTensorInfoQ4_0MatchesSnapshot(tensor_info,
                                                 staged_n_dims,
                                                 staged_ggml_type,
                                                 staged_offset,
                                                 staged_dims))
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_block_count > 0x7FFFFFFFFFFFFFFF)
        return GGUF_READER_ERR_OVERFLOW;
    if (staged_required_bytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_READER_ERR_OVERFLOW;

    staged_blocks_read_i64 = (I64)staged_block_count;
    staged_bytes_read_i64 = (I64)staged_required_bytes;

    // Preflight-only: never writes out_q4_0_blocks, only diagnostics.
    *out_blocks_read = staged_blocks_read_i64;
    *out_bytes_read = staged_bytes_read_i64;
    return GGUF_READER_OK;
}

I32 GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnlyParity(
    U8 *gguf_bytes,
    U64 gguf_nbytes,
    U64 tensor_data_base,
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U8 *out_q4_0_blocks,
    U64 out_q4_0_blocks_capacity,
    I64 *out_blocks_read,
    I64 *out_bytes_read)
{
    U32 i;
    U32 staged_n_dims;
    U32 staged_ggml_type;
    U64 staged_dims[4];
    U64 staged_offset;
    U64 staged_out_capacity;
    I64 staged_blocks_read_i64;
    I64 staged_bytes_read_i64;
    U64 recomputed_element_count;
    U64 recomputed_block_count;
    U64 recomputed_required_bytes;
    U64 recomputed_payload_start;
    U64 recomputed_payload_end;
    U64 recomputed_out_end;
    I64 recomputed_blocks_read_i64;
    I64 recomputed_bytes_read_i64;
    I32 status;

    if (!gguf_bytes || !tensor_info || !out_q4_0_blocks ||
        !out_blocks_read || !out_bytes_read)
        return GGUF_READER_ERR_NULL_PTR;

    if (out_blocks_read == out_bytes_read)
        return GGUF_READER_ERR_BAD_PARAM;

    if ((U8 *)out_blocks_read == out_q4_0_blocks ||
        (U8 *)out_bytes_read == out_q4_0_blocks)
        return GGUF_READER_ERR_BAD_PARAM;

    // Keep diagnostics outputs disjoint from tensor metadata storage so
    // snapshot revalidation cannot be corrupted by aliasing writes.
    if ((U8 *)out_blocks_read == (U8 *)tensor_info ||
        (U8 *)out_bytes_read == (U8 *)tensor_info)
        return GGUF_READER_ERR_BAD_PARAM;

    // Snapshot immutable tensor metadata for caller mutation detection.
    staged_n_dims = tensor_info->n_dims;
    staged_ggml_type = tensor_info->ggml_type;
    staged_offset = tensor_info->offset;
    staged_out_capacity = out_q4_0_blocks_capacity;
    for (i = 0; i < 4; ++i)
        staged_dims[i] = tensor_info->dims[i];

    // Stage canonical diagnostics from the no-write preflight helper.
    status = GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnly(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q4_0_blocks,
        out_q4_0_blocks_capacity,
        &staged_blocks_read_i64,
        &staged_bytes_read_i64);
    if (status != GGUF_READER_OK)
        return status;

    // Recompute canonical checked-read geometry from immutable snapshots.
    // This enforces exact tuple parity under strict zero-write semantics.
    if (!staged_n_dims || staged_n_dims > 4)
        return GGUF_READER_ERR_BAD_DIM;
    if (staged_ggml_type != GGUF_READER_GGML_TYPE_Q4_0)
        return GGUF_READER_ERR_BAD_TYPE;

    recomputed_element_count = 1;
    for (i = 0; i < staged_n_dims; ++i)
    {
        if (!staged_dims[i])
            return GGUF_READER_ERR_BAD_DIM;
        if (!GGUFReaderTryMulU64(recomputed_element_count,
                                 staged_dims[i],
                                 &recomputed_element_count))
            return GGUF_READER_ERR_OVERFLOW;
    }

    if (recomputed_element_count % GGUF_READER_Q4_0_BLOCK_SIZE)
        return GGUF_READER_ERR_BAD_DIM;

    recomputed_block_count =
        recomputed_element_count / GGUF_READER_Q4_0_BLOCK_SIZE;
    if (!GGUFReaderTryMulU64(recomputed_block_count,
                             GGUF_READER_Q4_0_BLOCK_BYTES,
                             &recomputed_required_bytes))
        return GGUF_READER_ERR_OVERFLOW;

    if (!GGUFReaderTryAddU64(tensor_data_base,
                             staged_offset,
                             &recomputed_payload_start))
        return GGUF_READER_ERR_OVERFLOW;
    if (!GGUFReaderTryAddU64(recomputed_payload_start,
                             recomputed_required_bytes,
                             &recomputed_payload_end))
        return GGUF_READER_ERR_OVERFLOW;

    if (recomputed_payload_start > gguf_nbytes)
        return GGUF_READER_ERR_TRUNCATED;
    if (recomputed_payload_end > gguf_nbytes)
        return GGUF_READER_ERR_TRUNCATED;

    if (!GGUFReaderTryAddU64(0,
                             recomputed_required_bytes,
                             &recomputed_out_end))
        return GGUF_READER_ERR_OVERFLOW;
    if (recomputed_out_end > staged_out_capacity)
        return GGUF_READER_ERR_BAD_CAPACITY;

    if (recomputed_block_count > 0x7FFFFFFFFFFFFFFF)
        return GGUF_READER_ERR_OVERFLOW;
    if (recomputed_required_bytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_READER_ERR_OVERFLOW;

    recomputed_blocks_read_i64 = (I64)recomputed_block_count;
    recomputed_bytes_read_i64 = (I64)recomputed_required_bytes;

    if (!GGUFReaderTensorInfoQ4_0MatchesSnapshot(tensor_info,
                                                 staged_n_dims,
                                                 staged_ggml_type,
                                                 staged_offset,
                                                 staged_dims))
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_out_capacity != out_q4_0_blocks_capacity)
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_blocks_read_i64 != recomputed_blocks_read_i64 ||
        staged_bytes_read_i64 != recomputed_bytes_read_i64)
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_blocks_read_i64 < 0 || staged_bytes_read_i64 < 0)
        return GGUF_READER_ERR_OVERFLOW;
    if ((U64)staged_bytes_read_i64 > staged_out_capacity)
        return GGUF_READER_ERR_BAD_CAPACITY;

    // Atomic diagnostics publish only after exact tuple parity succeeds.
    *out_blocks_read = staged_blocks_read_i64;
    *out_bytes_read = staged_bytes_read_i64;
    return GGUF_READER_OK;
}


I32 GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
    U8 *gguf_bytes,
    U64 gguf_nbytes,
    U64 tensor_data_base,
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U8 *out_q4_0_blocks,
    U64 out_q4_0_blocks_capacity,
    I64 *out_blocks_read,
    I64 *out_bytes_read)
{
    U32 i;
    U32 staged_n_dims;
    U32 staged_ggml_type;
    U64 staged_dims[4];
    U64 staged_offset;
    U64 staged_out_capacity;
    I64 staged_blocks_read_i64;
    I64 staged_bytes_read_i64;
    I32 status;

    if (!gguf_bytes || !tensor_info || !out_q4_0_blocks ||
        !out_blocks_read || !out_bytes_read)
        return GGUF_READER_ERR_NULL_PTR;

    if (out_blocks_read == out_bytes_read)
        return GGUF_READER_ERR_BAD_PARAM;

    if ((U8 *)out_blocks_read == out_q4_0_blocks ||
        (U8 *)out_bytes_read == out_q4_0_blocks)
        return GGUF_READER_ERR_BAD_PARAM;

    // Snapshot immutable tensor metadata and caller-reported output capacity
    // before staging diagnostics to enforce strict commit-only semantics.
    staged_n_dims = tensor_info->n_dims;
    staged_ggml_type = tensor_info->ggml_type;
    staged_offset = tensor_info->offset;
    staged_out_capacity = out_q4_0_blocks_capacity;
    for (i = 0; i < 4; ++i)
        staged_dims[i] = tensor_info->dims[i];

    status = GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnlyParity(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q4_0_blocks,
        out_q4_0_blocks_capacity,
        &staged_blocks_read_i64,
        &staged_bytes_read_i64);
    if (status != GGUF_READER_OK)
        return status;

    if (!GGUFReaderTensorInfoQ4_0MatchesSnapshot(tensor_info,
                                                 staged_n_dims,
                                                 staged_ggml_type,
                                                 staged_offset,
                                                 staged_dims))
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_out_capacity != out_q4_0_blocks_capacity)
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_blocks_read_i64 < 0 || staged_bytes_read_i64 < 0)
        return GGUF_READER_ERR_OVERFLOW;

    // Capacity revalidation on staged diagnostics before atomic publish.
    if ((U64)staged_bytes_read_i64 > staged_out_capacity)
        return GGUF_READER_ERR_BAD_CAPACITY;

    *out_blocks_read = staged_blocks_read_i64;
    *out_bytes_read = staged_bytes_read_i64;
    return GGUF_READER_OK;
}


I32 GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    U8 *gguf_bytes,
    U64 gguf_nbytes,
    U64 tensor_data_base,
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U8 *out_q4_0_blocks,
    U64 out_q4_0_blocks_capacity,
    I64 *out_blocks_read,
    I64 *out_bytes_read)
{
    U32 i;
    U32 staged_n_dims;
    U32 staged_ggml_type;
    U64 staged_dims[4];
    U64 staged_offset;
    U64 staged_out_capacity;
    I64 staged_blocks_read_i64;
    I64 staged_bytes_read_i64;
    I64 recomputed_blocks_read_i64;
    I64 recomputed_bytes_read_i64;
    I32 status;

    if (!gguf_bytes || !tensor_info || !out_q4_0_blocks ||
        !out_blocks_read || !out_bytes_read)
        return GGUF_READER_ERR_NULL_PTR;

    if (out_blocks_read == out_bytes_read)
        return GGUF_READER_ERR_BAD_PARAM;

    if ((U8 *)out_blocks_read == out_q4_0_blocks ||
        (U8 *)out_bytes_read == out_q4_0_blocks)
        return GGUF_READER_ERR_BAD_PARAM;

    // Snapshot immutable tensor metadata and caller-reported capacity so this
    // diagnostics-only wrapper can reject caller mutation before publish.
    staged_n_dims = tensor_info->n_dims;
    staged_ggml_type = tensor_info->ggml_type;
    staged_offset = tensor_info->offset;
    staged_out_capacity = out_q4_0_blocks_capacity;
    for (i = 0; i < 4; ++i)
        staged_dims[i] = tensor_info->dims[i];

    // Stage diagnostics from commit-only parity wrapper.
    status = GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q4_0_blocks,
        out_q4_0_blocks_capacity,
        &staged_blocks_read_i64,
        &staged_bytes_read_i64);
    if (status != GGUF_READER_OK)
        return status;

    // Recompute canonical diagnostics from the no-write preflight helper and
    // require exact tuple parity before atomic publish.
    status = GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnly(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q4_0_blocks,
        out_q4_0_blocks_capacity,
        &recomputed_blocks_read_i64,
        &recomputed_bytes_read_i64);
    if (status != GGUF_READER_OK)
        return status;

    if (!GGUFReaderTensorInfoQ4_0MatchesSnapshot(tensor_info,
                                                 staged_n_dims,
                                                 staged_ggml_type,
                                                 staged_offset,
                                                 staged_dims))
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_out_capacity != out_q4_0_blocks_capacity)
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_blocks_read_i64 != recomputed_blocks_read_i64 ||
        staged_bytes_read_i64 != recomputed_bytes_read_i64)
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_blocks_read_i64 < 0 || staged_bytes_read_i64 < 0)
        return GGUF_READER_ERR_OVERFLOW;

    if ((U64)staged_bytes_read_i64 > staged_out_capacity)
        return GGUF_READER_ERR_BAD_CAPACITY;

    // Preflight-only semantics: publish diagnostics only; payload stays
    // untouched because both composed helpers are no-write variants.
    *out_blocks_read = staged_blocks_read_i64;
    *out_bytes_read = staged_bytes_read_i64;
    return GGUF_READER_OK;
}


I32 GGUFReadTensorDataQ8_0CheckedNoPartial(
    U8 *gguf_bytes,
    U64 gguf_nbytes,
    U64 tensor_data_base,
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U8 *out_q8_0_blocks,
    U64 out_q8_0_blocks_capacity,
    I64 *out_blocks_read,
    I64 *out_bytes_read)
{
    U32 i;
    U32 staged_n_dims;
    U32 staged_ggml_type;
    U64 staged_dims[4];
    U64 staged_offset;
    U64 staged_element_count;
    U64 staged_block_count;
    U64 staged_required_bytes;
    U64 staged_payload_start;
    U64 staged_payload_end;
    U64 staged_out_end;
    I64 staged_blocks_read_i64;
    I64 staged_bytes_read_i64;
    U64 copy_idx;

    if (!gguf_bytes || !tensor_info || !out_q8_0_blocks ||
        !out_blocks_read || !out_bytes_read)
        return GGUF_READER_ERR_NULL_PTR;

    // Snapshot immutable tensor geometry before any math/copy so all publish
    // happens from one consistent view.
    staged_n_dims = tensor_info->n_dims;
    staged_ggml_type = tensor_info->ggml_type;
    staged_offset = tensor_info->offset;
    for (i = 0; i < 4; ++i)
        staged_dims[i] = tensor_info->dims[i];

    if (!staged_n_dims || staged_n_dims > 4)
        return GGUF_READER_ERR_BAD_DIM;
    if (staged_ggml_type != GGUF_READER_GGML_TYPE_Q8_0)
        return GGUF_READER_ERR_BAD_TYPE;

    staged_element_count = 1;
    for (i = 0; i < staged_n_dims; ++i)
    {
        if (!staged_dims[i])
            return GGUF_READER_ERR_BAD_DIM;
        if (!GGUFReaderTryMulU64(staged_element_count,
                                 staged_dims[i],
                                 &staged_element_count))
            return GGUF_READER_ERR_OVERFLOW;
    }

    if (staged_element_count % GGUF_READER_Q8_0_BLOCK_SIZE)
        return GGUF_READER_ERR_BAD_DIM;

    staged_block_count = staged_element_count / GGUF_READER_Q8_0_BLOCK_SIZE;
    if (!GGUFReaderTryMulU64(staged_block_count,
                             GGUF_READER_Q8_0_BLOCK_BYTES,
                             &staged_required_bytes))
        return GGUF_READER_ERR_OVERFLOW;

    if (!GGUFReaderTryAddU64(tensor_data_base,
                             staged_offset,
                             &staged_payload_start))
        return GGUF_READER_ERR_OVERFLOW;
    if (!GGUFReaderTryAddU64(staged_payload_start,
                             staged_required_bytes,
                             &staged_payload_end))
        return GGUF_READER_ERR_OVERFLOW;

    if (staged_payload_start > gguf_nbytes)
        return GGUF_READER_ERR_TRUNCATED;
    if (staged_payload_end > gguf_nbytes)
        return GGUF_READER_ERR_TRUNCATED;

    if (!GGUFReaderTryAddU64(0,
                             staged_required_bytes,
                             &staged_out_end))
        return GGUF_READER_ERR_OVERFLOW;
    if (staged_out_end > out_q8_0_blocks_capacity)
        return GGUF_READER_ERR_BAD_CAPACITY;

    if (!GGUFReaderTensorInfoQ8_0MatchesSnapshot(tensor_info,
                                                 staged_n_dims,
                                                 staged_ggml_type,
                                                 staged_offset,
                                                 staged_dims))
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_block_count > 0x7FFFFFFFFFFFFFFF)
        return GGUF_READER_ERR_OVERFLOW;
    if (staged_required_bytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_READER_ERR_OVERFLOW;

    staged_blocks_read_i64 = (I64)staged_block_count;
    staged_bytes_read_i64 = (I64)staged_required_bytes;

    // No-partial semantics: all bounds/capacity checks complete before copy.
    for (copy_idx = 0; copy_idx < staged_required_bytes; ++copy_idx)
        out_q8_0_blocks[copy_idx] = gguf_bytes[staged_payload_start + copy_idx];

    // Atomic diagnostics publish: commit both counters from staged locals.
    *out_blocks_read = staged_blocks_read_i64;
    *out_bytes_read = staged_bytes_read_i64;
    return GGUF_READER_OK;
}

I32 GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnly(
    U8 *gguf_bytes,
    U64 gguf_nbytes,
    U64 tensor_data_base,
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U8 *out_q8_0_blocks,
    U64 out_q8_0_blocks_capacity,
    I64 *out_blocks_read,
    I64 *out_bytes_read)
{
    U32 i;
    U32 staged_n_dims;
    U32 staged_ggml_type;
    U64 staged_dims[4];
    U64 staged_offset;
    I64 staged_blocks_read_i64;
    I64 staged_bytes_read_i64;
    I32 status;

    if (!gguf_bytes || !tensor_info || !out_q8_0_blocks ||
        !out_blocks_read || !out_bytes_read)
        return GGUF_READER_ERR_NULL_PTR;

    // Snapshot immutable tensor metadata so diagnostics publish only if the
    // caller did not mutate geometry while the checked read ran.
    staged_n_dims = tensor_info->n_dims;
    staged_ggml_type = tensor_info->ggml_type;
    staged_offset = tensor_info->offset;
    for (i = 0; i < 4; ++i)
        staged_dims[i] = tensor_info->dims[i];

    status = GGUFReadTensorDataQ8_0CheckedNoPartial(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q8_0_blocks,
        out_q8_0_blocks_capacity,
        &staged_blocks_read_i64,
        &staged_bytes_read_i64);
    if (status != GGUF_READER_OK)
        return status;

    if (!GGUFReaderTensorInfoQ8_0MatchesSnapshot(tensor_info,
                                                 staged_n_dims,
                                                 staged_ggml_type,
                                                 staged_offset,
                                                 staged_dims))
        return GGUF_READER_ERR_BAD_PARAM;

    // Commit-only diagnostics publish.
    *out_blocks_read = staged_blocks_read_i64;
    *out_bytes_read = staged_bytes_read_i64;
    return GGUF_READER_OK;
}

I32 GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnly(
    U8 *gguf_bytes,
    U64 gguf_nbytes,
    U64 tensor_data_base,
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U8 *out_q8_0_blocks,
    U64 out_q8_0_blocks_capacity,
    I64 *out_blocks_read,
    I64 *out_bytes_read)
{
    U32 i;
    U32 staged_n_dims;
    U32 staged_ggml_type;
    U64 staged_dims[4];
    U64 staged_offset;
    U64 staged_element_count;
    U64 staged_block_count;
    U64 staged_required_bytes;
    U64 staged_payload_start;
    U64 staged_payload_end;
    U64 staged_out_end;
    I64 staged_blocks_read_i64;
    I64 staged_bytes_read_i64;

    if (!gguf_bytes || !tensor_info || !out_q8_0_blocks ||
        !out_blocks_read || !out_bytes_read)
        return GGUF_READER_ERR_NULL_PTR;

    // Snapshot immutable tensor geometry first, then derive diagnostics from
    // this single view so publish is atomic and deterministic.
    staged_n_dims = tensor_info->n_dims;
    staged_ggml_type = tensor_info->ggml_type;
    staged_offset = tensor_info->offset;
    for (i = 0; i < 4; ++i)
        staged_dims[i] = tensor_info->dims[i];

    if (!staged_n_dims || staged_n_dims > 4)
        return GGUF_READER_ERR_BAD_DIM;
    if (staged_ggml_type != GGUF_READER_GGML_TYPE_Q8_0)
        return GGUF_READER_ERR_BAD_TYPE;

    staged_element_count = 1;
    for (i = 0; i < staged_n_dims; ++i)
    {
        if (!staged_dims[i])
            return GGUF_READER_ERR_BAD_DIM;
        if (!GGUFReaderTryMulU64(staged_element_count,
                                 staged_dims[i],
                                 &staged_element_count))
            return GGUF_READER_ERR_OVERFLOW;
    }

    // Q8_0 stores 32 quantized elements per block.
    if (staged_element_count % GGUF_READER_Q8_0_BLOCK_SIZE)
        return GGUF_READER_ERR_BAD_DIM;

    staged_block_count = staged_element_count / GGUF_READER_Q8_0_BLOCK_SIZE;
    if (!GGUFReaderTryMulU64(staged_block_count,
                             GGUF_READER_Q8_0_BLOCK_BYTES,
                             &staged_required_bytes))
        return GGUF_READER_ERR_OVERFLOW;

    if (!GGUFReaderTryAddU64(tensor_data_base,
                             staged_offset,
                             &staged_payload_start))
        return GGUF_READER_ERR_OVERFLOW;
    if (!GGUFReaderTryAddU64(staged_payload_start,
                             staged_required_bytes,
                             &staged_payload_end))
        return GGUF_READER_ERR_OVERFLOW;

    if (staged_payload_start > gguf_nbytes)
        return GGUF_READER_ERR_TRUNCATED;
    if (staged_payload_end > gguf_nbytes)
        return GGUF_READER_ERR_TRUNCATED;

    if (!GGUFReaderTryAddU64(0,
                             staged_required_bytes,
                             &staged_out_end))
        return GGUF_READER_ERR_OVERFLOW;
    if (staged_out_end > out_q8_0_blocks_capacity)
        return GGUF_READER_ERR_BAD_CAPACITY;

    if (!GGUFReaderTensorInfoQ8_0MatchesSnapshot(tensor_info,
                                                 staged_n_dims,
                                                 staged_ggml_type,
                                                 staged_offset,
                                                 staged_dims))
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_block_count > 0x7FFFFFFFFFFFFFFF)
        return GGUF_READER_ERR_OVERFLOW;
    if (staged_required_bytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_READER_ERR_OVERFLOW;

    staged_blocks_read_i64 = (I64)staged_block_count;
    staged_bytes_read_i64 = (I64)staged_required_bytes;

    // Preflight-only: never writes out_q8_0_blocks, only diagnostics.
    *out_blocks_read = staged_blocks_read_i64;
    *out_bytes_read = staged_bytes_read_i64;
    return GGUF_READER_OK;
}


I32 GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnlyParity(
    U8 *gguf_bytes,
    U64 gguf_nbytes,
    U64 tensor_data_base,
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U8 *out_q8_0_blocks,
    U64 out_q8_0_blocks_capacity,
    I64 *out_blocks_read,
    I64 *out_bytes_read)
{
    U32 i;
    U32 staged_n_dims;
    U32 staged_ggml_type;
    U64 staged_dims[4];
    U64 staged_offset;
    U64 staged_out_capacity;
    I64 staged_blocks_read_i64;
    I64 staged_bytes_read_i64;
    U64 recomputed_element_count;
    U64 recomputed_block_count;
    U64 recomputed_required_bytes;
    U64 recomputed_payload_start;
    U64 recomputed_payload_end;
    U64 recomputed_out_end;
    I64 recomputed_blocks_read_i64;
    I64 recomputed_bytes_read_i64;
    I32 status;

    if (!gguf_bytes || !tensor_info || !out_q8_0_blocks ||
        !out_blocks_read || !out_bytes_read)
        return GGUF_READER_ERR_NULL_PTR;

    if (out_blocks_read == out_bytes_read)
        return GGUF_READER_ERR_BAD_PARAM;

    if ((U8 *)out_blocks_read == out_q8_0_blocks ||
        (U8 *)out_bytes_read == out_q8_0_blocks)
        return GGUF_READER_ERR_BAD_PARAM;

    // Keep diagnostics outputs disjoint from tensor metadata storage so
    // snapshot revalidation cannot be corrupted by aliasing writes.
    if ((U8 *)out_blocks_read == (U8 *)tensor_info ||
        (U8 *)out_bytes_read == (U8 *)tensor_info)
        return GGUF_READER_ERR_BAD_PARAM;

    // Snapshot immutable tensor metadata for caller mutation detection.
    staged_n_dims = tensor_info->n_dims;
    staged_ggml_type = tensor_info->ggml_type;
    staged_offset = tensor_info->offset;
    staged_out_capacity = out_q8_0_blocks_capacity;
    for (i = 0; i < 4; ++i)
        staged_dims[i] = tensor_info->dims[i];

    // Stage canonical diagnostics from the no-write preflight helper.
    status = GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnly(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q8_0_blocks,
        out_q8_0_blocks_capacity,
        &staged_blocks_read_i64,
        &staged_bytes_read_i64);
    if (status != GGUF_READER_OK)
        return status;

    // Recompute canonical checked-read geometry from immutable snapshots.
    // This enforces exact tuple parity under strict zero-write semantics.
    if (!staged_n_dims || staged_n_dims > 4)
        return GGUF_READER_ERR_BAD_DIM;
    if (staged_ggml_type != GGUF_READER_GGML_TYPE_Q8_0)
        return GGUF_READER_ERR_BAD_TYPE;

    recomputed_element_count = 1;
    for (i = 0; i < staged_n_dims; ++i)
    {
        if (!staged_dims[i])
            return GGUF_READER_ERR_BAD_DIM;
        if (!GGUFReaderTryMulU64(recomputed_element_count,
                                 staged_dims[i],
                                 &recomputed_element_count))
            return GGUF_READER_ERR_OVERFLOW;
    }

    if (recomputed_element_count % GGUF_READER_Q8_0_BLOCK_SIZE)
        return GGUF_READER_ERR_BAD_DIM;

    recomputed_block_count =
        recomputed_element_count / GGUF_READER_Q8_0_BLOCK_SIZE;
    if (!GGUFReaderTryMulU64(recomputed_block_count,
                             GGUF_READER_Q8_0_BLOCK_BYTES,
                             &recomputed_required_bytes))
        return GGUF_READER_ERR_OVERFLOW;

    if (!GGUFReaderTryAddU64(tensor_data_base,
                             staged_offset,
                             &recomputed_payload_start))
        return GGUF_READER_ERR_OVERFLOW;
    if (!GGUFReaderTryAddU64(recomputed_payload_start,
                             recomputed_required_bytes,
                             &recomputed_payload_end))
        return GGUF_READER_ERR_OVERFLOW;

    if (recomputed_payload_start > gguf_nbytes)
        return GGUF_READER_ERR_TRUNCATED;
    if (recomputed_payload_end > gguf_nbytes)
        return GGUF_READER_ERR_TRUNCATED;

    if (!GGUFReaderTryAddU64(0,
                             recomputed_required_bytes,
                             &recomputed_out_end))
        return GGUF_READER_ERR_OVERFLOW;
    if (recomputed_out_end > staged_out_capacity)
        return GGUF_READER_ERR_BAD_CAPACITY;

    if (recomputed_block_count > 0x7FFFFFFFFFFFFFFF)
        return GGUF_READER_ERR_OVERFLOW;
    if (recomputed_required_bytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_READER_ERR_OVERFLOW;

    recomputed_blocks_read_i64 = (I64)recomputed_block_count;
    recomputed_bytes_read_i64 = (I64)recomputed_required_bytes;

    if (!GGUFReaderTensorInfoQ8_0MatchesSnapshot(tensor_info,
                                                 staged_n_dims,
                                                 staged_ggml_type,
                                                 staged_offset,
                                                 staged_dims))
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_out_capacity != out_q8_0_blocks_capacity)
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_blocks_read_i64 != recomputed_blocks_read_i64 ||
        staged_bytes_read_i64 != recomputed_bytes_read_i64)
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_blocks_read_i64 < 0 || staged_bytes_read_i64 < 0)
        return GGUF_READER_ERR_OVERFLOW;
    if ((U64)staged_bytes_read_i64 > staged_out_capacity)
        return GGUF_READER_ERR_BAD_CAPACITY;

    // Atomic diagnostics publish only after exact tuple parity succeeds.
    *out_blocks_read = staged_blocks_read_i64;
    *out_bytes_read = staged_bytes_read_i64;
    return GGUF_READER_OK;
}

I32 GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
    U8 *gguf_bytes,
    U64 gguf_nbytes,
    U64 tensor_data_base,
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U8 *out_q8_0_blocks,
    U64 out_q8_0_blocks_capacity,
    I64 *out_blocks_read,
    I64 *out_bytes_read)
{
    U32 i;
    U32 staged_n_dims;
    U32 staged_ggml_type;
    U64 staged_dims[4];
    U64 staged_offset;
    U64 staged_out_capacity;
    I64 staged_blocks_read_i64;
    I64 staged_bytes_read_i64;
    I32 status;

    if (!gguf_bytes || !tensor_info || !out_q8_0_blocks ||
        !out_blocks_read || !out_bytes_read)
        return GGUF_READER_ERR_NULL_PTR;

    if (out_blocks_read == out_bytes_read)
        return GGUF_READER_ERR_BAD_PARAM;

    if ((U8 *)out_blocks_read == out_q8_0_blocks ||
        (U8 *)out_bytes_read == out_q8_0_blocks)
        return GGUF_READER_ERR_BAD_PARAM;

    // Keep diagnostics outputs disjoint from tensor metadata storage so
    // post-stage snapshot revalidation cannot be corrupted by aliasing writes.
    if ((U8 *)out_blocks_read == (U8 *)tensor_info ||
        (U8 *)out_bytes_read == (U8 *)tensor_info)
        return GGUF_READER_ERR_BAD_PARAM;

    // Snapshot immutable tensor metadata and caller-reported output capacity
    // before staging diagnostics to enforce strict commit-only semantics.
    staged_n_dims = tensor_info->n_dims;
    staged_ggml_type = tensor_info->ggml_type;
    staged_offset = tensor_info->offset;
    staged_out_capacity = out_q8_0_blocks_capacity;
    for (i = 0; i < 4; ++i)
        staged_dims[i] = tensor_info->dims[i];

    status = GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnlyParity(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q8_0_blocks,
        out_q8_0_blocks_capacity,
        &staged_blocks_read_i64,
        &staged_bytes_read_i64);
    if (status != GGUF_READER_OK)
        return status;

    if (!GGUFReaderTensorInfoQ8_0MatchesSnapshot(tensor_info,
                                                 staged_n_dims,
                                                 staged_ggml_type,
                                                 staged_offset,
                                                 staged_dims))
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_out_capacity != out_q8_0_blocks_capacity)
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_blocks_read_i64 < 0 || staged_bytes_read_i64 < 0)
        return GGUF_READER_ERR_OVERFLOW;

    // Capacity revalidation on staged diagnostics before atomic publish.
    if ((U64)staged_bytes_read_i64 > staged_out_capacity)
        return GGUF_READER_ERR_BAD_CAPACITY;

    *out_blocks_read = staged_blocks_read_i64;
    *out_bytes_read = staged_bytes_read_i64;
    return GGUF_READER_OK;
}


I32 GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    U8 *gguf_bytes,
    U64 gguf_nbytes,
    U64 tensor_data_base,
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U8 *out_q8_0_blocks,
    U64 out_q8_0_blocks_capacity,
    I64 *out_blocks_read,
    I64 *out_bytes_read)
{
    U32 i;
    U32 staged_n_dims;
    U32 staged_ggml_type;
    U64 staged_dims[4];
    U64 staged_offset;
    U64 staged_out_capacity;
    I64 staged_blocks_read_i64;
    I64 staged_bytes_read_i64;
    I64 recomputed_blocks_read_i64;
    I64 recomputed_bytes_read_i64;
    I32 status;

    if (!gguf_bytes || !tensor_info || !out_q8_0_blocks ||
        !out_blocks_read || !out_bytes_read)
        return GGUF_READER_ERR_NULL_PTR;

    if (out_blocks_read == out_bytes_read)
        return GGUF_READER_ERR_BAD_PARAM;

    if ((U8 *)out_blocks_read == out_q8_0_blocks ||
        (U8 *)out_bytes_read == out_q8_0_blocks)
        return GGUF_READER_ERR_BAD_PARAM;

    // Snapshot immutable tensor metadata and caller-reported capacity so this
    // diagnostics-only wrapper can reject caller mutation before publish.
    staged_n_dims = tensor_info->n_dims;
    staged_ggml_type = tensor_info->ggml_type;
    staged_offset = tensor_info->offset;
    staged_out_capacity = out_q8_0_blocks_capacity;
    for (i = 0; i < 4; ++i)
        staged_dims[i] = tensor_info->dims[i];

    // Stage diagnostics from commit-only parity wrapper.
    status = GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q8_0_blocks,
        out_q8_0_blocks_capacity,
        &staged_blocks_read_i64,
        &staged_bytes_read_i64);
    if (status != GGUF_READER_OK)
        return status;

    // Recompute canonical diagnostics from the no-write preflight helper and
    // require exact tuple parity before atomic publish.
    status = GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnly(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q8_0_blocks,
        out_q8_0_blocks_capacity,
        &recomputed_blocks_read_i64,
        &recomputed_bytes_read_i64);
    if (status != GGUF_READER_OK)
        return status;

    if (!GGUFReaderTensorInfoQ8_0MatchesSnapshot(tensor_info,
                                                 staged_n_dims,
                                                 staged_ggml_type,
                                                 staged_offset,
                                                 staged_dims))
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_out_capacity != out_q8_0_blocks_capacity)
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_blocks_read_i64 != recomputed_blocks_read_i64 ||
        staged_bytes_read_i64 != recomputed_bytes_read_i64)
        return GGUF_READER_ERR_BAD_PARAM;

    if (staged_blocks_read_i64 < 0 || staged_bytes_read_i64 < 0)
        return GGUF_READER_ERR_OVERFLOW;

    if ((U64)staged_bytes_read_i64 > staged_out_capacity)
        return GGUF_READER_ERR_BAD_CAPACITY;

    // Preflight-only semantics: publish diagnostics only; payload stays
    // untouched because both composed helpers are no-write variants.
    *out_blocks_read = staged_blocks_read_i64;
    *out_bytes_read = staged_bytes_read_i64;
    return GGUF_READER_OK;
}


I32 GGUFReaderLoadTensorSliceQ8_0Checked(
    U8 *gguf_bytes,
    U64 gguf_nbytes,
    U64 tensor_data_base,
    GGUFReaderTensorInfoQ4_0 *tensor_info,
    U64 slice_block_start,
    U64 slice_block_count,
    U8 *out_q8_0_blocks,
    U64 out_q8_0_blocks_capacity,
    I64 *out_blocks_read,
    I64 *out_bytes_read,
    U64 *out_slice_payload_offset)
{
    U32 i;
    U32 staged_n_dims;
    U32 staged_ggml_type;
    U64 staged_dims[4];
    U64 staged_offset;
    U64 staged_total_elements;
    U64 staged_total_blocks;
    U64 staged_tensor_bytes;
    U64 staged_slice_end_block;
    U64 staged_slice_byte_offset;
    U64 staged_slice_bytes;
    U64 staged_tensor_payload_start;
    U64 staged_tensor_payload_end;
    U64 staged_slice_payload_start;
    U64 staged_slice_payload_end;
    U64 staged_out_end;
    I64 staged_blocks_read_i64;
    I64 staged_bytes_read_i64;
    U64 copy_idx;

    if (!gguf_bytes || !tensor_info || !out_q8_0_blocks ||
        !out_blocks_read || !out_bytes_read || !out_slice_payload_offset)
        return GGUF_READER_ERR_NULL_PTR;

    // Snapshot immutable tensor geometry and offsets so every bounds/range
    // decision and final diagnostics publish comes from one consistent view.
    staged_n_dims = tensor_info->n_dims;
    staged_ggml_type = tensor_info->ggml_type;
    staged_offset = tensor_info->offset;
    for (i = 0; i < 4; ++i)
        staged_dims[i] = tensor_info->dims[i];

    if (!staged_n_dims || staged_n_dims > 4)
        return GGUF_READER_ERR_BAD_DIM;
    if (staged_ggml_type != GGUF_READER_GGML_TYPE_Q8_0)
        return GGUF_READER_ERR_BAD_TYPE;

    staged_total_elements = 1;
    for (i = 0; i < staged_n_dims; ++i)
    {
        if (!staged_dims[i])
            return GGUF_READER_ERR_BAD_DIM;
        if (!GGUFReaderTryMulU64(staged_total_elements,
                                 staged_dims[i],
                                 &staged_total_elements))
            return GGUF_READER_ERR_OVERFLOW;
    }

    // Q8_0 block math is integer-only: 32 quantized elements per block,
    // 34-byte packed block payload (2-byte d + 32 signed quants).
    if (staged_total_elements % GGUF_READER_Q8_0_BLOCK_SIZE)
        return GGUF_READER_ERR_BAD_DIM;

    staged_total_blocks = staged_total_elements / GGUF_READER_Q8_0_BLOCK_SIZE;
    if (slice_block_start > staged_total_blocks)
        return GGUF_READER_ERR_BAD_PARAM;
    if (!GGUFReaderTryAddU64(slice_block_start,
                             slice_block_count,
                             &staged_slice_end_block))
        return GGUF_READER_ERR_OVERFLOW;
    if (staged_slice_end_block > staged_total_blocks)
        return GGUF_READER_ERR_BAD_PARAM;

    if (!GGUFReaderTryMulU64(staged_total_blocks,
                             GGUF_READER_Q8_0_BLOCK_BYTES,
                             &staged_tensor_bytes))
        return GGUF_READER_ERR_OVERFLOW;
    if (!GGUFReaderTryMulU64(slice_block_start,
                             GGUF_READER_Q8_0_BLOCK_BYTES,
                             &staged_slice_byte_offset))
        return GGUF_READER_ERR_OVERFLOW;
    if (!GGUFReaderTryMulU64(slice_block_count,
                             GGUF_READER_Q8_0_BLOCK_BYTES,
                             &staged_slice_bytes))
        return GGUF_READER_ERR_OVERFLOW;

    if (!GGUFReaderTryAddU64(tensor_data_base,
                             staged_offset,
                             &staged_tensor_payload_start))
        return GGUF_READER_ERR_OVERFLOW;
    if (!GGUFReaderTryAddU64(staged_tensor_payload_start,
                             staged_tensor_bytes,
                             &staged_tensor_payload_end))
        return GGUF_READER_ERR_OVERFLOW;

    if (staged_tensor_payload_start > gguf_nbytes)
        return GGUF_READER_ERR_TRUNCATED;
    if (staged_tensor_payload_end > gguf_nbytes)
        return GGUF_READER_ERR_TRUNCATED;

    if (!GGUFReaderTryAddU64(staged_tensor_payload_start,
                             staged_slice_byte_offset,
                             &staged_slice_payload_start))
        return GGUF_READER_ERR_OVERFLOW;
    if (!GGUFReaderTryAddU64(staged_slice_payload_start,
                             staged_slice_bytes,
                             &staged_slice_payload_end))
        return GGUF_READER_ERR_OVERFLOW;

    if (staged_slice_payload_start > staged_tensor_payload_end)
        return GGUF_READER_ERR_BAD_PARAM;
    if (staged_slice_payload_end > staged_tensor_payload_end)
        return GGUF_READER_ERR_BAD_PARAM;

    if (!GGUFReaderTryAddU64(0,
                             staged_slice_bytes,
                             &staged_out_end))
        return GGUF_READER_ERR_OVERFLOW;
    if (staged_out_end > out_q8_0_blocks_capacity)
        return GGUF_READER_ERR_BAD_CAPACITY;

    if (!GGUFReaderTensorInfoQ8_0MatchesSnapshot(tensor_info,
                                                 staged_n_dims,
                                                 staged_ggml_type,
                                                 staged_offset,
                                                 staged_dims))
        return GGUF_READER_ERR_BAD_PARAM;

    if (slice_block_count > 0x7FFFFFFFFFFFFFFF)
        return GGUF_READER_ERR_OVERFLOW;
    if (staged_slice_bytes > 0x7FFFFFFFFFFFFFFF)
        return GGUF_READER_ERR_OVERFLOW;

    staged_blocks_read_i64 = (I64)slice_block_count;
    staged_bytes_read_i64 = (I64)staged_slice_bytes;

    // No-partial publish: full preflight above, then one contiguous copy.
    for (copy_idx = 0; copy_idx < staged_slice_bytes; ++copy_idx)
        out_q8_0_blocks[copy_idx] = gguf_bytes[staged_slice_payload_start + copy_idx];

    *out_blocks_read = staged_blocks_read_i64;
    *out_bytes_read = staged_bytes_read_i64;
    *out_slice_payload_offset = staged_slice_payload_start;
    return GGUF_READER_OK;
}
