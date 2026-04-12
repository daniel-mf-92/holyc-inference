// tensor_data_base.HC
// GGUF tensor-data base alignment helpers for HolyC inference runtime.

#define GGUF_TDBASE_OK 0
#define GGUF_TDBASE_ERR_NULL_PTR 1
#define GGUF_TDBASE_ERR_BAD_ALIGNMENT 2
#define GGUF_TDBASE_ERR_OVERFLOW 3
#define GGUF_TDBASE_ERR_MISALIGNED_BASE 4
#define GGUF_TDBASE_ERR_MISALIGNED_TENSOR_OFFSET 5
#define GGUF_TDBASE_ERR_BAD_TYPE 6
#define GGUF_TDBASE_ERR_OUT_OF_BOUNDS 7
#define GGUF_TDBASE_ERR_BAD_BLOCK_MULTIPLE 8
#define GGUF_TDBASE_ERR_OVERLAP 9

// GGUF default tensor alignment when metadata key `general.alignment`
// is absent.
#define GGUF_DEFAULT_ALIGNMENT 32

// Subset of ggml tensor types currently enabled in HolyC inference runtime.
// Keep this list strict and explicit so sizing math never silently accepts
// unknown quantization tags from malformed GGUF files.
#define GGML_TYPE_F32   0
#define GGML_TYPE_F16   1
#define GGML_TYPE_Q4_0  2
#define GGML_TYPE_Q8_0  8

// GGML quantization blocks for runtime-critical formats:
// - F32/F16 are scalar types (block size 1).
// - Q4_0 and Q8_0 are packed in 32-element blocks.
#define GGML_BLCK_SIZE_F32   1
#define GGML_BLCK_SIZE_F16   1
#define GGML_BLCK_SIZE_Q4_0  32
#define GGML_BLCK_SIZE_Q8_0  32

// Block payload byte widths from ggml struct layouts:
// - Q4_0: fp16 scale (2 bytes) + 16 packed nibbles = 18 bytes.
// - Q8_0: fp16 scale (2 bytes) + 32 signed bytes = 34 bytes.
#define GGML_TYPE_SIZE_F32   4
#define GGML_TYPE_SIZE_F16   2
#define GGML_TYPE_SIZE_Q4_0  18
#define GGML_TYPE_SIZE_Q8_0  34

I64 GGUFTensorTypeBlockSize(U32 ggml_type, U64 *out_block_size)
{
    if (!out_block_size)
        return GGUF_TDBASE_ERR_NULL_PTR;

    switch (ggml_type)
    {
        case GGML_TYPE_F32:
            *out_block_size = GGML_BLCK_SIZE_F32;
            return GGUF_TDBASE_OK;
        case GGML_TYPE_F16:
            *out_block_size = GGML_BLCK_SIZE_F16;
            return GGUF_TDBASE_OK;
        case GGML_TYPE_Q4_0:
            *out_block_size = GGML_BLCK_SIZE_Q4_0;
            return GGUF_TDBASE_OK;
        case GGML_TYPE_Q8_0:
            *out_block_size = GGML_BLCK_SIZE_Q8_0;
            return GGUF_TDBASE_OK;
    }

    return GGUF_TDBASE_ERR_BAD_TYPE;
}

I64 GGUFTensorTypeBlockBytes(U32 ggml_type, U64 *out_block_bytes)
{
    if (!out_block_bytes)
        return GGUF_TDBASE_ERR_NULL_PTR;

    switch (ggml_type)
    {
        case GGML_TYPE_F32:
            *out_block_bytes = GGML_TYPE_SIZE_F32;
            return GGUF_TDBASE_OK;
        case GGML_TYPE_F16:
            *out_block_bytes = GGML_TYPE_SIZE_F16;
            return GGUF_TDBASE_OK;
        case GGML_TYPE_Q4_0:
            *out_block_bytes = GGML_TYPE_SIZE_Q4_0;
            return GGUF_TDBASE_OK;
        case GGML_TYPE_Q8_0:
            *out_block_bytes = GGML_TYPE_SIZE_Q8_0;
            return GGUF_TDBASE_OK;
    }

    return GGUF_TDBASE_ERR_BAD_TYPE;
}

I64 GGUFTensorBytesForType(U32 ggml_type,
                           U64 element_count,
                           U64 *out_tensor_nbytes)
{
    U64 block_size;
    U64 block_bytes;
    U64 block_count;
    I64 err;

    if (!out_tensor_nbytes)
        return GGUF_TDBASE_ERR_NULL_PTR;

    err = GGUFTensorTypeBlockSize(ggml_type, &block_size);
    if (err != GGUF_TDBASE_OK)
        return err;

    err = GGUFTensorTypeBlockBytes(ggml_type, &block_bytes);
    if (err != GGUF_TDBASE_OK)
        return err;

    // Quantized formats are stored in fixed-size element blocks.
    // Reject non-multiples instead of silently rounding/truncating.
    if (element_count % block_size)
        return GGUF_TDBASE_ERR_BAD_BLOCK_MULTIPLE;

    block_count = element_count / block_size;

    // Byte size = number of blocks * bytes per block.
    // Guard overflow before multiplication for malformed giant tensors.
    if (block_count > 0xFFFFFFFFFFFFFFFF / block_bytes)
        return GGUF_TDBASE_ERR_OVERFLOW;

    *out_tensor_nbytes = block_count * block_bytes;
    return GGUF_TDBASE_OK;
}

Bool GGUFIsPow2U32(U32 value)
{
    if (!value)
        return FALSE;
    return (value & (value - 1)) == 0;
}

I64 GGUFValidateAlignment(U32 alignment)
{
    if (!GGUFIsPow2U32(alignment))
        return GGUF_TDBASE_ERR_BAD_ALIGNMENT;
    return GGUF_TDBASE_OK;
}

I64 GGUFAlignUpU64(U64 value, U32 alignment, U64 *out)
{
    U64 mask;

    if (!out)
        return GGUF_TDBASE_ERR_NULL_PTR;

    if (GGUFValidateAlignment(alignment) != GGUF_TDBASE_OK)
        return GGUF_TDBASE_ERR_BAD_ALIGNMENT;

    // For power-of-two alignments, align-up is:
    //   (value + (alignment - 1)) & ~(alignment - 1)
    // Keep overflow checks explicit so giant malformed offsets are rejected.
    mask = (U64)alignment - 1;
    if (value > 0xFFFFFFFFFFFFFFFF - mask)
        return GGUF_TDBASE_ERR_OVERFLOW;

    *out = (value + mask) & ~mask;
    return GGUF_TDBASE_OK;
}

I64 GGUFTensorDataBaseAlign(U64 cursor_after_tensor_info,
                            U32 alignment,
                            U64 *out_tensor_data_base)
{
    return GGUFAlignUpU64(cursor_after_tensor_info,
                          alignment,
                          out_tensor_data_base);
}

I64 GGUFTensorDataBaseAlignDefault(U64 cursor_after_tensor_info,
                                   U64 *out_tensor_data_base)
{
    return GGUFTensorDataBaseAlign(cursor_after_tensor_info,
                                   GGUF_DEFAULT_ALIGNMENT,
                                   out_tensor_data_base);
}

Bool GGUFTensorDataBaseIsAligned(U64 offset, U32 alignment)
{
    U64 mask;

    if (GGUFValidateAlignment(alignment) != GGUF_TDBASE_OK)
        return FALSE;

    mask = (U64)alignment - 1;
    return (offset & mask) == 0;
}

I64 GGUFTensorDataBaseOffset(U64 tensor_data_base,
                             U64 tensor_rel_offset,
                             U32 alignment,
                             U64 *out_tensor_abs)
{
    if (!out_tensor_abs)
        return GGUF_TDBASE_ERR_NULL_PTR;

    if (GGUFValidateAlignment(alignment) != GGUF_TDBASE_OK)
        return GGUF_TDBASE_ERR_BAD_ALIGNMENT;

    // Both base and relative tensor offsets must preserve GGUF alignment.
    // Enforcing this here keeps every downstream payload read auditable.
    if (!GGUFTensorDataBaseIsAligned(tensor_data_base, alignment))
        return GGUF_TDBASE_ERR_MISALIGNED_BASE;
    if (!GGUFTensorDataBaseIsAligned(tensor_rel_offset, alignment))
        return GGUF_TDBASE_ERR_MISALIGNED_TENSOR_OFFSET;

    // Absolute payload pointer is base + relative offset.
    // Keep overflow check explicit for malformed large files.
    if (tensor_data_base > 0xFFFFFFFFFFFFFFFF - tensor_rel_offset)
        return GGUF_TDBASE_ERR_OVERFLOW;

    *out_tensor_abs = tensor_data_base + tensor_rel_offset;
    return GGUF_TDBASE_OK;
}

I64 GGUFTensorDataBaseOffsetDefault(U64 tensor_data_base,
                                    U64 tensor_rel_offset,
                                    U64 *out_tensor_abs)
{
    return GGUFTensorDataBaseOffset(tensor_data_base,
                                    tensor_rel_offset,
                                    GGUF_DEFAULT_ALIGNMENT,
                                    out_tensor_abs);
}

I64 GGUFTensorResolveRange(U64 tensor_data_base,
                           U64 tensor_rel_offset,
                           U64 tensor_nbytes,
                           U64 gguf_file_nbytes,
                           U32 alignment,
                           U64 *out_tensor_abs_start,
                           U64 *out_tensor_abs_end)
{
    U64 abs_start;
    U64 abs_end;
    I64 err;

    if (!out_tensor_abs_start || !out_tensor_abs_end)
        return GGUF_TDBASE_ERR_NULL_PTR;

    // Resolve and validate the absolute tensor start offset first.
    err = GGUFTensorDataBaseOffset(tensor_data_base,
                                   tensor_rel_offset,
                                   alignment,
                                   &abs_start);
    if (err != GGUF_TDBASE_OK)
        return err;

    // Tensor byte-range is [abs_start, abs_start + tensor_nbytes).
    // Guard add overflow before constructing the exclusive end.
    if (abs_start > 0xFFFFFFFFFFFFFFFF - tensor_nbytes)
        return GGUF_TDBASE_ERR_OVERFLOW;
    abs_end = abs_start + tensor_nbytes;

    // Reject ranges that run past the file boundary.
    // A tensor may legally end exactly at gguf_file_nbytes.
    if (abs_end > gguf_file_nbytes)
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;

    *out_tensor_abs_start = abs_start;
    *out_tensor_abs_end   = abs_end;
    return GGUF_TDBASE_OK;
}

I64 GGUFTensorResolveRangeDefault(U64 tensor_data_base,
                                  U64 tensor_rel_offset,
                                  U64 tensor_nbytes,
                                  U64 gguf_file_nbytes,
                                  U64 *out_tensor_abs_start,
                                  U64 *out_tensor_abs_end)
{
    return GGUFTensorResolveRange(tensor_data_base,
                                  tensor_rel_offset,
                                  tensor_nbytes,
                                  gguf_file_nbytes,
                                  GGUF_DEFAULT_ALIGNMENT,
                                  out_tensor_abs_start,
                                  out_tensor_abs_end);
}

I64 GGUFValidateTensorRanges(U64 tensor_data_base,
                             U32 alignment,
                             U64 gguf_file_nbytes,
                             U64 tensor_count,
                             U64 *tensor_rel_offsets,
                             U64 *tensor_nbytes,
                             U64 *out_bad_index)
{
    U64 i;
    U64 j;
    U64 start_i;
    U64 end_i;
    U64 start_j;
    U64 end_j;
    I64 err;

    if (!tensor_rel_offsets || !tensor_nbytes)
        return GGUF_TDBASE_ERR_NULL_PTR;

    if (out_bad_index)
        *out_bad_index = 0;

    // Pass 1: each tensor range must be individually valid.
    // This enforces aligned offsets, add-overflow safety, and EOF bounds.
    for (i = 0; i < tensor_count; ++i)
    {
        err = GGUFTensorResolveRange(tensor_data_base,
                                     tensor_rel_offsets[i],
                                     tensor_nbytes[i],
                                     gguf_file_nbytes,
                                     alignment,
                                     &start_i,
                                     &end_i);
        if (err != GGUF_TDBASE_OK)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return err;
        }
    }

    // Pass 2: pairwise overlap check.
    // Explicit O(n^2) keeps logic simple and fully auditable in HolyC.
    for (i = 0; i < tensor_count; ++i)
    {
        err = GGUFTensorResolveRange(tensor_data_base,
                                     tensor_rel_offsets[i],
                                     tensor_nbytes[i],
                                     gguf_file_nbytes,
                                     alignment,
                                     &start_i,
                                     &end_i);
        if (err != GGUF_TDBASE_OK)
            return err;

        for (j = i + 1; j < tensor_count; ++j)
        {
            err = GGUFTensorResolveRange(tensor_data_base,
                                         tensor_rel_offsets[j],
                                         tensor_nbytes[j],
                                         gguf_file_nbytes,
                                         alignment,
                                         &start_j,
                                         &end_j);
            if (err != GGUF_TDBASE_OK)
                return err;

            // Half-open intervals [start, end) overlap when each interval starts
            // before the other ends. Exact touching boundaries are legal.
            if ((start_i < end_j) && (start_j < end_i))
            {
                if (out_bad_index)
                    *out_bad_index = j;
                return GGUF_TDBASE_ERR_OVERLAP;
            }
        }
    }

    return GGUF_TDBASE_OK;
}

U0 GGUFSwapU64(U64 *a, U64 *b)
{
    U64 tmp;

    tmp = *a;
    *a = *b;
    *b = tmp;
}

Bool GGUFTensorRangeGreater(U64 a_off,
                            U64 a_nbytes,
                            U64 b_off,
                            U64 b_nbytes)
{
    if (a_off > b_off)
        return TRUE;
    if (a_off < b_off)
        return FALSE;

    // Tie-break by nbytes so equal offsets are ordered shortest-first.
    // This preserves GGUFValidateTensorRanges behavior for zero-byte ranges.
    return a_nbytes > b_nbytes;
}

U0 GGUFTensorRangeSiftDown(U64 *tensor_rel_offsets,
                           U64 *tensor_nbytes,
                           U64 root,
                           U64 end)
{
    U64 child;
    U64 swap_idx;

    while (TRUE)
    {
        child = root * 2 + 1;
        if (child > end)
            return;

        swap_idx = root;

        if (GGUFTensorRangeGreater(tensor_rel_offsets[child],
                                   tensor_nbytes[child],
                                   tensor_rel_offsets[swap_idx],
                                   tensor_nbytes[swap_idx]))
            swap_idx = child;

        if (child + 1 <= end &&
            GGUFTensorRangeGreater(tensor_rel_offsets[child + 1],
                                   tensor_nbytes[child + 1],
                                   tensor_rel_offsets[swap_idx],
                                   tensor_nbytes[swap_idx]))
            swap_idx = child + 1;

        if (swap_idx == root)
            return;

        GGUFSwapU64(&tensor_rel_offsets[root], &tensor_rel_offsets[swap_idx]);
        GGUFSwapU64(&tensor_nbytes[root], &tensor_nbytes[swap_idx]);
        root = swap_idx;
    }
}

U0 GGUFTensorSortRangesByOffset(U64 tensor_count,
                                U64 *tensor_rel_offsets,
                                U64 *tensor_nbytes)
{
    U64 start;
    U64 end;

    if (tensor_count < 2)
        return;

    // In-place heapsort by relative offset so overlap detection can scan once.
    // We swap offset and nbytes together to preserve each tensor range pair.
    start = (tensor_count - 2) / 2 + 1;
    while (start > 0)
    {
        --start;
        GGUFTensorRangeSiftDown(tensor_rel_offsets,
                                tensor_nbytes,
                                start,
                                tensor_count - 1);
    }

    end = tensor_count - 1;
    while (end > 0)
    {
        GGUFSwapU64(&tensor_rel_offsets[0], &tensor_rel_offsets[end]);
        GGUFSwapU64(&tensor_nbytes[0], &tensor_nbytes[end]);

        --end;
        GGUFTensorRangeSiftDown(tensor_rel_offsets,
                                tensor_nbytes,
                                0,
                                end);
    }
}

I64 GGUFValidateTensorRangesSorted(U64 tensor_data_base,
                                   U32 alignment,
                                   U64 gguf_file_nbytes,
                                   U64 tensor_count,
                                   U64 *tensor_rel_offsets,
                                   U64 *tensor_nbytes,
                                   U64 *out_bad_index)
{
    U64 i;
    U64 start_i;
    U64 end_i;
    U64 max_end;
    I64 err;

    if (!tensor_rel_offsets || !tensor_nbytes)
        return GGUF_TDBASE_ERR_NULL_PTR;

    if (out_bad_index)
        *out_bad_index = 0;

    // Pass 1: preserve strict per-range validation behavior from
    // GGUFValidateTensorRanges before any in-place reordering.
    for (i = 0; i < tensor_count; ++i)
    {
        err = GGUFTensorResolveRange(tensor_data_base,
                                     tensor_rel_offsets[i],
                                     tensor_nbytes[i],
                                     gguf_file_nbytes,
                                     alignment,
                                     &start_i,
                                     &end_i);
        if (err != GGUF_TDBASE_OK)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return err;
        }
    }

    if (tensor_count < 2)
        return GGUF_TDBASE_OK;

    // Sort in-place by tensor relative offsets.
    GGUFTensorSortRangesByOffset(tensor_count,
                                 tensor_rel_offsets,
                                 tensor_nbytes);

    // Single linear overlap scan over half-open ranges [start, end).
    err = GGUFTensorResolveRange(tensor_data_base,
                                 tensor_rel_offsets[0],
                                 tensor_nbytes[0],
                                 gguf_file_nbytes,
                                 alignment,
                                 &start_i,
                                 &max_end);
    if (err != GGUF_TDBASE_OK)
        return err;

    for (i = 1; i < tensor_count; ++i)
    {
        err = GGUFTensorResolveRange(tensor_data_base,
                                     tensor_rel_offsets[i],
                                     tensor_nbytes[i],
                                     gguf_file_nbytes,
                                     alignment,
                                     &start_i,
                                     &end_i);
        if (err != GGUF_TDBASE_OK)
            return err;

        if (start_i < max_end)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return GGUF_TDBASE_ERR_OVERLAP;
        }

        if (end_i > max_end)
            max_end = end_i;
    }

    return GGUF_TDBASE_OK;
}
