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

Bool GGUFTensorAbsRangeGreater(U64 a_start,
                               U64 a_end,
                               U64 a_index,
                               U64 b_start,
                               U64 b_end,
                               U64 b_index)
{
    if (a_start > b_start)
        return TRUE;
    if (a_start < b_start)
        return FALSE;

    if (a_end > b_end)
        return TRUE;
    if (a_end < b_end)
        return FALSE;

    // Final tie-break on original tensor index keeps ordering deterministic
    // for exact-equal ranges from degenerate zero-byte tensors.
    return a_index > b_index;
}

U0 GGUFTensorAbsRangeSiftDown(U64 *tensor_abs_starts,
                              U64 *tensor_abs_ends,
                              U64 *tensor_indices,
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

        if (GGUFTensorAbsRangeGreater(tensor_abs_starts[child],
                                      tensor_abs_ends[child],
                                      tensor_indices[child],
                                      tensor_abs_starts[swap_idx],
                                      tensor_abs_ends[swap_idx],
                                      tensor_indices[swap_idx]))
            swap_idx = child;

        if (child + 1 <= end &&
            GGUFTensorAbsRangeGreater(tensor_abs_starts[child + 1],
                                      tensor_abs_ends[child + 1],
                                      tensor_indices[child + 1],
                                      tensor_abs_starts[swap_idx],
                                      tensor_abs_ends[swap_idx],
                                      tensor_indices[swap_idx]))
            swap_idx = child + 1;

        if (swap_idx == root)
            return;

        GGUFSwapU64(&tensor_abs_starts[root], &tensor_abs_starts[swap_idx]);
        GGUFSwapU64(&tensor_abs_ends[root], &tensor_abs_ends[swap_idx]);
        GGUFSwapU64(&tensor_indices[root], &tensor_indices[swap_idx]);
        root = swap_idx;
    }
}

U0 GGUFTensorSortAbsRangesWithIndices(U64 tensor_count,
                                      U64 *tensor_abs_starts,
                                      U64 *tensor_abs_ends,
                                      U64 *tensor_indices)
{
    U64 start;
    U64 end;

    if (tensor_count < 2)
        return;

    // In-place heapsort over (abs_start, abs_end, original_index) tuples.
    start = (tensor_count - 2) / 2 + 1;
    while (start > 0)
    {
        --start;
        GGUFTensorAbsRangeSiftDown(tensor_abs_starts,
                                   tensor_abs_ends,
                                   tensor_indices,
                                   start,
                                   tensor_count - 1);
    }

    end = tensor_count - 1;
    while (end > 0)
    {
        GGUFSwapU64(&tensor_abs_starts[0], &tensor_abs_starts[end]);
        GGUFSwapU64(&tensor_abs_ends[0], &tensor_abs_ends[end]);
        GGUFSwapU64(&tensor_indices[0], &tensor_indices[end]);

        --end;
        GGUFTensorAbsRangeSiftDown(tensor_abs_starts,
                                   tensor_abs_ends,
                                   tensor_indices,
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

I64 GGUFTensorInfoResolveByteSpans(U64 tensor_data_base,
                                   U32 alignment,
                                   U64 gguf_file_nbytes,
                                   U64 tensor_count,
                                   U64 *tensor_rel_offsets,
                                   U64 *tensor_element_counts,
                                   U32 *tensor_ggml_types,
                                   U64 *out_tensor_nbytes,
                                   U64 *out_bad_index)
{
    U64 i;
    I64 err;

    if (!tensor_rel_offsets || !tensor_element_counts ||
        !tensor_ggml_types || !out_tensor_nbytes)
        return GGUF_TDBASE_ERR_NULL_PTR;

    if (out_bad_index)
        *out_bad_index = 0;

    // Pass 1: compute payload byte spans directly from ggml type + element count.
    // This keeps tensor sizing math centralized in GGUFTensorBytesForType.
    for (i = 0; i < tensor_count; ++i)
    {
        err = GGUFTensorBytesForType(tensor_ggml_types[i],
                                     tensor_element_counts[i],
                                     &out_tensor_nbytes[i]);
        if (err != GGUF_TDBASE_OK)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return err;
        }
    }

    // Pass 2: validate and sort ranges using the computed byte spans.
    // GGUFValidateTensorRangesSorted reorders offsets and nbytes in-place.
    return GGUFValidateTensorRangesSorted(tensor_data_base,
                                          alignment,
                                          gguf_file_nbytes,
                                          tensor_count,
                                          tensor_rel_offsets,
                                          out_tensor_nbytes,
                                          out_bad_index);
}

I64 GGUFTensorInfoResolveAbsRanges(U64 tensor_data_base,
                                   U32 alignment,
                                   U64 gguf_file_nbytes,
                                   U64 tensor_count,
                                   U64 *tensor_rel_offsets,
                                   U64 *tensor_element_counts,
                                   U32 *tensor_ggml_types,
                                   U64 *out_tensor_nbytes,
                                   U64 *out_abs_starts,
                                   U64 *out_abs_ends,
                                   U64 *out_bad_index)
{
    U64 i;
    U64 abs_start;
    I64 err;

    if (!out_abs_starts || !out_abs_ends)
        return GGUF_TDBASE_ERR_NULL_PTR;

    // First derive byte spans, enforce alignment/EOF safety, and sort
    // tensor_rel_offsets + out_tensor_nbytes in-place for monotonic scanning.
    err = GGUFTensorInfoResolveByteSpans(tensor_data_base,
                                         alignment,
                                         gguf_file_nbytes,
                                         tensor_count,
                                         tensor_rel_offsets,
                                         tensor_element_counts,
                                         tensor_ggml_types,
                                         out_tensor_nbytes,
                                         out_bad_index);
    if (err != GGUF_TDBASE_OK)
        return err;

    // With validated sorted ranges, resolve absolute [start,end) spans.
    for (i = 0; i < tensor_count; ++i)
    {
        err = GGUFTensorDataBaseOffset(tensor_data_base,
                                       tensor_rel_offsets[i],
                                       alignment,
                                       &abs_start);
        if (err != GGUF_TDBASE_OK)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return err;
        }

        if (abs_start > 0xFFFFFFFFFFFFFFFF - out_tensor_nbytes[i])
        {
            if (out_bad_index)
                *out_bad_index = i;
            return GGUF_TDBASE_ERR_OVERFLOW;
        }

        out_abs_starts[i] = abs_start;
        out_abs_ends[i] = abs_start + out_tensor_nbytes[i];
    }

    return GGUF_TDBASE_OK;
}

I64 GGUFTensorInfoBuildOffsetIndex(U64 tensor_data_base,
                                   U32 alignment,
                                   U64 gguf_file_nbytes,
                                   U64 tensor_count,
                                   U64 *tensor_rel_offsets,
                                   U64 *tensor_element_counts,
                                   U32 *tensor_ggml_types,
                                   U64 *out_abs_starts,
                                   U64 *out_abs_ends,
                                   U64 *out_sorted_tensor_indices,
                                   U64 *out_bad_index)
{
    U64 i;
    U64 tensor_nbytes;
    U64 max_end;
    I64 err;

    if (!tensor_rel_offsets || !tensor_element_counts || !tensor_ggml_types ||
        !out_abs_starts || !out_abs_ends || !out_sorted_tensor_indices)
        return GGUF_TDBASE_ERR_NULL_PTR;

    if (out_bad_index)
        *out_bad_index = 0;

    // Build per-tensor absolute ranges and preserve each tensor's original
    // index position before sorting.
    for (i = 0; i < tensor_count; ++i)
    {
        err = GGUFTensorBytesForType(tensor_ggml_types[i],
                                     tensor_element_counts[i],
                                     &tensor_nbytes);
        if (err != GGUF_TDBASE_OK)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return err;
        }

        err = GGUFTensorResolveRange(tensor_data_base,
                                     tensor_rel_offsets[i],
                                     tensor_nbytes,
                                     gguf_file_nbytes,
                                     alignment,
                                     &out_abs_starts[i],
                                     &out_abs_ends[i]);
        if (err != GGUF_TDBASE_OK)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return err;
        }

        out_sorted_tensor_indices[i] = i;
    }

    if (tensor_count < 2)
        return GGUF_TDBASE_OK;

    // Sort absolute ranges and carry their original tensor indices.
    GGUFTensorSortAbsRangesWithIndices(tensor_count,
                                       out_abs_starts,
                                       out_abs_ends,
                                       out_sorted_tensor_indices);

    // Validate non-overlap after sorting.
    max_end = out_abs_ends[0];
    for (i = 1; i < tensor_count; ++i)
    {
        if (out_abs_ends[i] < out_abs_starts[i])
        {
            if (out_bad_index)
                *out_bad_index = out_sorted_tensor_indices[i];
            return GGUF_TDBASE_ERR_OVERFLOW;
        }

        if (out_abs_starts[i] < max_end)
        {
            if (out_bad_index)
                *out_bad_index = out_sorted_tensor_indices[i];
            return GGUF_TDBASE_ERR_OVERLAP;
        }

        if (out_abs_ends[i] > max_end)
            max_end = out_abs_ends[i];
    }

    return GGUF_TDBASE_OK;
}

I64 GGUFTensorRangeFindByAbsOffset(U64 abs_offset,
                                   U64 tensor_count,
                                   U64 *tensor_abs_starts,
                                   U64 *tensor_abs_ends,
                                   U64 *out_tensor_index)
{
    U64 lo;
    U64 hi;
    U64 mid;
    U64 start_mid;
    U64 end_mid;

    if (!tensor_abs_starts || !tensor_abs_ends || !out_tensor_index)
        return GGUF_TDBASE_ERR_NULL_PTR;

    // Preserve deterministic output even on misses.
    *out_tensor_index = 0;

    if (!tensor_count)
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;

    // Binary search over sorted half-open ranges [start, end).
    // If abs_offset is below the current range, search left.
    // If abs_offset is at/after current range end, search right.
    lo = 0;
    hi = tensor_count;
    while (lo < hi)
    {
        mid = lo + ((hi - lo) >> 1);
        start_mid = tensor_abs_starts[mid];
        end_mid = tensor_abs_ends[mid];

        // Reject malformed range tables from corrupt callers.
        if (end_mid < start_mid)
            return GGUF_TDBASE_ERR_OVERFLOW;

        if (abs_offset < start_mid)
        {
            hi = mid;
            continue;
        }

        if (abs_offset >= end_mid)
        {
            lo = mid + 1;
            continue;
        }

        *out_tensor_index = mid;
        return GGUF_TDBASE_OK;
    }

    return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;
}

I64 GGUFTensorRangeFindByRelOffset(U64 tensor_rel_offset,
                                   U64 tensor_data_base,
                                   U32 alignment,
                                   U64 tensor_count,
                                   U64 *tensor_abs_starts,
                                   U64 *tensor_abs_ends,
                                   U64 *out_tensor_index)
{
    U64 abs_offset;
    I64 err;

    if (!out_tensor_index)
        return GGUF_TDBASE_ERR_NULL_PTR;

    // Preserve deterministic output even when offset translation fails.
    *out_tensor_index = 0;

    // Translate relative tensor offset to absolute file offset using the
    // same alignment + overflow checks as tensor range resolution.
    err = GGUFTensorDataBaseOffset(tensor_data_base,
                                   tensor_rel_offset,
                                   alignment,
                                   &abs_offset);
    if (err != GGUF_TDBASE_OK)
        return err;

    // Delegate lookup to the binary search helper over sorted absolute ranges.
    return GGUFTensorRangeFindByAbsOffset(abs_offset,
                                          tensor_count,
                                          tensor_abs_starts,
                                          tensor_abs_ends,
                                          out_tensor_index);
}

I64 GGUFTensorRangeFindByRelOffsetDefault(U64 tensor_rel_offset,
                                          U64 tensor_data_base,
                                          U64 tensor_count,
                                          U64 *tensor_abs_starts,
                                          U64 *tensor_abs_ends,
                                          U64 *out_tensor_index)
{
    // GGUF default tensor payload alignment is 32 bytes when
    // `general.alignment` metadata key is absent.
    return GGUFTensorRangeFindByRelOffset(tensor_rel_offset,
                                          tensor_data_base,
                                          GGUF_DEFAULT_ALIGNMENT,
                                          tensor_count,
                                          tensor_abs_starts,
                                          tensor_abs_ends,
                                          out_tensor_index);
}

I64 GGUFTensorRangeFindIndexByAbsOffset(U64 abs_offset,
                                        U64 tensor_count,
                                        U64 *tensor_abs_starts,
                                        U64 *tensor_abs_ends,
                                        U64 *sorted_tensor_indices,
                                        U64 *out_tensor_index)
{
    U64 sorted_pos;
    I64 err;

    if (!sorted_tensor_indices || !out_tensor_index)
        return GGUF_TDBASE_ERR_NULL_PTR;

    // Preserve deterministic output even on misses/errors.
    *out_tensor_index = 0;

    // Find sorted range slot first, then translate to the caller-visible
    // original tensor index captured by GGUFTensorInfoBuildOffsetIndex.
    err = GGUFTensorRangeFindByAbsOffset(abs_offset,
                                         tensor_count,
                                         tensor_abs_starts,
                                         tensor_abs_ends,
                                         &sorted_pos);
    if (err != GGUF_TDBASE_OK)
        return err;

    if (sorted_pos >= tensor_count)
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;

    if (sorted_tensor_indices[sorted_pos] >= tensor_count)
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;

    *out_tensor_index = sorted_tensor_indices[sorted_pos];
    return GGUF_TDBASE_OK;
}

I64 GGUFTensorRangeFindIndexByRelOffset(U64 tensor_rel_offset,
                                        U64 tensor_data_base,
                                        U32 alignment,
                                        U64 tensor_count,
                                        U64 *tensor_abs_starts,
                                        U64 *tensor_abs_ends,
                                        U64 *sorted_tensor_indices,
                                        U64 *out_tensor_index)
{
    U64 abs_offset;
    I64 err;

    if (!out_tensor_index)
        return GGUF_TDBASE_ERR_NULL_PTR;

    // Preserve deterministic output even when relative->absolute translation
    // fails before lookup.
    *out_tensor_index = 0;

    err = GGUFTensorDataBaseOffset(tensor_data_base,
                                   tensor_rel_offset,
                                   alignment,
                                   &abs_offset);
    if (err != GGUF_TDBASE_OK)
        return err;

    return GGUFTensorRangeFindIndexByAbsOffset(abs_offset,
                                               tensor_count,
                                               tensor_abs_starts,
                                               tensor_abs_ends,
                                               sorted_tensor_indices,
                                               out_tensor_index);
}

I64 GGUFTensorRangeFindIndexByRelOffsetDefault(U64 tensor_rel_offset,
                                               U64 tensor_data_base,
                                               U64 tensor_count,
                                               U64 *tensor_abs_starts,
                                               U64 *tensor_abs_ends,
                                               U64 *sorted_tensor_indices,
                                               U64 *out_tensor_index)
{
    // Default GGUF tensor payload alignment when `general.alignment`
    // metadata is absent.
    return GGUFTensorRangeFindIndexByRelOffset(tensor_rel_offset,
                                               tensor_data_base,
                                               GGUF_DEFAULT_ALIGNMENT,
                                               tensor_count,
                                               tensor_abs_starts,
                                               tensor_abs_ends,
                                               sorted_tensor_indices,
                                               out_tensor_index);
}

I64 GGUFValidateSortedTensorIndex(U64 tensor_count,
                                  U64 *tensor_abs_starts,
                                  U64 *tensor_abs_ends,
                                  U64 *sorted_tensor_indices,
                                  U64 *out_bad_index)
{
    U64 i;
    U64 j;
    U64 start_i;
    U64 end_i;
    U64 prev_start;
    U64 prev_end;
    U64 sorted_idx_i;

    if (!tensor_abs_starts || !tensor_abs_ends || !sorted_tensor_indices)
        return GGUF_TDBASE_ERR_NULL_PTR;

    if (out_bad_index)
        *out_bad_index = 0;

    if (!tensor_count)
        return GGUF_TDBASE_OK;

    for (i = 0; i < tensor_count; ++i)
    {
        start_i = tensor_abs_starts[i];
        end_i = tensor_abs_ends[i];
        sorted_idx_i = sorted_tensor_indices[i];

        // Ranges are half-open [start,end). End before start means
        // malformed table arithmetic or corrupted caller memory.
        if (end_i < start_i)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return GGUF_TDBASE_ERR_OVERFLOW;
        }

        // Binary search helpers require starts to be non-decreasing.
        // We also reject overlapping ranges up-front.
        if (i)
        {
            prev_start = tensor_abs_starts[i - 1];
            prev_end = tensor_abs_ends[i - 1];

            if (start_i < prev_start)
            {
                if (out_bad_index)
                    *out_bad_index = i;
                return GGUF_TDBASE_ERR_OVERLAP;
            }

            if (start_i == prev_start && end_i < prev_end)
            {
                if (out_bad_index)
                    *out_bad_index = i;
                return GGUF_TDBASE_ERR_OVERLAP;
            }

            if (start_i < prev_end)
            {
                if (out_bad_index)
                    *out_bad_index = i;
                return GGUF_TDBASE_ERR_OVERLAP;
            }
        }

        // The sorted->original map must be a permutation in [0, tensor_count).
        if (sorted_idx_i >= tensor_count)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;
        }

        for (j = 0; j < i; ++j)
        {
            if (sorted_tensor_indices[j] == sorted_idx_i)
            {
                if (out_bad_index)
                    *out_bad_index = i;
                return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;
            }
        }
    }

    return GGUF_TDBASE_OK;
}

I64 GGUFBuildTensorSortedPositionMap(U64 tensor_count,
                                     U64 *sorted_tensor_indices,
                                     U64 *out_sorted_position_by_tensor,
                                     U64 *out_bad_index)
{
    U64 i;
    U64 tensor_index;

    if (!sorted_tensor_indices || !out_sorted_position_by_tensor)
        return GGUF_TDBASE_ERR_NULL_PTR;

    if (out_bad_index)
        *out_bad_index = 0;

    // Initialize to sentinel `tensor_count` so duplicates and missing entries
    // can be detected in one forward pass without extra allocations.
    for (i = 0; i < tensor_count; ++i)
        out_sorted_position_by_tensor[i] = tensor_count;

    // Invert sorted->original map into original->sorted position map.
    for (i = 0; i < tensor_count; ++i)
    {
        tensor_index = sorted_tensor_indices[i];

        if (tensor_index >= tensor_count)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;
        }

        if (out_sorted_position_by_tensor[tensor_index] != tensor_count)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;
        }

        out_sorted_position_by_tensor[tensor_index] = i;
    }

    // Confirm every original tensor index appeared exactly once.
    for (i = 0; i < tensor_count; ++i)
    {
        if (out_sorted_position_by_tensor[i] == tensor_count)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;
        }
    }

    return GGUF_TDBASE_OK;
}

I64 GGUFValidateSortedPositionMap(U64 tensor_count,
                                  U64 *sorted_tensor_indices,
                                  U64 *sorted_position_by_tensor,
                                  U64 *out_bad_index)
{
    U64 i;
    U64 j;
    U64 sorted_pos;
    U64 orig_idx;

    if (!sorted_tensor_indices || !sorted_position_by_tensor)
        return GGUF_TDBASE_ERR_NULL_PTR;

    if (out_bad_index)
        *out_bad_index = 0;

    // Pass 1: validate original->sorted map bounds, enforce uniqueness,
    // and check round-trip consistency through sorted_tensor_indices.
    for (i = 0; i < tensor_count; ++i)
    {
        sorted_pos = sorted_position_by_tensor[i];

        if (sorted_pos >= tensor_count)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;
        }

        for (j = 0; j < i; ++j)
        {
            if (sorted_position_by_tensor[j] == sorted_pos)
            {
                if (out_bad_index)
                    *out_bad_index = i;
                return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;
            }
        }

        if (sorted_tensor_indices[sorted_pos] != i)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;
        }
    }

    // Pass 2: validate sorted->original entries are in-range and that
    // each sorted position points back to itself via inverse map.
    for (i = 0; i < tensor_count; ++i)
    {
        orig_idx = sorted_tensor_indices[i];

        if (orig_idx >= tensor_count)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;
        }

        if (sorted_position_by_tensor[orig_idx] != i)
        {
            if (out_bad_index)
                *out_bad_index = i;
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;
        }
    }

    return GGUF_TDBASE_OK;
}

I64 GGUFTensorIndexToAbsRange(U64 tensor_index,
                              U64 tensor_count,
                              U64 *tensor_abs_starts,
                              U64 *tensor_abs_ends,
                              U64 *sorted_position_by_tensor,
                              U64 *out_abs_start,
                              U64 *out_abs_end)
{
    U64 sorted_pos;
    U64 abs_start;
    U64 abs_end;

    if (!tensor_abs_starts || !tensor_abs_ends || !sorted_position_by_tensor ||
        !out_abs_start || !out_abs_end)
        return GGUF_TDBASE_ERR_NULL_PTR;

    // Keep deterministic outputs for callers even on invalid inputs.
    *out_abs_start = 0;
    *out_abs_end = 0;

    // Caller uses original tensor index space [0, tensor_count).
    if (tensor_index >= tensor_count)
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;

    // Inverse map gives O(1) jump from original index to sorted span slot.
    sorted_pos = sorted_position_by_tensor[tensor_index];
    if (sorted_pos >= tensor_count)
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;

    abs_start = tensor_abs_starts[sorted_pos];
    abs_end = tensor_abs_ends[sorted_pos];

    // Spans are half-open [start,end); end before start is malformed.
    if (abs_end < abs_start)
        return GGUF_TDBASE_ERR_OVERFLOW;

    *out_abs_start = abs_start;
    *out_abs_end = abs_end;
    return GGUF_TDBASE_OK;
}

I64 GGUFTensorRangeFindByIndex(U64 tensor_index,
                               U64 tensor_count,
                               U64 *tensor_abs_starts,
                               U64 *tensor_abs_ends,
                               U64 *sorted_position_by_tensor,
                               U64 *out_abs_start,
                               U64 *out_abs_end)
{
    // Public helper alias for index->range resolution. Keep one canonical
    // implementation path so validation/overflow checks stay identical.
    return GGUFTensorIndexToAbsRange(tensor_index,
                                     tensor_count,
                                     tensor_abs_starts,
                                     tensor_abs_ends,
                                     sorted_position_by_tensor,
                                     out_abs_start,
                                     out_abs_end);
}

I64 GGUFTensorLookupByRelOffset(U64 tensor_rel_offset,
                                U64 tensor_data_base,
                                U32 alignment,
                                U64 tensor_count,
                                U64 *tensor_abs_starts,
                                U64 *tensor_abs_ends,
                                U64 *sorted_tensor_indices,
                                U64 *out_tensor_index,
                                U64 *out_abs_start,
                                U64 *out_abs_end)
{
    U64 sorted_pos;
    U64 tensor_index;
    U64 abs_start;
    U64 abs_end;
    I64 err;

    if (!sorted_tensor_indices || !out_tensor_index || !out_abs_start ||
        !out_abs_end)
        return GGUF_TDBASE_ERR_NULL_PTR;

    // Preserve deterministic output contract on all failures.
    *out_tensor_index = 0;
    *out_abs_start = 0;
    *out_abs_end = 0;

    // Step 1: locate sorted range slot for caller-provided relative offset.
    err = GGUFTensorRangeFindByRelOffset(tensor_rel_offset,
                                         tensor_data_base,
                                         alignment,
                                         tensor_count,
                                         tensor_abs_starts,
                                         tensor_abs_ends,
                                         &sorted_pos);
    if (err != GGUF_TDBASE_OK)
        return err;

    if (sorted_pos >= tensor_count)
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;

    // Step 2: map sorted slot back to original tensor index via
    // tensor table permutation produced by GGUFTensorInfoBuildOffsetIndex.
    tensor_index = sorted_tensor_indices[sorted_pos];
    if (tensor_index >= tensor_count)
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS;

    abs_start = tensor_abs_starts[sorted_pos];
    abs_end = tensor_abs_ends[sorted_pos];
    if (abs_end < abs_start)
        return GGUF_TDBASE_ERR_OVERFLOW;

    *out_tensor_index = tensor_index;
    *out_abs_start = abs_start;
    *out_abs_end = abs_end;
    return GGUF_TDBASE_OK;
}

I64 GGUFTensorLookupByRelOffsetDefault(U64 tensor_rel_offset,
                                       U64 tensor_data_base,
                                       U64 tensor_count,
                                       U64 *tensor_abs_starts,
                                       U64 *tensor_abs_ends,
                                       U64 *sorted_tensor_indices,
                                       U64 *out_tensor_index,
                                       U64 *out_abs_start,
                                       U64 *out_abs_end)
{
    // Default GGUF tensor payload alignment when `general.alignment`
    // metadata key is absent.
    return GGUFTensorLookupByRelOffset(tensor_rel_offset,
                                       tensor_data_base,
                                       GGUF_DEFAULT_ALIGNMENT,
                                       tensor_count,
                                       tensor_abs_starts,
                                       tensor_abs_ends,
                                       sorted_tensor_indices,
                                       out_tensor_index,
                                       out_abs_start,
                                       out_abs_end);
}
