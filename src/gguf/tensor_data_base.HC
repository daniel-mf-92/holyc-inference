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
