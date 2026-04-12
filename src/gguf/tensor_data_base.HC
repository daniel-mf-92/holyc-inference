// tensor_data_base.HC
// GGUF tensor-data base alignment helpers for HolyC inference runtime.

#define GGUF_TDBASE_OK 0
#define GGUF_TDBASE_ERR_NULL_PTR 1
#define GGUF_TDBASE_ERR_BAD_ALIGNMENT 2
#define GGUF_TDBASE_ERR_OVERFLOW 3
#define GGUF_TDBASE_ERR_MISALIGNED_BASE 4
#define GGUF_TDBASE_ERR_MISALIGNED_TENSOR_OFFSET 5

// GGUF default tensor alignment when metadata key `general.alignment`
// is absent.
#define GGUF_DEFAULT_ALIGNMENT 32

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
