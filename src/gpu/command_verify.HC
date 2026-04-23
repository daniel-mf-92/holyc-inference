// command_verify.HC
// GPU command-stream preflight verifier.
//
// Security contract:
// - Reject unknown descriptor type/op before any MMIO submission path.
// - Enforce byte-range bounds against the dedicated GPU region.
// - Keep integer-only checked math; never allow silent wrap.

#define GPU_CMD_VERIFY_OK                 0
#define GPU_CMD_VERIFY_ERR_NULL_PTR       1
#define GPU_CMD_VERIFY_ERR_BAD_PARAM      2
#define GPU_CMD_VERIFY_ERR_BAD_TYPE       3
#define GPU_CMD_VERIFY_ERR_BAD_OP         4
#define GPU_CMD_VERIFY_ERR_BAD_RANGE      5
#define GPU_CMD_VERIFY_ERR_BAD_FLAGS      6
#define GPU_CMD_VERIFY_ERR_OVERFLOW       7
#define GPU_CMD_VERIFY_ERR_BAD_COMBO      8
#define GPU_CMD_VERIFY_ERR_BAD_ALIGNMENT  9
#define GPU_CMD_VERIFY_ERR_STREAM_BUDGET 10

#define GPU_CMD_VERIFY_REASON_ALLOW            0
#define GPU_CMD_VERIFY_REASON_NULL_DESC        1
#define GPU_CMD_VERIFY_REASON_BAD_GEOMETRY     2
#define GPU_CMD_VERIFY_REASON_BAD_TYPE         3
#define GPU_CMD_VERIFY_REASON_BAD_OP           4
#define GPU_CMD_VERIFY_REASON_BAD_BOUNDS       5
#define GPU_CMD_VERIFY_REASON_BAD_FLAGS        6
#define GPU_CMD_VERIFY_REASON_OVERFLOW         7
#define GPU_CMD_VERIFY_REASON_BAD_COMBO        8
#define GPU_CMD_VERIFY_REASON_BAD_ALIGNMENT    9
#define GPU_CMD_VERIFY_REASON_STREAM_BUDGET   10

#define GPU_CMD_DESC_TYPE_QMATMUL   1
#define GPU_CMD_DESC_TYPE_ATTN      2
#define GPU_CMD_DESC_TYPE_ELEMENT   3

#define GPU_CMD_OP_COPY_IN   1
#define GPU_CMD_OP_COMPUTE   2
#define GPU_CMD_OP_COPY_OUT  3

#define GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES 16

class GPUCommandDescriptor
{
    I64 desc_type;
    I64 desc_op;
    I64 src_offset_bytes;
    I64 dst_offset_bytes;
    I64 byte_count;
    I64 flags;
};

Bool GPUCommandDescTypeIsValid(I64 desc_type)
{
    return (desc_type == GPU_CMD_DESC_TYPE_QMATMUL ||
            desc_type == GPU_CMD_DESC_TYPE_ATTN ||
            desc_type == GPU_CMD_DESC_TYPE_ELEMENT);
}

Bool GPUCommandDescOpIsValid(I64 desc_op)
{
    return (desc_op == GPU_CMD_OP_COPY_IN ||
            desc_op == GPU_CMD_OP_COMPUTE ||
            desc_op == GPU_CMD_OP_COPY_OUT);
}

Bool GPUCommandDescTypeOpPairIsValid(I64 desc_type,
                                     I64 desc_op)
{
    if (desc_type == GPU_CMD_DESC_TYPE_ELEMENT)
        return (desc_op == GPU_CMD_OP_COMPUTE);

    return (desc_op == GPU_CMD_OP_COPY_IN ||
            desc_op == GPU_CMD_OP_COMPUTE ||
            desc_op == GPU_CMD_OP_COPY_OUT);
}

Bool GPUCommandIsAlignedPow2(I64 value,
                             I64 alignment)
{
    if (alignment <= 0)
        return FALSE;
    return (value & (alignment - 1)) == 0;
}

Bool GPUCommandTryAddNonNegativeI64(I64 a, I64 b, I64 *out)
{
    if (!out)
        return FALSE;
    if (a < 0 || b < 0)
        return FALSE;
    if (a > 0x7FFFFFFFFFFFFFFFLL - b)
        return FALSE;

    *out = a + b;
    return TRUE;
}

I32 GPUCommandVerifyDescriptorChecked(GPUCommandDescriptor *desc,
                                      I64 region_nbytes,
                                      I64 max_cmd_nbytes,
                                      I64 allowed_flags_mask,
                                      I64 *out_reason_code)
{
    I64 src_end;
    I64 dst_end;

    if (!out_reason_code)
        return GPU_CMD_VERIFY_ERR_NULL_PTR;

    *out_reason_code = GPU_CMD_VERIFY_REASON_BAD_GEOMETRY;

    if (!desc)
    {
        *out_reason_code = GPU_CMD_VERIFY_REASON_NULL_DESC;
        return GPU_CMD_VERIFY_ERR_NULL_PTR;
    }

    if (region_nbytes <= 0 || max_cmd_nbytes <= 0 ||
        max_cmd_nbytes > region_nbytes || allowed_flags_mask < 0)
        return GPU_CMD_VERIFY_ERR_BAD_PARAM;

    if (!GPUCommandDescTypeIsValid(desc->desc_type))
    {
        *out_reason_code = GPU_CMD_VERIFY_REASON_BAD_TYPE;
        return GPU_CMD_VERIFY_ERR_BAD_TYPE;
    }

    if (!GPUCommandDescOpIsValid(desc->desc_op))
    {
        *out_reason_code = GPU_CMD_VERIFY_REASON_BAD_OP;
        return GPU_CMD_VERIFY_ERR_BAD_OP;
    }

    if (!GPUCommandDescTypeOpPairIsValid(desc->desc_type,
                                         desc->desc_op))
    {
        *out_reason_code = GPU_CMD_VERIFY_REASON_BAD_COMBO;
        return GPU_CMD_VERIFY_ERR_BAD_COMBO;
    }

    if (desc->flags < 0 || (desc->flags & (~allowed_flags_mask)) != 0)
    {
        *out_reason_code = GPU_CMD_VERIFY_REASON_BAD_FLAGS;
        return GPU_CMD_VERIFY_ERR_BAD_FLAGS;
    }

    if (desc->src_offset_bytes < 0 || desc->dst_offset_bytes < 0 ||
        desc->byte_count <= 0 || desc->byte_count > max_cmd_nbytes)
    {
        *out_reason_code = GPU_CMD_VERIFY_REASON_BAD_BOUNDS;
        return GPU_CMD_VERIFY_ERR_BAD_RANGE;
    }

    // DMA/MMIO command streams are 16-byte aligned so descriptor math is
    // deterministic across host and TempleOS verifier implementations.
    if (!GPUCommandIsAlignedPow2(desc->src_offset_bytes,
                                 GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES) ||
        !GPUCommandIsAlignedPow2(desc->dst_offset_bytes,
                                 GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES) ||
        !GPUCommandIsAlignedPow2(desc->byte_count,
                                 GPU_CMD_VERIFY_DMA_ALIGNMENT_BYTES))
    {
        *out_reason_code = GPU_CMD_VERIFY_REASON_BAD_ALIGNMENT;
        return GPU_CMD_VERIFY_ERR_BAD_ALIGNMENT;
    }

    if (!GPUCommandTryAddNonNegativeI64(desc->src_offset_bytes,
                                         desc->byte_count,
                                         &src_end) ||
        !GPUCommandTryAddNonNegativeI64(desc->dst_offset_bytes,
                                         desc->byte_count,
                                         &dst_end))
    {
        *out_reason_code = GPU_CMD_VERIFY_REASON_OVERFLOW;
        return GPU_CMD_VERIFY_ERR_OVERFLOW;
    }

    if (src_end > region_nbytes || dst_end > region_nbytes)
    {
        *out_reason_code = GPU_CMD_VERIFY_REASON_BAD_BOUNDS;
        return GPU_CMD_VERIFY_ERR_BAD_RANGE;
    }

    *out_reason_code = GPU_CMD_VERIFY_REASON_ALLOW;
    return GPU_CMD_VERIFY_OK;
}

I32 GPUCommandVerifyStreamChecked(GPUCommandDescriptor *descs,
                                  I64 desc_count,
                                  I64 region_nbytes,
                                  I64 max_cmd_nbytes,
                                  I64 allowed_flags_mask,
                                  I64 *out_bad_index,
                                  I64 *out_reason_code)
{
    I64 i;
    I64 reason;
    I64 stream_total_bytes;
    I32 status;

    if (!descs || !out_bad_index || !out_reason_code)
        return GPU_CMD_VERIFY_ERR_NULL_PTR;

    *out_bad_index = -1;
    *out_reason_code = GPU_CMD_VERIFY_REASON_BAD_GEOMETRY;

    if (desc_count <= 0)
        return GPU_CMD_VERIFY_ERR_BAD_PARAM;

    stream_total_bytes = 0;

    for (i = 0; i < desc_count; ++i)
    {
        reason = GPU_CMD_VERIFY_REASON_BAD_GEOMETRY;
        status = GPUCommandVerifyDescriptorChecked(&descs[i],
                                                   region_nbytes,
                                                   max_cmd_nbytes,
                                                   allowed_flags_mask,
                                                   &reason);
        if (status != GPU_CMD_VERIFY_OK)
        {
            *out_bad_index = i;
            *out_reason_code = reason;
            return status;
        }

        if (!GPUCommandTryAddNonNegativeI64(stream_total_bytes,
                                            descs[i].byte_count,
                                            &stream_total_bytes))
        {
            *out_bad_index = i;
            *out_reason_code = GPU_CMD_VERIFY_REASON_OVERFLOW;
            return GPU_CMD_VERIFY_ERR_OVERFLOW;
        }

        // Command-list budget guard: the sum of all transfer spans in one
        // submission may never exceed the guarded GPU region.
        if (stream_total_bytes > region_nbytes)
        {
            *out_bad_index = i;
            *out_reason_code = GPU_CMD_VERIFY_REASON_STREAM_BUDGET;
            return GPU_CMD_VERIFY_ERR_STREAM_BUDGET;
        }
    }

    *out_reason_code = GPU_CMD_VERIFY_REASON_ALLOW;
    return GPU_CMD_VERIFY_OK;
}
