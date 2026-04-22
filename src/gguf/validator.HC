// validator.HC
// Checked GGUF tensor-layout validation for model payload bounds.

#define GGUF_MODEL_VALIDATE_OK                 0
#define GGUF_MODEL_VALIDATE_ERR_NULL_PTR       1
#define GGUF_MODEL_VALIDATE_ERR_BAD_PARAM      2
#define GGUF_MODEL_VALIDATE_ERR_OVERFLOW       3
#define GGUF_MODEL_VALIDATE_ERR_BAD_TYPE       4
#define GGUF_MODEL_VALIDATE_ERR_BAD_DIMS       5
#define GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT  6
#define GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS  7
#define GGUF_MODEL_VALIDATE_ERR_OVERLAP        8

#define GGUF_MODEL_VALIDATE_MAX_TENSORS (1 << 20)
#define GGUF_MODEL_VALIDATE_MAX_DIMS    8

#define GGML_TYPE_F32   0
#define GGML_TYPE_F16   1
#define GGML_TYPE_Q4_0  2
#define GGML_TYPE_Q4_1  3
#define GGML_TYPE_Q5_0  6
#define GGML_TYPE_Q5_1  7
#define GGML_TYPE_Q8_0  8
#define GGML_TYPE_Q8_1  9
#define GGML_TYPE_I8    24
#define GGML_TYPE_I16   25
#define GGML_TYPE_I32   26
#define GGML_TYPE_I64   27
#define GGML_TYPE_F64   28
#define GGML_TYPE_BF16  30

#define GGUF_MODEL_VALIDATE_VERSION_V2 2
#define GGUF_MODEL_VALIDATE_VERSION_V3 3

#define GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_VERSION_SUPPORTED      (1 << 0)
#define GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_ALIGNMENT_POW2         (1 << 1)
#define GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_BASE_ALIGNED           (1 << 2)
#define GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_OFFSETS_ALIGNED        (1 << 3)
#define GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_ALL_OK \
    (GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_VERSION_SUPPORTED \
   | GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_ALIGNMENT_POW2 \
   | GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_BASE_ALIGNED \
   | GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_OFFSETS_ALIGNED)

Bool GGUFModelValidateVersionSupported(U32 version)
{
    return version == GGUF_MODEL_VALIDATE_VERSION_V2 ||
           version == GGUF_MODEL_VALIDATE_VERSION_V3;
}

Bool GGUFModelValidateIsPow2U32(U32 value)
{
    if (!value)
        return FALSE;
    return (value & (value - 1)) == 0;
}

Bool GGUFModelValidateTryAddU64(U64 a, U64 b, U64 *out_sum)
{
    if (!out_sum)
        return FALSE;
    if (a > 0xFFFFFFFFFFFFFFFF - b)
        return FALSE;

    *out_sum = a + b;
    return TRUE;
}

Bool GGUFModelValidateTryMulU64(U64 a, U64 b, U64 *out_prod)
{
    if (!out_prod)
        return FALSE;
    if (a && b > 0xFFFFFFFFFFFFFFFF / a)
        return FALSE;

    *out_prod = a * b;
    return TRUE;
}

Bool GGUFModelValidateTypeTraits(U32 ggml_type,
                                 U64 *out_block_elems,
                                 U64 *out_block_bytes)
{
    if (!out_block_elems || !out_block_bytes)
        return FALSE;

    switch (ggml_type)
    {
        case GGML_TYPE_F32:  *out_block_elems = 1;  *out_block_bytes = 4;  return TRUE;
        case GGML_TYPE_F16:  *out_block_elems = 1;  *out_block_bytes = 2;  return TRUE;
        case GGML_TYPE_BF16: *out_block_elems = 1;  *out_block_bytes = 2;  return TRUE;
        case GGML_TYPE_F64:  *out_block_elems = 1;  *out_block_bytes = 8;  return TRUE;
        case GGML_TYPE_I8:   *out_block_elems = 1;  *out_block_bytes = 1;  return TRUE;
        case GGML_TYPE_I16:  *out_block_elems = 1;  *out_block_bytes = 2;  return TRUE;
        case GGML_TYPE_I32:  *out_block_elems = 1;  *out_block_bytes = 4;  return TRUE;
        case GGML_TYPE_I64:  *out_block_elems = 1;  *out_block_bytes = 8;  return TRUE;

        // ggml quantized block layouts.
        case GGML_TYPE_Q4_0: *out_block_elems = 32; *out_block_bytes = 18; return TRUE;
        case GGML_TYPE_Q4_1: *out_block_elems = 32; *out_block_bytes = 20; return TRUE;
        case GGML_TYPE_Q5_0: *out_block_elems = 32; *out_block_bytes = 22; return TRUE;
        case GGML_TYPE_Q5_1: *out_block_elems = 32; *out_block_bytes = 24; return TRUE;
        case GGML_TYPE_Q8_0: *out_block_elems = 32; *out_block_bytes = 34; return TRUE;
        case GGML_TYPE_Q8_1: *out_block_elems = 32; *out_block_bytes = 36; return TRUE;
    }

    return FALSE;
}

I64 GGUFModelValidateTensorDataBytesChecked(U32 ggml_type,
                                            U64 n_dims,
                                            U64 *dims,
                                            U64 *out_payload_bytes)
{
    U64 block_elems;
    U64 block_bytes;
    U64 n_elems;
    U64 dim_i;
    U64 dim_value;
    U64 n_blocks;
    U64 bytes;

    if (!dims || !out_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_NULL_PTR;
    if (n_dims == 0 || n_dims > GGUF_MODEL_VALIDATE_MAX_DIMS)
        return GGUF_MODEL_VALIDATE_ERR_BAD_DIMS;

    if (!GGUFModelValidateTypeTraits(ggml_type, &block_elems, &block_bytes))
        return GGUF_MODEL_VALIDATE_ERR_BAD_TYPE;

    n_elems = 1;
    for (dim_i = 0; dim_i < n_dims; dim_i++)
    {
        dim_value = dims[dim_i];
        if (dim_value == 0)
            return GGUF_MODEL_VALIDATE_ERR_BAD_DIMS;
        if (!GGUFModelValidateTryMulU64(n_elems, dim_value, &n_elems))
            return GGUF_MODEL_VALIDATE_ERR_OVERFLOW;
    }

    if (block_elems == 1)
    {
        if (!GGUFModelValidateTryMulU64(n_elems, block_bytes, &bytes))
            return GGUF_MODEL_VALIDATE_ERR_OVERFLOW;
        *out_payload_bytes = bytes;
        return GGUF_MODEL_VALIDATE_OK;
    }

    if (n_elems % block_elems != 0)
        return GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT;

    n_blocks = n_elems / block_elems;
    if (!GGUFModelValidateTryMulU64(n_blocks, block_bytes, &bytes))
        return GGUF_MODEL_VALIDATE_ERR_OVERFLOW;

    *out_payload_bytes = bytes;
    return GGUF_MODEL_VALIDATE_OK;
}

// Validates tensor payload layout inside GGUF data region before model load.
// Each tensor consumes one row in all input arrays. Dimensions are flattened as
// `tensor_dims_flat[tensor_idx * dims_stride + dim_idx]`.
I64 GGUFModelValidateTensorLayoutCheckedNoPartial(U32 *tensor_types,
                                                  U64 *tensor_offsets,
                                                  U64 *tensor_n_dims,
                                                  U64 *tensor_dims_flat,
                                                  U64 tensor_count,
                                                  U64 dims_stride,
                                                  U64 data_region_nbytes,
                                                  U64 *out_validated_tensor_count,
                                                  U64 *out_total_payload_bytes)
{
    U64 tensor_i;
    U64 dim_base;
    U64 n_dims;
    U64 payload_bytes;
    U64 tensor_offset;
    U64 tensor_end;
    U64 prev_end;
    U64 staged_validated_count;
    U64 staged_total_payload;
    U64 next_total_payload;
    I64 status;

    if (!tensor_types || !tensor_offsets || !tensor_n_dims || !tensor_dims_flat ||
        !out_validated_tensor_count || !out_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_NULL_PTR;

    if (tensor_count > GGUF_MODEL_VALIDATE_MAX_TENSORS)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (dims_stride == 0 || dims_stride > GGUF_MODEL_VALIDATE_MAX_DIMS)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    prev_end = 0;
    staged_validated_count = 0;
    staged_total_payload = 0;

    for (tensor_i = 0; tensor_i < tensor_count; tensor_i++)
    {
        n_dims = tensor_n_dims[tensor_i];
        if (n_dims == 0 || n_dims > dims_stride)
            return GGUF_MODEL_VALIDATE_ERR_BAD_DIMS;

        if (!GGUFModelValidateTryMulU64(tensor_i, dims_stride, &dim_base))
            return GGUF_MODEL_VALIDATE_ERR_OVERFLOW;

        status = GGUFModelValidateTensorDataBytesChecked(
            tensor_types[tensor_i],
            n_dims,
            &tensor_dims_flat[dim_base],
            &payload_bytes);
        if (status != GGUF_MODEL_VALIDATE_OK)
            return status;

        tensor_offset = tensor_offsets[tensor_i];
        if (tensor_offset > data_region_nbytes)
            return GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS;
        if (!GGUFModelValidateTryAddU64(tensor_offset, payload_bytes, &tensor_end))
            return GGUF_MODEL_VALIDATE_ERR_OVERFLOW;
        if (tensor_end > data_region_nbytes)
            return GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS;

        // Monotonic offsets with no overlap in payload space.
        if (tensor_i > 0 && tensor_offset < prev_end)
            return GGUF_MODEL_VALIDATE_ERR_OVERLAP;

        if (!GGUFModelValidateTryAddU64(staged_total_payload, payload_bytes, &next_total_payload))
            return GGUF_MODEL_VALIDATE_ERR_OVERFLOW;

        staged_total_payload = next_total_payload;
        prev_end = tensor_end;
        staged_validated_count++;
    }

    *out_validated_tensor_count = staged_validated_count;
    *out_total_payload_bytes = staged_total_payload;
    return GGUF_MODEL_VALIDATE_OK;
}

// IQ-1120 diagnostics-only commit wrapper over tensor-layout validator.
//
// Contract:
// - Run canonical validator twice from immutable snapshots.
// - Enforce strict tuple parity on
//   `{validated_tensor_count,total_payload_bytes}`.
// - Publish outputs atomically only after snapshot+parity checks pass.
I64 GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnly(
    U32 *tensor_types,
    U64 *tensor_offsets,
    U64 *tensor_n_dims,
    U64 *tensor_dims_flat,
    U64 tensor_count,
    U64 dims_stride,
    U64 data_region_nbytes,
    U64 *out_validated_tensor_count,
    U64 *out_total_payload_bytes)
{
    U32 *snapshot_tensor_types;
    U64 *snapshot_tensor_offsets;
    U64 *snapshot_tensor_n_dims;
    U64 *snapshot_tensor_dims_flat;
    U64 snapshot_tensor_count;
    U64 snapshot_dims_stride;
    U64 snapshot_data_region_nbytes;
    U64 *snapshot_out_validated_tensor_count;
    U64 *snapshot_out_total_payload_bytes;

    U64 staged_validated_tensor_count;
    U64 staged_total_payload_bytes;
    U64 canonical_validated_tensor_count;
    U64 canonical_total_payload_bytes;

    I64 status;

    if (!tensor_types || !tensor_offsets || !tensor_n_dims || !tensor_dims_flat ||
        !out_validated_tensor_count || !out_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_NULL_PTR;

    if (out_validated_tensor_count == out_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if ((U8 *)out_validated_tensor_count == (U8 *)tensor_types ||
        (U8 *)out_validated_tensor_count == (U8 *)tensor_offsets ||
        (U8 *)out_validated_tensor_count == (U8 *)tensor_n_dims ||
        (U8 *)out_validated_tensor_count == (U8 *)tensor_dims_flat ||
        (U8 *)out_total_payload_bytes == (U8 *)tensor_types ||
        (U8 *)out_total_payload_bytes == (U8 *)tensor_offsets ||
        (U8 *)out_total_payload_bytes == (U8 *)tensor_n_dims ||
        (U8 *)out_total_payload_bytes == (U8 *)tensor_dims_flat)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (tensor_count > GGUF_MODEL_VALIDATE_MAX_TENSORS)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (dims_stride == 0 || dims_stride > GGUF_MODEL_VALIDATE_MAX_DIMS)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    snapshot_tensor_types = tensor_types;
    snapshot_tensor_offsets = tensor_offsets;
    snapshot_tensor_n_dims = tensor_n_dims;
    snapshot_tensor_dims_flat = tensor_dims_flat;
    snapshot_tensor_count = tensor_count;
    snapshot_dims_stride = dims_stride;
    snapshot_data_region_nbytes = data_region_nbytes;
    snapshot_out_validated_tensor_count = out_validated_tensor_count;
    snapshot_out_total_payload_bytes = out_total_payload_bytes;

    status = GGUFModelValidateTensorLayoutCheckedNoPartial(
        tensor_types,
        tensor_offsets,
        tensor_n_dims,
        tensor_dims_flat,
        tensor_count,
        dims_stride,
        data_region_nbytes,
        &staged_validated_tensor_count,
        &staged_total_payload_bytes);
    if (status != GGUF_MODEL_VALIDATE_OK)
        return status;

    status = GGUFModelValidateTensorLayoutCheckedNoPartial(
        tensor_types,
        tensor_offsets,
        tensor_n_dims,
        tensor_dims_flat,
        tensor_count,
        dims_stride,
        data_region_nbytes,
        &canonical_validated_tensor_count,
        &canonical_total_payload_bytes);
    if (status != GGUF_MODEL_VALIDATE_OK)
        return status;

    if (snapshot_tensor_types != tensor_types ||
        snapshot_tensor_offsets != tensor_offsets ||
        snapshot_tensor_n_dims != tensor_n_dims ||
        snapshot_tensor_dims_flat != tensor_dims_flat)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (snapshot_tensor_count != tensor_count ||
        snapshot_dims_stride != dims_stride ||
        snapshot_data_region_nbytes != data_region_nbytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (snapshot_out_validated_tensor_count != out_validated_tensor_count ||
        snapshot_out_total_payload_bytes != out_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (staged_validated_tensor_count != canonical_validated_tensor_count ||
        staged_total_payload_bytes != canonical_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (staged_validated_tensor_count > tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (canonical_validated_tensor_count > tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (tensor_count > 0 && staged_validated_tensor_count != tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (tensor_count > 0 && canonical_validated_tensor_count != tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (staged_total_payload_bytes > data_region_nbytes)
        return GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS;

    *out_validated_tensor_count = staged_validated_tensor_count;
    *out_total_payload_bytes = staged_total_payload_bytes;
    return GGUF_MODEL_VALIDATE_OK;
}


// IQ-1121 diagnostics parity gate over commit-only preflight wrapper.
//
// Contract:
// - Stage tuple via `...CommitOnlyPreflightOnly` from immutable snapshots.
// - Recompute canonical tuple via `...CheckedNoPartial`.
// - Publish only when strict tuple parity holds.
I64 GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnlyParity(
    U32 *tensor_types,
    U64 *tensor_offsets,
    U64 *tensor_n_dims,
    U64 *tensor_dims_flat,
    U64 tensor_count,
    U64 dims_stride,
    U64 data_region_nbytes,
    U64 *out_validated_tensor_count,
    U64 *out_total_payload_bytes)
{
    U32 *snapshot_tensor_types;
    U64 *snapshot_tensor_offsets;
    U64 *snapshot_tensor_n_dims;
    U64 *snapshot_tensor_dims_flat;
    U64 snapshot_tensor_count;
    U64 snapshot_dims_stride;
    U64 snapshot_data_region_nbytes;
    U64 *snapshot_out_validated_tensor_count;
    U64 *snapshot_out_total_payload_bytes;

    U64 staged_validated_tensor_count;
    U64 staged_total_payload_bytes;
    U64 canonical_validated_tensor_count;
    U64 canonical_total_payload_bytes;

    I64 status;

    if (!tensor_types || !tensor_offsets || !tensor_n_dims || !tensor_dims_flat ||
        !out_validated_tensor_count || !out_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_NULL_PTR;

    if (out_validated_tensor_count == out_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (tensor_count > GGUF_MODEL_VALIDATE_MAX_TENSORS)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (dims_stride == 0 || dims_stride > GGUF_MODEL_VALIDATE_MAX_DIMS)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    snapshot_tensor_types = tensor_types;
    snapshot_tensor_offsets = tensor_offsets;
    snapshot_tensor_n_dims = tensor_n_dims;
    snapshot_tensor_dims_flat = tensor_dims_flat;
    snapshot_tensor_count = tensor_count;
    snapshot_dims_stride = dims_stride;
    snapshot_data_region_nbytes = data_region_nbytes;
    snapshot_out_validated_tensor_count = out_validated_tensor_count;
    snapshot_out_total_payload_bytes = out_total_payload_bytes;

    status = GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnly(
        tensor_types,
        tensor_offsets,
        tensor_n_dims,
        tensor_dims_flat,
        tensor_count,
        dims_stride,
        data_region_nbytes,
        &staged_validated_tensor_count,
        &staged_total_payload_bytes);
    if (status != GGUF_MODEL_VALIDATE_OK)
        return status;

    status = GGUFModelValidateTensorLayoutCheckedNoPartial(
        tensor_types,
        tensor_offsets,
        tensor_n_dims,
        tensor_dims_flat,
        tensor_count,
        dims_stride,
        data_region_nbytes,
        &canonical_validated_tensor_count,
        &canonical_total_payload_bytes);
    if (status != GGUF_MODEL_VALIDATE_OK)
        return status;

    if (snapshot_tensor_types != tensor_types ||
        snapshot_tensor_offsets != tensor_offsets ||
        snapshot_tensor_n_dims != tensor_n_dims ||
        snapshot_tensor_dims_flat != tensor_dims_flat)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (snapshot_tensor_count != tensor_count ||
        snapshot_dims_stride != dims_stride ||
        snapshot_data_region_nbytes != data_region_nbytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (snapshot_out_validated_tensor_count != out_validated_tensor_count ||
        snapshot_out_total_payload_bytes != out_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (staged_validated_tensor_count != canonical_validated_tensor_count ||
        staged_total_payload_bytes != canonical_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (staged_validated_tensor_count > tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (staged_total_payload_bytes > data_region_nbytes)
        return GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS;

    *out_validated_tensor_count = staged_validated_tensor_count;
    *out_total_payload_bytes = staged_total_payload_bytes;
    return GGUF_MODEL_VALIDATE_OK;
}

// IQ-1122 commit-only diagnostics wrapper over tensor-layout parity gate.
//
// Contract:
// - Stage tuple through `...CommitOnlyPreflightOnlyParity`.
// - Recompute canonical tuple through `...CommitOnlyPreflightOnly`.
// - Enforce immutable input/output snapshots and strict tuple parity.
// - Publish atomically only after all checks pass.
I64 GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
    U32 *tensor_types,
    U64 *tensor_offsets,
    U64 *tensor_n_dims,
    U64 *tensor_dims_flat,
    U64 tensor_count,
    U64 dims_stride,
    U64 data_region_nbytes,
    U64 *out_validated_tensor_count,
    U64 *out_total_payload_bytes)
{
    U32 *snapshot_tensor_types;
    U64 *snapshot_tensor_offsets;
    U64 *snapshot_tensor_n_dims;
    U64 *snapshot_tensor_dims_flat;
    U64 snapshot_tensor_count;
    U64 snapshot_dims_stride;
    U64 snapshot_data_region_nbytes;
    U64 *snapshot_out_validated_tensor_count;
    U64 *snapshot_out_total_payload_bytes;

    U64 staged_validated_tensor_count;
    U64 staged_total_payload_bytes;
    U64 canonical_validated_tensor_count;
    U64 canonical_total_payload_bytes;

    I64 status;

    if (!tensor_types || !tensor_offsets || !tensor_n_dims || !tensor_dims_flat ||
        !out_validated_tensor_count || !out_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_NULL_PTR;

    if (out_validated_tensor_count == out_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if ((U8 *)out_validated_tensor_count == (U8 *)tensor_types ||
        (U8 *)out_validated_tensor_count == (U8 *)tensor_offsets ||
        (U8 *)out_validated_tensor_count == (U8 *)tensor_n_dims ||
        (U8 *)out_validated_tensor_count == (U8 *)tensor_dims_flat ||
        (U8 *)out_total_payload_bytes == (U8 *)tensor_types ||
        (U8 *)out_total_payload_bytes == (U8 *)tensor_offsets ||
        (U8 *)out_total_payload_bytes == (U8 *)tensor_n_dims ||
        (U8 *)out_total_payload_bytes == (U8 *)tensor_dims_flat)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (tensor_count > GGUF_MODEL_VALIDATE_MAX_TENSORS)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (dims_stride == 0 || dims_stride > GGUF_MODEL_VALIDATE_MAX_DIMS)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    snapshot_tensor_types = tensor_types;
    snapshot_tensor_offsets = tensor_offsets;
    snapshot_tensor_n_dims = tensor_n_dims;
    snapshot_tensor_dims_flat = tensor_dims_flat;
    snapshot_tensor_count = tensor_count;
    snapshot_dims_stride = dims_stride;
    snapshot_data_region_nbytes = data_region_nbytes;
    snapshot_out_validated_tensor_count = out_validated_tensor_count;
    snapshot_out_total_payload_bytes = out_total_payload_bytes;

    status = GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnlyParity(
        tensor_types,
        tensor_offsets,
        tensor_n_dims,
        tensor_dims_flat,
        tensor_count,
        dims_stride,
        data_region_nbytes,
        &staged_validated_tensor_count,
        &staged_total_payload_bytes);
    if (status != GGUF_MODEL_VALIDATE_OK)
        return status;

    status = GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnly(
        tensor_types,
        tensor_offsets,
        tensor_n_dims,
        tensor_dims_flat,
        tensor_count,
        dims_stride,
        data_region_nbytes,
        &canonical_validated_tensor_count,
        &canonical_total_payload_bytes);
    if (status != GGUF_MODEL_VALIDATE_OK)
        return status;

    if (snapshot_tensor_types != tensor_types ||
        snapshot_tensor_offsets != tensor_offsets ||
        snapshot_tensor_n_dims != tensor_n_dims ||
        snapshot_tensor_dims_flat != tensor_dims_flat)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (snapshot_tensor_count != tensor_count ||
        snapshot_dims_stride != dims_stride ||
        snapshot_data_region_nbytes != data_region_nbytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (snapshot_out_validated_tensor_count != out_validated_tensor_count ||
        snapshot_out_total_payload_bytes != out_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (staged_validated_tensor_count != canonical_validated_tensor_count ||
        staged_total_payload_bytes != canonical_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (staged_validated_tensor_count > tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (canonical_validated_tensor_count > tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (tensor_count > 0 && staged_validated_tensor_count != tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (tensor_count > 0 && canonical_validated_tensor_count != tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (staged_total_payload_bytes > data_region_nbytes)
        return GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS;

    *out_validated_tensor_count = staged_validated_tensor_count;
    *out_total_payload_bytes = staged_total_payload_bytes;
    return GGUF_MODEL_VALIDATE_OK;
}

// IQ-1127 diagnostics-only zero-write companion over tensor-layout
// parity-commit wrapper.
//
// Contract:
// - Stage tuple through `...ParityCommitOnly` into local scratch outputs.
// - Recompute canonical tuple through `...PreflightOnlyParity`.
// - Enforce immutable input/output snapshots and strict tuple parity.
// - Publish atomically only after exact tuple agreement.
I64 GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    U32 *tensor_types,
    U64 *tensor_offsets,
    U64 *tensor_n_dims,
    U64 *tensor_dims_flat,
    U64 tensor_count,
    U64 dims_stride,
    U64 data_region_nbytes,
    U64 *out_validated_tensor_count,
    U64 *out_total_payload_bytes)
{
    U32 *snapshot_tensor_types;
    U64 *snapshot_tensor_offsets;
    U64 *snapshot_tensor_n_dims;
    U64 *snapshot_tensor_dims_flat;
    U64 snapshot_tensor_count;
    U64 snapshot_dims_stride;
    U64 snapshot_data_region_nbytes;
    U64 *snapshot_out_validated_tensor_count;
    U64 *snapshot_out_total_payload_bytes;

    U64 staged_validated_tensor_count;
    U64 staged_total_payload_bytes;
    U64 canonical_validated_tensor_count;
    U64 canonical_total_payload_bytes;

    I64 status;

    if (!tensor_types || !tensor_offsets || !tensor_n_dims || !tensor_dims_flat ||
        !out_validated_tensor_count || !out_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_NULL_PTR;

    if (out_validated_tensor_count == out_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if ((U8 *)out_validated_tensor_count == (U8 *)tensor_types ||
        (U8 *)out_validated_tensor_count == (U8 *)tensor_offsets ||
        (U8 *)out_validated_tensor_count == (U8 *)tensor_n_dims ||
        (U8 *)out_validated_tensor_count == (U8 *)tensor_dims_flat ||
        (U8 *)out_total_payload_bytes == (U8 *)tensor_types ||
        (U8 *)out_total_payload_bytes == (U8 *)tensor_offsets ||
        (U8 *)out_total_payload_bytes == (U8 *)tensor_n_dims ||
        (U8 *)out_total_payload_bytes == (U8 *)tensor_dims_flat)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (tensor_count > GGUF_MODEL_VALIDATE_MAX_TENSORS)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (dims_stride == 0 || dims_stride > GGUF_MODEL_VALIDATE_MAX_DIMS)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    snapshot_tensor_types = tensor_types;
    snapshot_tensor_offsets = tensor_offsets;
    snapshot_tensor_n_dims = tensor_n_dims;
    snapshot_tensor_dims_flat = tensor_dims_flat;
    snapshot_tensor_count = tensor_count;
    snapshot_dims_stride = dims_stride;
    snapshot_data_region_nbytes = data_region_nbytes;
    snapshot_out_validated_tensor_count = out_validated_tensor_count;
    snapshot_out_total_payload_bytes = out_total_payload_bytes;

    status = GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        tensor_types,
        tensor_offsets,
        tensor_n_dims,
        tensor_dims_flat,
        tensor_count,
        dims_stride,
        data_region_nbytes,
        &staged_validated_tensor_count,
        &staged_total_payload_bytes);
    if (status != GGUF_MODEL_VALIDATE_OK)
        return status;

    status = GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnlyParity(
        tensor_types,
        tensor_offsets,
        tensor_n_dims,
        tensor_dims_flat,
        tensor_count,
        dims_stride,
        data_region_nbytes,
        &canonical_validated_tensor_count,
        &canonical_total_payload_bytes);
    if (status != GGUF_MODEL_VALIDATE_OK)
        return status;

    if (snapshot_tensor_types != tensor_types ||
        snapshot_tensor_offsets != tensor_offsets ||
        snapshot_tensor_n_dims != tensor_n_dims ||
        snapshot_tensor_dims_flat != tensor_dims_flat)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (snapshot_tensor_count != tensor_count ||
        snapshot_dims_stride != dims_stride ||
        snapshot_data_region_nbytes != data_region_nbytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (snapshot_out_validated_tensor_count != out_validated_tensor_count ||
        snapshot_out_total_payload_bytes != out_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (staged_validated_tensor_count != canonical_validated_tensor_count ||
        staged_total_payload_bytes != canonical_total_payload_bytes)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    if (staged_validated_tensor_count > tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (canonical_validated_tensor_count > tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (tensor_count > 0 && staged_validated_tensor_count != tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (tensor_count > 0 && canonical_validated_tensor_count != tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (staged_total_payload_bytes > data_region_nbytes)
        return GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS;

    *out_validated_tensor_count = staged_validated_tensor_count;
    *out_total_payload_bytes = staged_total_payload_bytes;
    return GGUF_MODEL_VALIDATE_OK;
}

// IQ-1164
// Integer-only rolling checksum over GGUF header bytes + tensor directory bytes.
//
// Design notes:
// - Uses an FNV-1a style byte mixer (xor + multiply) as the rolling core.
// - Injects absolute byte position into each mix step so reordered chunks diverge.
// - Applies a final integer avalanche to spread local byte changes across all bits.
// - Keeps no-partial-write semantics: `*out_checksum` is published only on success.
I64 GGUFModelChecksumHeaderAndTensorDirectory(U8 *header_bytes,
                                              U64 header_nbytes,
                                              U8 *tensor_dir_bytes,
                                              U64 tensor_dir_nbytes,
                                              U64 seed,
                                              U64 *out_checksum)
{
    U64 header_base;
    U64 header_end;
    U64 tensor_base;
    U64 tensor_end;
    U64 out_addr;
    U64 state;
    U64 i;
    U64 abs_pos;
    U64 len_mix;

    if (!header_bytes || !tensor_dir_bytes || !out_checksum)
        return GGUF_MODEL_VALIDATE_ERR_NULL_PTR;

    header_base = (U64)header_bytes;
    tensor_base = (U64)tensor_dir_bytes;
    out_addr = (U64)out_checksum;

    if (header_nbytes && !GGUFModelValidateTryAddU64(header_base, header_nbytes, &header_end))
        return GGUF_MODEL_VALIDATE_ERR_OVERFLOW;
    if (tensor_dir_nbytes && !GGUFModelValidateTryAddU64(tensor_base, tensor_dir_nbytes, &tensor_end))
        return GGUF_MODEL_VALIDATE_ERR_OVERFLOW;

    if ((header_nbytes && out_addr >= header_base && out_addr < header_end) ||
        (tensor_dir_nbytes && out_addr >= tensor_base && out_addr < tensor_end))
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    // Start from caller seed mixed with fixed offset-basis for deterministic domain separation.
    state = seed ^ 0xCBF29CE484222325;
    abs_pos = 0;

    for (i = 0; i < header_nbytes; i++)
    {
        state ^= (U64)header_bytes[i] ^ (abs_pos << 8);
        state *= 0x100000001B3;
        abs_pos++;
    }

    for (i = 0; i < tensor_dir_nbytes; i++)
    {
        state ^= (U64)tensor_dir_bytes[i] ^ (abs_pos << 8);
        state *= 0x100000001B3;
        abs_pos++;
    }

    // Fold both span lengths into final state before avalanche so equal prefixes diverge.
    len_mix = header_nbytes;
    if (!GGUFModelValidateTryAddU64(len_mix, (tensor_dir_nbytes << 1), &len_mix))
        return GGUF_MODEL_VALIDATE_ERR_OVERFLOW;
    state ^= len_mix;

    // Integer avalanche (Murmur-style constants) to distribute bit influence.
    state ^= state >> 33;
    state *= 0xFF51AFD7ED558CCD;
    state ^= state >> 33;
    state *= 0xC4CEB9FE1A85EC53;
    state ^= state >> 33;

    *out_checksum = state;
    return GGUF_MODEL_VALIDATE_OK;
}

// IQ-1165
// Validate GGUF version + tensor-data alignment invariants with commit-only
// status publication for callers that gate model load before tensor payload IO.
//
// Invariants enforced:
// - GGUF version is in the supported set {2,3}.
// - `tensor_data_alignment` is non-zero power-of-two.
// - `tensor_data_base` is aligned to `tensor_data_alignment`.
// - Every `tensor_offsets[i]` is aligned to `tensor_data_alignment`.
//
// No-partial contract:
// - `*out_status_bits` and `*out_aligned_tensor_count` are only published on
//   full success; any failure path leaves caller-owned outputs untouched.
I64 GGUFModelValidateVersionAndAlignmentCheckedNoPartial(
    U32 gguf_version,
    U32 tensor_data_alignment,
    U64 tensor_data_base,
    U64 *tensor_offsets,
    U64 tensor_count,
    U32 *out_status_bits,
    U64 *out_aligned_tensor_count)
{
    U64 tensor_i;
    U64 mask;
    U32 staged_status_bits;
    U64 staged_aligned_tensor_count;

    if (!out_status_bits || !out_aligned_tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_NULL_PTR;

    if (tensor_count > GGUF_MODEL_VALIDATE_MAX_TENSORS)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (tensor_count && !tensor_offsets)
        return GGUF_MODEL_VALIDATE_ERR_NULL_PTR;

    if (!GGUFModelValidateVersionSupported(gguf_version))
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    staged_status_bits = GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_VERSION_SUPPORTED;

    if (!GGUFModelValidateIsPow2U32(tensor_data_alignment))
        return GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT;
    staged_status_bits |= GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_ALIGNMENT_POW2;

    mask = (U64)tensor_data_alignment - 1;

    if (tensor_data_base & mask)
        return GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT;
    staged_status_bits |= GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_BASE_ALIGNED;

    staged_aligned_tensor_count = 0;
    for (tensor_i = 0; tensor_i < tensor_count; tensor_i++)
    {
        if (tensor_offsets[tensor_i] & mask)
            return GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT;
        staged_aligned_tensor_count++;
    }
    staged_status_bits |= GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_OFFSETS_ALIGNED;

    if (staged_status_bits != GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_ALL_OK)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;
    if (staged_aligned_tensor_count != tensor_count)
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM;

    *out_status_bits = staged_status_bits;
    *out_aligned_tensor_count = staged_aligned_tensor_count;
    return GGUF_MODEL_VALIDATE_OK;
}
