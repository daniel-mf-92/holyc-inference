#!/usr/bin/env python3
"""Harness for IQ-1120 tensor-layout commit-only preflight wrapper."""

from __future__ import annotations

from pathlib import Path

GGUF_MODEL_VALIDATE_OK = 0
GGUF_MODEL_VALIDATE_ERR_NULL_PTR = 1
GGUF_MODEL_VALIDATE_ERR_BAD_PARAM = 2
GGUF_MODEL_VALIDATE_ERR_OVERFLOW = 3
GGUF_MODEL_VALIDATE_ERR_BAD_TYPE = 4
GGUF_MODEL_VALIDATE_ERR_BAD_DIMS = 5
GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT = 6
GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS = 7
GGUF_MODEL_VALIDATE_ERR_OVERLAP = 8

GGUF_MODEL_VALIDATE_MAX_TENSORS = 1 << 20
GGUF_MODEL_VALIDATE_MAX_DIMS = 8
U64_MAX = (1 << 64) - 1

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_I8 = 24


def _u64_add(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None
    if a > U64_MAX or b > U64_MAX:
        return None
    if a > U64_MAX - b:
        return None
    return a + b


def _u64_mul(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None
    if a > U64_MAX or b > U64_MAX:
        return None
    if a != 0 and b > U64_MAX // a:
        return None
    return a * b


def _type_traits(ggml_type: int) -> tuple[int, int] | None:
    table = {
        GGML_TYPE_F32: (1, 4),
        GGML_TYPE_F16: (1, 2),
        GGML_TYPE_I8: (1, 1),
        GGML_TYPE_Q4_0: (32, 18),
        GGML_TYPE_Q8_0: (32, 34),
    }
    return table.get(ggml_type)


def tensor_data_bytes_checked_reference(*, ggml_type: int, n_dims: int, dims: list[int]) -> tuple[int, int]:
    if n_dims <= 0 or n_dims > GGUF_MODEL_VALIDATE_MAX_DIMS:
        return GGUF_MODEL_VALIDATE_ERR_BAD_DIMS, 0

    traits = _type_traits(ggml_type)
    if traits is None:
        return GGUF_MODEL_VALIDATE_ERR_BAD_TYPE, 0
    block_elems, block_bytes = traits

    n_elems = 1
    for idx in range(n_dims):
        dim_value = dims[idx]
        if dim_value == 0:
            return GGUF_MODEL_VALIDATE_ERR_BAD_DIMS, 0
        next_elems = _u64_mul(n_elems, dim_value)
        if next_elems is None:
            return GGUF_MODEL_VALIDATE_ERR_OVERFLOW, 0
        n_elems = next_elems

    if block_elems == 1:
        payload = _u64_mul(n_elems, block_bytes)
        if payload is None:
            return GGUF_MODEL_VALIDATE_ERR_OVERFLOW, 0
        return GGUF_MODEL_VALIDATE_OK, payload

    if n_elems % block_elems != 0:
        return GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT, 0

    n_blocks = n_elems // block_elems
    payload = _u64_mul(n_blocks, block_bytes)
    if payload is None:
        return GGUF_MODEL_VALIDATE_ERR_OVERFLOW, 0
    return GGUF_MODEL_VALIDATE_OK, payload


def gguf_model_validate_tensor_layout_checked_nopartial_reference(
    *,
    tensor_types: list[int] | None,
    tensor_offsets: list[int] | None,
    tensor_n_dims: list[int] | None,
    tensor_dims_flat: list[int] | None,
    tensor_count: int,
    dims_stride: int,
    data_region_nbytes: int,
    out_validated_tensor_count_ref: list[int] | None,
    out_total_payload_bytes_ref: list[int] | None,
) -> int:
    if (
        tensor_types is None
        or tensor_offsets is None
        or tensor_n_dims is None
        or tensor_dims_flat is None
        or out_validated_tensor_count_ref is None
        or out_total_payload_bytes_ref is None
    ):
        return GGUF_MODEL_VALIDATE_ERR_NULL_PTR

    if tensor_count > GGUF_MODEL_VALIDATE_MAX_TENSORS:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM
    if dims_stride == 0 or dims_stride > GGUF_MODEL_VALIDATE_MAX_DIMS:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM

    prev_end = 0
    staged_validated_count = 0
    staged_total_payload = 0

    for tensor_i in range(tensor_count):
        n_dims = tensor_n_dims[tensor_i]
        if n_dims == 0 or n_dims > dims_stride:
            return GGUF_MODEL_VALIDATE_ERR_BAD_DIMS

        dim_base = _u64_mul(tensor_i, dims_stride)
        if dim_base is None:
            return GGUF_MODEL_VALIDATE_ERR_OVERFLOW

        rc, payload_bytes = tensor_data_bytes_checked_reference(
            ggml_type=tensor_types[tensor_i],
            n_dims=n_dims,
            dims=tensor_dims_flat[dim_base : dim_base + n_dims],
        )
        if rc != GGUF_MODEL_VALIDATE_OK:
            return rc

        tensor_offset = tensor_offsets[tensor_i]
        if tensor_offset > data_region_nbytes:
            return GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS
        tensor_end = _u64_add(tensor_offset, payload_bytes)
        if tensor_end is None:
            return GGUF_MODEL_VALIDATE_ERR_OVERFLOW
        if tensor_end > data_region_nbytes:
            return GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS

        if tensor_i > 0 and tensor_offset < prev_end:
            return GGUF_MODEL_VALIDATE_ERR_OVERLAP

        next_payload_total = _u64_add(staged_total_payload, payload_bytes)
        if next_payload_total is None:
            return GGUF_MODEL_VALIDATE_ERR_OVERFLOW

        staged_total_payload = next_payload_total
        prev_end = tensor_end
        staged_validated_count += 1

    out_validated_tensor_count_ref[0] = staged_validated_count
    out_total_payload_bytes_ref[0] = staged_total_payload
    return GGUF_MODEL_VALIDATE_OK


def gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_reference(
    *,
    tensor_types: list[int] | None,
    tensor_offsets: list[int] | None,
    tensor_n_dims: list[int] | None,
    tensor_dims_flat: list[int] | None,
    tensor_count: int,
    dims_stride: int,
    data_region_nbytes: int,
    out_validated_tensor_count_ref: list[int] | None,
    out_total_payload_bytes_ref: list[int] | None,
) -> int:
    if (
        tensor_types is None
        or tensor_offsets is None
        or tensor_n_dims is None
        or tensor_dims_flat is None
        or out_validated_tensor_count_ref is None
        or out_total_payload_bytes_ref is None
    ):
        return GGUF_MODEL_VALIDATE_ERR_NULL_PTR

    if out_validated_tensor_count_ref is out_total_payload_bytes_ref:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM

    if tensor_count > GGUF_MODEL_VALIDATE_MAX_TENSORS:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM
    if dims_stride == 0 or dims_stride > GGUF_MODEL_VALIDATE_MAX_DIMS:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM

    snap_tensor_types = tensor_types
    snap_tensor_offsets = tensor_offsets
    snap_tensor_n_dims = tensor_n_dims
    snap_tensor_dims_flat = tensor_dims_flat
    snap_tensor_count = tensor_count
    snap_dims_stride = dims_stride
    snap_data_region_nbytes = data_region_nbytes
    snap_out_validated = out_validated_tensor_count_ref
    snap_out_total = out_total_payload_bytes_ref

    staged_validated = [0]
    staged_total = [0]
    canonical_validated = [0]
    canonical_total = [0]

    rc = gguf_model_validate_tensor_layout_checked_nopartial_reference(
        tensor_types=tensor_types,
        tensor_offsets=tensor_offsets,
        tensor_n_dims=tensor_n_dims,
        tensor_dims_flat=tensor_dims_flat,
        tensor_count=tensor_count,
        dims_stride=dims_stride,
        data_region_nbytes=data_region_nbytes,
        out_validated_tensor_count_ref=staged_validated,
        out_total_payload_bytes_ref=staged_total,
    )
    if rc != GGUF_MODEL_VALIDATE_OK:
        return rc

    rc = gguf_model_validate_tensor_layout_checked_nopartial_reference(
        tensor_types=tensor_types,
        tensor_offsets=tensor_offsets,
        tensor_n_dims=tensor_n_dims,
        tensor_dims_flat=tensor_dims_flat,
        tensor_count=tensor_count,
        dims_stride=dims_stride,
        data_region_nbytes=data_region_nbytes,
        out_validated_tensor_count_ref=canonical_validated,
        out_total_payload_bytes_ref=canonical_total,
    )
    if rc != GGUF_MODEL_VALIDATE_OK:
        return rc

    if (
        snap_tensor_types is not tensor_types
        or snap_tensor_offsets is not tensor_offsets
        or snap_tensor_n_dims is not tensor_n_dims
        or snap_tensor_dims_flat is not tensor_dims_flat
    ):
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM

    if (
        snap_tensor_count != tensor_count
        or snap_dims_stride != dims_stride
        or snap_data_region_nbytes != data_region_nbytes
    ):
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM

    if snap_out_validated is not out_validated_tensor_count_ref or snap_out_total is not out_total_payload_bytes_ref:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM

    if staged_validated[0] != canonical_validated[0] or staged_total[0] != canonical_total[0]:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM

    if staged_validated[0] > tensor_count:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM
    if staged_total[0] > data_region_nbytes:
        return GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS

    out_validated_tensor_count_ref[0] = staged_validated[0]
    out_total_payload_bytes_ref[0] = staged_total[0]
    return GGUF_MODEL_VALIDATE_OK


def test_source_contains_iq1120_signature_and_contract() -> None:
    source = Path("src/gguf/validator.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnly("
    assert source.count(sig) == 1

    body = source.split(sig, 1)[1]
    assert "GGUFModelValidateTensorLayoutCheckedNoPartial(" in body
    assert "snapshot_tensor_types != tensor_types" in body
    assert "staged_validated_tensor_count != canonical_validated_tensor_count" in body
    assert "if (out_validated_tensor_count == out_total_payload_bytes)" in body


def test_success_publishes_expected_tuple() -> None:
    dims_stride = 4
    tensor_types = [GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F32]
    tensor_offsets = [0, 18, 52]
    tensor_n_dims = [1, 1, 2]
    tensor_dims_flat = [32, 0, 0, 0, 32, 0, 0, 0, 2, 3, 0, 0]

    out_validated = [777]
    out_total = [888]

    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_reference(
        tensor_types=tensor_types,
        tensor_offsets=tensor_offsets,
        tensor_n_dims=tensor_n_dims,
        tensor_dims_flat=tensor_dims_flat,
        tensor_count=3,
        dims_stride=dims_stride,
        data_region_nbytes=76,
        out_validated_tensor_count_ref=out_validated,
        out_total_payload_bytes_ref=out_total,
    )

    assert rc == GGUF_MODEL_VALIDATE_OK
    assert out_validated == [3]
    assert out_total == [76]


def test_bad_param_alias_and_no_partial_on_failure() -> None:
    dims_stride = 2
    tensor_types = [GGML_TYPE_Q8_0, GGML_TYPE_F32]
    tensor_offsets = [0, 33]
    tensor_n_dims = [1, 1]
    tensor_dims_flat = [32, 0, 1, 0]

    out_validated = [123]
    out_total = [456]

    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_reference(
        tensor_types=tensor_types,
        tensor_offsets=tensor_offsets,
        tensor_n_dims=tensor_n_dims,
        tensor_dims_flat=tensor_dims_flat,
        tensor_count=2,
        dims_stride=dims_stride,
        data_region_nbytes=64,
        out_validated_tensor_count_ref=out_validated,
        out_total_payload_bytes_ref=out_total,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_OVERLAP
    assert out_validated == [123]
    assert out_total == [456]

    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_reference(
        tensor_types=tensor_types,
        tensor_offsets=tensor_offsets,
        tensor_n_dims=tensor_n_dims,
        tensor_dims_flat=tensor_dims_flat,
        tensor_count=2,
        dims_stride=dims_stride,
        data_region_nbytes=64,
        out_validated_tensor_count_ref=out_validated,
        out_total_payload_bytes_ref=out_validated,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_BAD_PARAM


def run() -> None:
    test_source_contains_iq1120_signature_and_contract()
    test_success_publishes_expected_tuple()
    test_bad_param_alias_and_no_partial_on_failure()
    print("gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only=ok")


if __name__ == "__main__":
    run()
