#!/usr/bin/env python3
"""Harness for GGUFModelValidateTensorLayoutCheckedNoPartial (IQ-1100)."""

from __future__ import annotations

import random
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
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_BF16 = 30


def _type_traits(ggml_type: int) -> tuple[int, int] | None:
    traits = {
        GGML_TYPE_F32: (1, 4),
        GGML_TYPE_F16: (1, 2),
        GGML_TYPE_BF16: (1, 2),
        GGML_TYPE_F64: (1, 8),
        GGML_TYPE_I8: (1, 1),
        GGML_TYPE_I16: (1, 2),
        GGML_TYPE_I32: (1, 4),
        GGML_TYPE_I64: (1, 8),
        GGML_TYPE_Q4_0: (32, 18),
        GGML_TYPE_Q4_1: (32, 20),
        GGML_TYPE_Q5_0: (32, 22),
        GGML_TYPE_Q5_1: (32, 24),
        GGML_TYPE_Q8_0: (32, 34),
        GGML_TYPE_Q8_1: (32, 36),
    }
    return traits.get(ggml_type)


def _u64_add(a: int, b: int) -> int | None:
    s = a + b
    if s > U64_MAX:
        return None
    return s


def _u64_mul(a: int, b: int) -> int | None:
    p = a * b
    if p > U64_MAX:
        return None
    return p


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
    if dims_stride <= 0 or dims_stride > GGUF_MODEL_VALIDATE_MAX_DIMS:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM

    prev_end = 0
    staged_validated_count = 0
    staged_total_payload = 0

    for tensor_i in range(tensor_count):
        n_dims = tensor_n_dims[tensor_i]
        if n_dims <= 0 or n_dims > dims_stride:
            return GGUF_MODEL_VALIDATE_ERR_BAD_DIMS

        dim_base = _u64_mul(tensor_i, dims_stride)
        if dim_base is None:
            return GGUF_MODEL_VALIDATE_ERR_OVERFLOW

        traits = _type_traits(tensor_types[tensor_i])
        if traits is None:
            return GGUF_MODEL_VALIDATE_ERR_BAD_TYPE
        block_elems, block_bytes = traits

        n_elems = 1
        for dim_i in range(n_dims):
            dim_value = tensor_dims_flat[dim_base + dim_i]
            if dim_value == 0:
                return GGUF_MODEL_VALIDATE_ERR_BAD_DIMS
            n_elems_next = _u64_mul(n_elems, dim_value)
            if n_elems_next is None:
                return GGUF_MODEL_VALIDATE_ERR_OVERFLOW
            n_elems = n_elems_next

        if block_elems == 1:
            payload_bytes = _u64_mul(n_elems, block_bytes)
            if payload_bytes is None:
                return GGUF_MODEL_VALIDATE_ERR_OVERFLOW
        else:
            if n_elems % block_elems != 0:
                return GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT
            n_blocks = n_elems // block_elems
            payload_bytes = _u64_mul(n_blocks, block_bytes)
            if payload_bytes is None:
                return GGUF_MODEL_VALIDATE_ERR_OVERFLOW

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


def _run_case(
    tensor_types: list[int],
    tensor_offsets: list[int],
    tensor_n_dims: list[int],
    tensor_dims_flat: list[int],
    tensor_count: int,
    dims_stride: int,
    data_region_nbytes: int,
) -> tuple[int, int, int]:
    out_validated = [777]
    out_total = [888]
    rc = gguf_model_validate_tensor_layout_checked_nopartial_reference(
        tensor_types=tensor_types,
        tensor_offsets=tensor_offsets,
        tensor_n_dims=tensor_n_dims,
        tensor_dims_flat=tensor_dims_flat,
        tensor_count=tensor_count,
        dims_stride=dims_stride,
        data_region_nbytes=data_region_nbytes,
        out_validated_tensor_count_ref=out_validated,
        out_total_payload_bytes_ref=out_total,
    )
    return rc, out_validated[0], out_total[0]


def test_success_mixed_tensor_layout_and_payload_sum() -> None:
    dims_stride = 4
    tensor_types = [GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F32]
    tensor_offsets = [0, 18, 52]
    tensor_n_dims = [1, 1, 2]
    tensor_dims_flat = [32, 0, 0, 0, 32, 0, 0, 0, 2, 3, 0, 0]

    rc, validated, payload = _run_case(
        tensor_types, tensor_offsets, tensor_n_dims, tensor_dims_flat, 3, dims_stride, 76
    )

    assert rc == GGUF_MODEL_VALIDATE_OK
    assert validated == 3
    assert payload == 76


def test_overlap_rejected() -> None:
    dims_stride = 2
    tensor_types = [GGML_TYPE_Q8_0, GGML_TYPE_F32]
    tensor_offsets = [0, 33]
    tensor_n_dims = [1, 1]
    tensor_dims_flat = [32, 0, 1, 0]

    rc, validated, payload = _run_case(
        tensor_types, tensor_offsets, tensor_n_dims, tensor_dims_flat, 2, dims_stride, 64
    )

    assert rc == GGUF_MODEL_VALIDATE_ERR_OVERLAP
    assert validated == 777
    assert payload == 888


def test_quantized_alignment_mismatch_rejected() -> None:
    dims_stride = 2
    tensor_types = [GGML_TYPE_Q4_0]
    tensor_offsets = [0]
    tensor_n_dims = [1]
    tensor_dims_flat = [33, 0]

    rc, _, _ = _run_case(tensor_types, tensor_offsets, tensor_n_dims, tensor_dims_flat, 1, dims_stride, 256)
    assert rc == GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT


def test_out_of_bounds_rejected() -> None:
    dims_stride = 2
    tensor_types = [GGML_TYPE_F32]
    tensor_offsets = [8]
    tensor_n_dims = [2]
    tensor_dims_flat = [2, 2]

    rc, _, _ = _run_case(tensor_types, tensor_offsets, tensor_n_dims, tensor_dims_flat, 1, dims_stride, 20)
    assert rc == GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS


def test_dim_multiply_overflow_rejected() -> None:
    dims_stride = 2
    tensor_types = [GGML_TYPE_F32]
    tensor_offsets = [0]
    tensor_n_dims = [2]
    tensor_dims_flat = [1 << 63, 3]

    rc, _, _ = _run_case(
        tensor_types,
        tensor_offsets,
        tensor_n_dims,
        tensor_dims_flat,
        1,
        dims_stride,
        U64_MAX,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_OVERFLOW


def test_randomized_valid_monotonic_layouts() -> None:
    rng = random.Random(20260422_1100)

    for _ in range(250):
        tensor_count = rng.randint(1, 18)
        dims_stride = 4
        tensor_types: list[int] = []
        tensor_offsets: list[int] = []
        tensor_n_dims: list[int] = []
        tensor_dims_flat: list[int] = [0] * (tensor_count * dims_stride)

        cursor = 0
        expected_total = 0

        for tidx in range(tensor_count):
            ggml_type = rng.choice([GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I8])
            block_elems, block_bytes = _type_traits(ggml_type)  # type: ignore[misc]

            if block_elems == 1:
                n_dims = rng.randint(1, 3)
                dims = [rng.randint(1, 5) for _ in range(n_dims)]
                n_elems = 1
                for d in dims:
                    n_elems *= d
                payload = n_elems * block_bytes
            else:
                n_dims = 1
                n_blocks = rng.randint(1, 4)
                dims = [n_blocks * block_elems]
                payload = n_blocks * block_bytes

            tensor_types.append(ggml_type)
            tensor_offsets.append(cursor)
            tensor_n_dims.append(n_dims)
            for d_i, d in enumerate(dims):
                tensor_dims_flat[tidx * dims_stride + d_i] = d

            cursor += payload
            expected_total += payload

        rc, validated, payload = _run_case(
            tensor_types,
            tensor_offsets,
            tensor_n_dims,
            tensor_dims_flat,
            tensor_count,
            dims_stride,
            cursor,
        )

        assert rc == GGUF_MODEL_VALIDATE_OK
        assert validated == tensor_count
        assert payload == expected_total


def test_source_contains_iq1100_function_and_layout_checks() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "gguf" / "validator.HC"
    body = src.read_text(encoding="utf-8")

    assert "I64 GGUFModelValidateTensorLayoutCheckedNoPartial(U32 *tensor_types," in body
    assert "GGUFModelValidateTensorDataBytesChecked(" in body
    assert "if (tensor_i > 0 && tensor_offset < prev_end)" in body
    assert "GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT" in body


def run() -> None:
    test_success_mixed_tensor_layout_and_payload_sum()
    test_overlap_rejected()
    test_quantized_alignment_mismatch_rejected()
    test_out_of_bounds_rejected()
    test_dim_multiply_overflow_rejected()
    test_randomized_valid_monotonic_layouts()
    test_source_contains_iq1100_function_and_layout_checks()
    print("gguf_model_validate_tensor_layout_checked_nopartial=ok")


if __name__ == "__main__":
    run()
