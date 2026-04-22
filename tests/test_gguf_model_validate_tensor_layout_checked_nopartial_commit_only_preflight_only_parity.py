#!/usr/bin/env python3
"""Harness for IQ-1121 tensor-layout commit-only preflight parity gate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only import (  # noqa: E402
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    GGUF_MODEL_VALIDATE_ERR_BAD_PARAM,
    GGUF_MODEL_VALIDATE_ERR_OVERLAP,
    GGUF_MODEL_VALIDATE_OK,
    GGUF_MODEL_VALIDATE_MAX_DIMS,
    GGUF_MODEL_VALIDATE_MAX_TENSORS,
    gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_reference,
    gguf_model_validate_tensor_layout_checked_nopartial_reference,
)


def gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity_reference(
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
        return 1

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

    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_reference(
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
        return 7

    out_validated_tensor_count_ref[0] = staged_validated[0]
    out_total_payload_bytes_ref[0] = staged_total[0]
    return GGUF_MODEL_VALIDATE_OK


def test_source_contains_iq1121_signature_and_contract() -> None:
    source = Path("src/gguf/validator.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert source.count(sig) == 1

    body = source.split(sig, 1)[1]
    assert "GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "GGUFModelValidateTensorLayoutCheckedNoPartial(" in body
    assert "snapshot_tensor_types != tensor_types" in body
    assert "staged_validated_tensor_count != canonical_validated_tensor_count" in body


def test_success_matches_canonical_publish() -> None:
    dims_stride = 4
    tensor_types = [GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F32]
    tensor_offsets = [0, 18, 52]
    tensor_n_dims = [1, 1, 2]
    tensor_dims_flat = [32, 0, 0, 0, 32, 0, 0, 0, 2, 3, 0, 0]

    out_validated = [777]
    out_total = [888]

    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity_reference(
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


def test_error_passthrough_and_no_partial() -> None:
    dims_stride = 2
    tensor_types = [GGML_TYPE_Q8_0, GGML_TYPE_F32]
    tensor_offsets = [0, 33]
    tensor_n_dims = [1, 1]
    tensor_dims_flat = [32, 0, 1, 0]

    out_validated = [123]
    out_total = [456]

    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity_reference(
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

    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity_reference(
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
    test_source_contains_iq1121_signature_and_contract()
    test_success_matches_canonical_publish()
    test_error_passthrough_and_no_partial()
    print("gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity=ok")


if __name__ == "__main__":
    run()
