#!/usr/bin/env python3
"""Harness for IQ-1122 tensor-layout parity commit-only wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gguf_model_validate_tensor_layout_checked_nopartial import (  # noqa: E402
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    GGUF_MODEL_VALIDATE_ERR_BAD_PARAM,
    GGUF_MODEL_VALIDATE_ERR_BAD_TYPE,
    GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS,
    GGUF_MODEL_VALIDATE_ERR_OVERLAP,
    GGUF_MODEL_VALIDATE_OK,
    GGUF_MODEL_VALIDATE_MAX_DIMS,
    GGUF_MODEL_VALIDATE_MAX_TENSORS,
)
from test_gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only import (  # noqa: E402
    gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_reference,
)
from test_gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity import (  # noqa: E402
    gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity_reference,
)


def gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
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
    if (
        out_validated_tensor_count_ref is tensor_types
        or out_validated_tensor_count_ref is tensor_offsets
        or out_validated_tensor_count_ref is tensor_n_dims
        or out_validated_tensor_count_ref is tensor_dims_flat
        or out_total_payload_bytes_ref is tensor_types
        or out_total_payload_bytes_ref is tensor_offsets
        or out_total_payload_bytes_ref is tensor_n_dims
        or out_total_payload_bytes_ref is tensor_dims_flat
    ):
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

    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity_reference(
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

    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_reference(
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
    if canonical_validated[0] > tensor_count:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM
    if tensor_count > 0 and staged_validated[0] != tensor_count:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM
    if tensor_count > 0 and canonical_validated[0] != tensor_count:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM
    if staged_total[0] > data_region_nbytes:
        return GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS

    out_validated_tensor_count_ref[0] = staged_validated[0]
    out_total_payload_bytes_ref[0] = staged_total[0]
    return GGUF_MODEL_VALIDATE_OK


def test_source_contains_iq1122_signature_and_contract() -> None:
    source = Path("src/gguf/validator.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly("
    assert source.count(sig) >= 1

    def_start = -1
    search_from = 0
    while True:
        idx = source.find(sig, search_from)
        if idx < 0:
            break
        brace_idx = source.find("{", idx)
        semi_idx = source.find(";", idx)
        if brace_idx >= 0 and (semi_idx < 0 or brace_idx < semi_idx):
            def_start = idx
            break
        search_from = idx + len(sig)

    assert def_start >= 0
    body = source[def_start:]
    assert "GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "GGUFModelValidateTensorLayoutCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "snapshot_tensor_types != tensor_types" in body
    assert "(U8 *)out_validated_tensor_count == (U8 *)tensor_offsets" in body
    assert "staged_validated_tensor_count != canonical_validated_tensor_count" in body
    assert "if (tensor_count > 0 && staged_validated_tensor_count != tensor_count)" in body


def test_success_matches_parity_and_preflight_publish() -> None:
    dims_stride = 4
    tensor_types = [GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F32]
    tensor_offsets = [0, 18, 52]
    tensor_n_dims = [1, 1, 2]
    tensor_dims_flat = [32, 0, 0, 0, 32, 0, 0, 0, 2, 3, 0, 0]

    out_validated = [900]
    out_total = [901]

    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
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


def test_overlap_order_adversarial_passthrough_and_no_partial() -> None:
    dims_stride = 2
    tensor_types = [GGML_TYPE_Q8_0, GGML_TYPE_F32]
    tensor_offsets = [0, 33]
    tensor_n_dims = [1, 1]
    tensor_dims_flat = [32, 0, 1, 0]

    out_validated = [123]
    out_total = [456]

    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
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


def test_type_mismatch_bad_type_passthrough() -> None:
    dims_stride = 2
    tensor_types = [777]
    tensor_offsets = [0]
    tensor_n_dims = [1]
    tensor_dims_flat = [4, 0]

    out_validated = [41]
    out_total = [42]

    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
        tensor_types=tensor_types,
        tensor_offsets=tensor_offsets,
        tensor_n_dims=tensor_n_dims,
        tensor_dims_flat=tensor_dims_flat,
        tensor_count=1,
        dims_stride=dims_stride,
        data_region_nbytes=256,
        out_validated_tensor_count_ref=out_validated,
        out_total_payload_bytes_ref=out_total,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_BAD_TYPE
    assert out_validated == [41]
    assert out_total == [42]


def test_out_of_bounds_adversarial_passthrough() -> None:
    dims_stride = 2
    tensor_types = [GGML_TYPE_F32]
    tensor_offsets = [9]
    tensor_n_dims = [2]
    tensor_dims_flat = [2, 2]

    out_validated = [71]
    out_total = [72]

    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
        tensor_types=tensor_types,
        tensor_offsets=tensor_offsets,
        tensor_n_dims=tensor_n_dims,
        tensor_dims_flat=tensor_dims_flat,
        tensor_count=1,
        dims_stride=dims_stride,
        data_region_nbytes=20,
        out_validated_tensor_count_ref=out_validated,
        out_total_payload_bytes_ref=out_total,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_OUT_OF_BOUNDS
    assert out_validated == [71]
    assert out_total == [72]


def test_alias_guard_rejected() -> None:
    dims_stride = 1
    tensor_types = [GGML_TYPE_Q4_0]
    tensor_offsets = [0]
    tensor_n_dims = [1]
    tensor_dims_flat = [32]

    shared = [555]
    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
        tensor_types=tensor_types,
        tensor_offsets=tensor_offsets,
        tensor_n_dims=tensor_n_dims,
        tensor_dims_flat=tensor_dims_flat,
        tensor_count=1,
        dims_stride=dims_stride,
        data_region_nbytes=18,
        out_validated_tensor_count_ref=shared,
        out_total_payload_bytes_ref=shared,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_BAD_PARAM



def test_output_input_alias_rejected() -> None:
    dims_stride = 1
    tensor_types = [GGML_TYPE_Q4_0]
    tensor_offsets = [0]
    tensor_n_dims = [1]
    tensor_dims_flat = [32]

    out_total = [91]

    rc = gguf_model_validate_tensor_layout_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
        tensor_types=tensor_types,
        tensor_offsets=tensor_offsets,
        tensor_n_dims=tensor_n_dims,
        tensor_dims_flat=tensor_dims_flat,
        tensor_count=1,
        dims_stride=dims_stride,
        data_region_nbytes=18,
        out_validated_tensor_count_ref=tensor_offsets,
        out_total_payload_bytes_ref=out_total,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_BAD_PARAM


def run() -> None:
    test_source_contains_iq1122_signature_and_contract()
    test_success_matches_parity_and_preflight_publish()
    test_overlap_order_adversarial_passthrough_and_no_partial()
    test_type_mismatch_bad_type_passthrough()
    test_out_of_bounds_adversarial_passthrough()
    test_alias_guard_rejected()
    test_output_input_alias_rejected()
    print(
        "gguf_model_validate_tensor_layout_checked_nopartial_"
        "commit_only_preflight_only_parity_commit_only=ok"
    )


if __name__ == "__main__":
    run()
