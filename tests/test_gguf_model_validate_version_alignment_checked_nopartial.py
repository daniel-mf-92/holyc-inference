#!/usr/bin/env python3
"""Harness for GGUFModelValidateVersionAndAlignmentCheckedNoPartial (IQ-1165)."""

from __future__ import annotations

from pathlib import Path

GGUF_MODEL_VALIDATE_OK = 0
GGUF_MODEL_VALIDATE_ERR_NULL_PTR = 1
GGUF_MODEL_VALIDATE_ERR_BAD_PARAM = 2
GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT = 6

GGUF_MODEL_VALIDATE_MAX_TENSORS = 1 << 20

GGUF_MODEL_VALIDATE_VERSION_V2 = 2
GGUF_MODEL_VALIDATE_VERSION_V3 = 3

GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_VERSION_SUPPORTED = 1 << 0
GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_ALIGNMENT_POW2 = 1 << 1
GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_BASE_ALIGNED = 1 << 2
GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_OFFSETS_ALIGNED = 1 << 3
GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_ALL_OK = (
    GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_VERSION_SUPPORTED
    | GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_ALIGNMENT_POW2
    | GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_BASE_ALIGNED
    | GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_OFFSETS_ALIGNED
)


def _is_pow2_u32(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def gguf_model_validate_version_alignment_checked_nopartial_reference(
    *,
    gguf_version: int,
    tensor_data_alignment: int,
    tensor_data_base: int,
    tensor_offsets: list[int] | None,
    tensor_count: int,
    out_status_bits_ref: list[int] | None,
    out_aligned_tensor_count_ref: list[int] | None,
) -> int:
    if out_status_bits_ref is None or out_aligned_tensor_count_ref is None:
        return GGUF_MODEL_VALIDATE_ERR_NULL_PTR

    if tensor_count > GGUF_MODEL_VALIDATE_MAX_TENSORS:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM
    if tensor_count > 0 and tensor_offsets is None:
        return GGUF_MODEL_VALIDATE_ERR_NULL_PTR

    if gguf_version not in (GGUF_MODEL_VALIDATE_VERSION_V2, GGUF_MODEL_VALIDATE_VERSION_V3):
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM
    staged_status = GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_VERSION_SUPPORTED

    if not _is_pow2_u32(tensor_data_alignment):
        return GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT
    staged_status |= GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_ALIGNMENT_POW2

    mask = tensor_data_alignment - 1

    if tensor_data_base & mask:
        return GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT
    staged_status |= GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_BASE_ALIGNED

    aligned_count = 0
    for tensor_i in range(tensor_count):
        assert tensor_offsets is not None
        if tensor_offsets[tensor_i] & mask:
            return GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT
        aligned_count += 1

    staged_status |= GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_OFFSETS_ALIGNED

    if staged_status != GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_ALL_OK:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM
    if aligned_count != tensor_count:
        return GGUF_MODEL_VALIDATE_ERR_BAD_PARAM

    out_status_bits_ref[0] = staged_status
    out_aligned_tensor_count_ref[0] = aligned_count
    return GGUF_MODEL_VALIDATE_OK


def test_source_contains_iq1165_signature_and_contract() -> None:
    source = Path("src/gguf/validator.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFModelValidateVersionAndAlignmentCheckedNoPartial("
    assert source.count(sig) >= 1

    body = source.split(sig, 1)[1]
    assert "GGUFModelValidateVersionSupported(gguf_version)" in body
    assert "GGUFModelValidateIsPow2U32(tensor_data_alignment)" in body
    assert "staged_status_bits |= GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_OFFSETS_ALIGNED" in body
    assert "*out_status_bits = staged_status_bits" in body
    assert "*out_aligned_tensor_count = staged_aligned_tensor_count" in body


def test_success_all_invariants_hold() -> None:
    out_status = [999]
    out_aligned = [888]

    rc = gguf_model_validate_version_alignment_checked_nopartial_reference(
        gguf_version=GGUF_MODEL_VALIDATE_VERSION_V3,
        tensor_data_alignment=32,
        tensor_data_base=0x1000,
        tensor_offsets=[0, 32, 64, 160],
        tensor_count=4,
        out_status_bits_ref=out_status,
        out_aligned_tensor_count_ref=out_aligned,
    )

    assert rc == GGUF_MODEL_VALIDATE_OK
    assert out_status == [GGUF_MODEL_VALIDATE_VERSION_ALIGN_STATUS_ALL_OK]
    assert out_aligned == [4]


def test_rejects_unsupported_version_no_partial_outputs() -> None:
    out_status = [11]
    out_aligned = [22]

    rc = gguf_model_validate_version_alignment_checked_nopartial_reference(
        gguf_version=1,
        tensor_data_alignment=32,
        tensor_data_base=0x1000,
        tensor_offsets=[0, 32],
        tensor_count=2,
        out_status_bits_ref=out_status,
        out_aligned_tensor_count_ref=out_aligned,
    )

    assert rc == GGUF_MODEL_VALIDATE_ERR_BAD_PARAM
    assert out_status == [11]
    assert out_aligned == [22]


def test_rejects_non_pow2_alignment_and_misaligned_base() -> None:
    out_status = [31]
    out_aligned = [41]

    rc = gguf_model_validate_version_alignment_checked_nopartial_reference(
        gguf_version=GGUF_MODEL_VALIDATE_VERSION_V2,
        tensor_data_alignment=24,
        tensor_data_base=0x1000,
        tensor_offsets=[0, 24],
        tensor_count=2,
        out_status_bits_ref=out_status,
        out_aligned_tensor_count_ref=out_aligned,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT
    assert out_status == [31]
    assert out_aligned == [41]

    rc = gguf_model_validate_version_alignment_checked_nopartial_reference(
        gguf_version=GGUF_MODEL_VALIDATE_VERSION_V2,
        tensor_data_alignment=32,
        tensor_data_base=0x1001,
        tensor_offsets=[0, 32],
        tensor_count=2,
        out_status_bits_ref=out_status,
        out_aligned_tensor_count_ref=out_aligned,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT
    assert out_status == [31]
    assert out_aligned == [41]


def test_rejects_misaligned_tensor_offset_no_partial() -> None:
    out_status = [71]
    out_aligned = [81]

    rc = gguf_model_validate_version_alignment_checked_nopartial_reference(
        gguf_version=GGUF_MODEL_VALIDATE_VERSION_V3,
        tensor_data_alignment=32,
        tensor_data_base=0x2000,
        tensor_offsets=[0, 32, 70],
        tensor_count=3,
        out_status_bits_ref=out_status,
        out_aligned_tensor_count_ref=out_aligned,
    )

    assert rc == GGUF_MODEL_VALIDATE_ERR_BAD_ALIGNMENT
    assert out_status == [71]
    assert out_aligned == [81]


def test_nulls_and_count_guards() -> None:
    out_status = [1]
    out_aligned = [2]

    rc = gguf_model_validate_version_alignment_checked_nopartial_reference(
        gguf_version=GGUF_MODEL_VALIDATE_VERSION_V2,
        tensor_data_alignment=32,
        tensor_data_base=0,
        tensor_offsets=None,
        tensor_count=1,
        out_status_bits_ref=out_status,
        out_aligned_tensor_count_ref=out_aligned,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_NULL_PTR
    assert out_status == [1]
    assert out_aligned == [2]

    rc = gguf_model_validate_version_alignment_checked_nopartial_reference(
        gguf_version=GGUF_MODEL_VALIDATE_VERSION_V2,
        tensor_data_alignment=32,
        tensor_data_base=0,
        tensor_offsets=[],
        tensor_count=GGUF_MODEL_VALIDATE_MAX_TENSORS + 1,
        out_status_bits_ref=out_status,
        out_aligned_tensor_count_ref=out_aligned,
    )
    assert rc == GGUF_MODEL_VALIDATE_ERR_BAD_PARAM
    assert out_status == [1]
    assert out_aligned == [2]


def run() -> None:
    test_source_contains_iq1165_signature_and_contract()
    test_success_all_invariants_hold()
    test_rejects_unsupported_version_no_partial_outputs()
    test_rejects_non_pow2_alignment_and_misaligned_base()
    test_rejects_misaligned_tensor_offset_no_partial()
    test_nulls_and_count_guards()
    print("gguf_model_validate_version_alignment_checked_nopartial=ok")


if __name__ == "__main__":
    run()
