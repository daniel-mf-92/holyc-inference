#!/usr/bin/env python3
"""Harness for InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnly (IQ-1108)."""

from __future__ import annotations

from pathlib import Path
import random
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from test_inference_forward_dispatch_model_arch_q16_checked_nopartial import (
    DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
    I64_MAX,
    SAMPLING_Q16_ERR_BAD_PARAM,
    SAMPLING_Q16_ERR_NULL_PTR,
    SAMPLING_Q16_ERR_OVERFLOW,
    SAMPLING_Q16_OK,
    dispatch_model_arch_checked_nopartial_model,
    parse_architecture_id_checked,
    try_mul_i64_checked,
)


def inference_forward_dispatch_model_arch_q16_checked_nopartial_commit_only_preflight_only_model(
    *,
    meta_key: bytes | None,
    arch_value: bytes | None,
    row_count: int,
    lane_count: int,
    out_rows_processed: list[int] | None,
    out_logits_written: list[int] | None,
    forward_status: int,
) -> int:
    if (
        meta_key is None
        or arch_value is None
        or out_rows_processed is None
        or out_logits_written is None
    ):
        return SAMPLING_Q16_ERR_NULL_PTR

    if out_rows_processed is out_logits_written:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if row_count < 0 or lane_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM

    snapshot_meta_key = meta_key
    snapshot_arch_value = arch_value
    snapshot_row_count = row_count
    snapshot_lane_count = lane_count
    snapshot_out_rows = out_rows_processed
    snapshot_out_logits = out_logits_written

    staged_rows = [0]
    staged_logits = [0]
    status, _ = dispatch_model_arch_checked_nopartial_model(
        meta_key=meta_key,
        arch_value=arch_value,
        row_count=row_count,
        lane_count=lane_count,
        out_rows_processed=staged_rows,
        out_logits_written=staged_logits,
        forward_status=forward_status,
    )
    if status != SAMPLING_Q16_OK:
        return status

    status, _ = parse_architecture_id_checked(snapshot_meta_key, snapshot_arch_value)
    if status != SAMPLING_Q16_OK:
        return status

    canonical_rows = snapshot_row_count
    status, canonical_logits = try_mul_i64_checked(snapshot_row_count, snapshot_lane_count)
    if status != SAMPLING_Q16_OK:
        return status

    if (
        snapshot_meta_key is not meta_key
        or snapshot_arch_value is not arch_value
        or snapshot_row_count != row_count
        or snapshot_lane_count != lane_count
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if snapshot_out_rows is not out_rows_processed or snapshot_out_logits is not out_logits_written:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if staged_rows[0] < 0 or staged_logits[0] < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if staged_rows[0] != canonical_rows or staged_logits[0] != canonical_logits:
        return SAMPLING_Q16_ERR_BAD_PARAM

    out_rows_processed[0] = staged_rows[0]
    out_logits_written[0] = staged_logits[0]
    return SAMPLING_Q16_OK


def test_source_contains_signature_and_contract_guards() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    sig = "I32 InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnly("
    assert source.count(sig) == 1
    body = source.split(sig, 1)[1].split(
        "I32 InferenceGenerateTokensPreflightChecked(",
        1,
    )[0]
    assert "InferenceForwardDispatchModelArchQ16CheckedNoPartial(" in body
    assert "InferenceDispatchParseArchitectureIdChecked(snapshot_meta_key_bytes," in body
    assert "status = InferenceDispatchTryMulI64Checked(row_count," in body
    assert "if (staged_rows_processed < 0 || staged_logits_written < 0)" in body


def test_nulls_and_aliases_rejected_no_partial() -> None:
    rows = [111]
    logits = [222]

    status = inference_forward_dispatch_model_arch_q16_checked_nopartial_commit_only_preflight_only_model(
        meta_key=None,
        arch_value=b"llama",
        row_count=4,
        lane_count=8,
        out_rows_processed=rows,
        out_logits_written=logits,
        forward_status=SAMPLING_Q16_OK,
    )
    assert status == SAMPLING_Q16_ERR_NULL_PTR
    assert rows == [111]
    assert logits == [222]

    status = inference_forward_dispatch_model_arch_q16_checked_nopartial_commit_only_preflight_only_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"llama",
        row_count=4,
        lane_count=8,
        out_rows_processed=rows,
        out_logits_written=rows,
        forward_status=SAMPLING_Q16_OK,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows == [111]


def test_supported_architectures_publish_expected_tuple() -> None:
    vectors = [b"llama", b"mistral", b"qwen2", b"qwen", b"phi3"]

    for arch in vectors:
        rows = [7]
        logits = [9]
        status = inference_forward_dispatch_model_arch_q16_checked_nopartial_commit_only_preflight_only_model(
            meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
            arch_value=arch,
            row_count=6,
            lane_count=11,
            out_rows_processed=rows,
            out_logits_written=logits,
            forward_status=SAMPLING_Q16_OK,
        )
        assert status == SAMPLING_Q16_OK
        assert rows == [6]
        assert logits == [66]


def test_unsupported_arch_and_forward_failures_are_no_partial() -> None:
    rows = [333]
    logits = [444]

    status = inference_forward_dispatch_model_arch_q16_checked_nopartial_commit_only_preflight_only_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"gemma",
        row_count=3,
        lane_count=5,
        out_rows_processed=rows,
        out_logits_written=logits,
        forward_status=SAMPLING_Q16_OK,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows == [333]
    assert logits == [444]

    status = inference_forward_dispatch_model_arch_q16_checked_nopartial_commit_only_preflight_only_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"llama",
        row_count=3,
        lane_count=5,
        out_rows_processed=rows,
        out_logits_written=logits,
        forward_status=SAMPLING_Q16_ERR_BAD_PARAM,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows == [333]
    assert logits == [444]


def test_overflow_and_negative_geometry_rejected_no_partial() -> None:
    rows = [12]
    logits = [34]

    status = inference_forward_dispatch_model_arch_q16_checked_nopartial_commit_only_preflight_only_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"phi3",
        row_count=(1 << 62),
        lane_count=8,
        out_rows_processed=rows,
        out_logits_written=logits,
        forward_status=SAMPLING_Q16_OK,
    )
    assert status == SAMPLING_Q16_ERR_OVERFLOW
    assert rows == [12]
    assert logits == [34]

    status = inference_forward_dispatch_model_arch_q16_checked_nopartial_commit_only_preflight_only_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"llama",
        row_count=-1,
        lane_count=8,
        out_rows_processed=rows,
        out_logits_written=logits,
        forward_status=SAMPLING_Q16_OK,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows == [12]
    assert logits == [34]


def test_deterministic_dispatch_vectors() -> None:
    rng = random.Random(20260422_1108)
    tags = [b"llama", b"mistral", b"qwen2", b"qwen", b"phi3"]

    for _ in range(256):
        row_count = rng.randint(0, 4096)
        lane_count = rng.randint(0, 4096)
        tag = rng.choice(tags)
        rows = [I64_MAX]
        logits = [I64_MAX]

        status = inference_forward_dispatch_model_arch_q16_checked_nopartial_commit_only_preflight_only_model(
            meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
            arch_value=tag,
            row_count=row_count,
            lane_count=lane_count,
            out_rows_processed=rows,
            out_logits_written=logits,
            forward_status=SAMPLING_Q16_OK,
        )

        assert status == SAMPLING_Q16_OK
        assert rows[0] == row_count
        assert logits[0] == row_count * lane_count


if __name__ == "__main__":
    test_source_contains_signature_and_contract_guards()
    test_nulls_and_aliases_rejected_no_partial()
    test_supported_architectures_publish_expected_tuple()
    test_unsupported_arch_and_forward_failures_are_no_partial()
    test_overflow_and_negative_geometry_rejected_no_partial()
    test_deterministic_dispatch_vectors()
    print("ok")
