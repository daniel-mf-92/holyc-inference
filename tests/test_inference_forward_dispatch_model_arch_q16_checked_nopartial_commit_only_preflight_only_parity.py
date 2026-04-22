#!/usr/bin/env python3
"""Parity harness for IQ-1111 dispatch parity gate wrapper."""

from __future__ import annotations

from pathlib import Path

from test_inference_forward_dispatch_model_arch_q16_checked_nopartial_commit_only_preflight_only import (
    DISPATCH_ARCH_Q16_ID_LLAMA,
    DISPATCH_ARCH_Q16_ID_MISTRAL,
    DISPATCH_ARCH_Q16_ID_PHI3,
    DISPATCH_ARCH_Q16_ID_QWEN2,
    DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
    SAMPLING_Q16_ERR_BAD_PARAM,
    SAMPLING_Q16_ERR_OVERFLOW,
    SAMPLING_Q16_ERR_NULL_PTR,
    SAMPLING_Q16_OK,
    dispatch_model_arch_commit_only_preflight_only_model,
    parse_architecture_id_checked,
    try_mul_i64_checked,
)


def dispatch_model_arch_commit_only_preflight_only_parity_model(
    *,
    meta_key: bytes | None,
    arch_value: bytes | None,
    row_count: int,
    lane_count: int,
    out_rows_processed: list[int] | None,
    out_logits_written: list[int] | None,
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
    snapshot_meta_key_len = len(meta_key)
    snapshot_arch_value = arch_value
    snapshot_arch_value_len = len(arch_value)
    snapshot_row_count = row_count
    snapshot_lane_count = lane_count
    snapshot_out_rows_processed = out_rows_processed
    snapshot_out_logits_written = out_logits_written

    staged_rows = [0]
    staged_logits = [0]
    status, _ = dispatch_model_arch_commit_only_preflight_only_model(
        meta_key=meta_key,
        arch_value=arch_value,
        row_count=row_count,
        lane_count=lane_count,
        out_rows_processed=staged_rows,
        out_logits_written=staged_logits,
    )
    if status != SAMPLING_Q16_OK:
        return status

    status, canonical_arch = parse_architecture_id_checked(snapshot_meta_key, snapshot_arch_value)
    if status != SAMPLING_Q16_OK:
        return status

    if canonical_arch not in {
        DISPATCH_ARCH_Q16_ID_LLAMA,
        DISPATCH_ARCH_Q16_ID_MISTRAL,
        DISPATCH_ARCH_Q16_ID_QWEN2,
        DISPATCH_ARCH_Q16_ID_PHI3,
    }:
        return SAMPLING_Q16_ERR_BAD_PARAM

    status, canonical_logits = try_mul_i64_checked(snapshot_row_count, snapshot_lane_count)
    if status != SAMPLING_Q16_OK:
        return status

    canonical_rows = snapshot_row_count

    if (
        snapshot_meta_key is not meta_key
        or snapshot_meta_key_len != len(meta_key)
        or snapshot_arch_value is not arch_value
        or snapshot_arch_value_len != len(arch_value)
        or snapshot_row_count != row_count
        or snapshot_lane_count != lane_count
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if (
        snapshot_out_rows_processed is not out_rows_processed
        or snapshot_out_logits_written is not out_logits_written
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if staged_rows[0] != canonical_rows or staged_logits[0] != canonical_logits:
        return SAMPLING_Q16_ERR_BAD_PARAM

    out_rows_processed[0] = staged_rows[0]
    out_logits_written[0] = staged_logits[0]
    return SAMPLING_Q16_OK


def test_source_contains_iq1111_signature_and_composition() -> None:
    src = Path("src/model/inference.HC").read_text(encoding="utf-8")
    sig = "I32 InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert src.count(sig) == 1
    body = src.split(sig, 1)[1].split("I32 InferenceGenerateTokensPreflightChecked(", 1)[0]

    assert "InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "InferenceDispatchParseArchitectureIdChecked" in body
    assert "InferenceDispatchTryMulI64Checked(" in body
    assert "if (snapshot_meta_key_bytes != meta_key_bytes ||" in body
    assert "if (snapshot_out_rows_processed != out_rows_processed ||" in body
    assert "if (staged_rows_processed != canonical_rows_processed ||" in body


def test_supported_architectures_publish_expected_tuple() -> None:
    vectors = [b"llama", b"mistral", b"qwen2", b"qwen", b"phi3"]

    for tag in vectors:
        out_rows = [700]
        out_logits = [800]
        rc = dispatch_model_arch_commit_only_preflight_only_parity_model(
            meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
            arch_value=tag,
            row_count=17,
            lane_count=19,
            out_rows_processed=out_rows,
            out_logits_written=out_logits,
        )
        assert rc == SAMPLING_Q16_OK
        assert out_rows[0] == 17
        assert out_logits[0] == 323


def test_unsupported_and_bad_key_no_partial_publish() -> None:
    out_rows = [91]
    out_logits = [92]

    rc = dispatch_model_arch_commit_only_preflight_only_parity_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"gemma",
        row_count=3,
        lane_count=5,
        out_rows_processed=out_rows,
        out_logits_written=out_logits,
    )
    assert rc == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_rows == [91]
    assert out_logits == [92]

    rc = dispatch_model_arch_commit_only_preflight_only_parity_model(
        meta_key=b"general.arch",
        arch_value=b"llama",
        row_count=3,
        lane_count=5,
        out_rows_processed=out_rows,
        out_logits_written=out_logits,
    )
    assert rc == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_rows == [91]
    assert out_logits == [92]


def test_alias_output_rejected_without_publish() -> None:
    shared = [55]
    rc = dispatch_model_arch_commit_only_preflight_only_parity_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"llama",
        row_count=4,
        lane_count=6,
        out_rows_processed=shared,
        out_logits_written=shared,
    )
    assert rc == SAMPLING_Q16_ERR_BAD_PARAM
    assert shared == [55]


def test_overflow_rejected_without_publish() -> None:
    out_rows = [61]
    out_logits = [62]

    rc = dispatch_model_arch_commit_only_preflight_only_parity_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"llama",
        row_count=(1 << 62),
        lane_count=4,
        out_rows_processed=out_rows,
        out_logits_written=out_logits,
    )
    assert rc == SAMPLING_Q16_ERR_OVERFLOW
    assert out_rows == [61]
    assert out_logits == [62]


def run() -> None:
    test_source_contains_iq1111_signature_and_composition()
    test_supported_architectures_publish_expected_tuple()
    test_unsupported_and_bad_key_no_partial_publish()
    test_alias_output_rejected_without_publish()
    test_overflow_rejected_without_publish()
    print("inference_forward_dispatch_model_arch_q16_checked_nopartial_commit_only_preflight_only_parity=ok")


if __name__ == "__main__":
    run()
