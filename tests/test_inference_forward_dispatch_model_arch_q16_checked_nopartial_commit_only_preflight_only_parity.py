#!/usr/bin/env python3
"""Parity harness for IQ-1111 dispatch preflight-only parity gate."""

from __future__ import annotations

from pathlib import Path
import random

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
SAMPLING_Q16_ERR_BAD_PARAM = 2
SAMPLING_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)

DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE = b"general.architecture"
DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE_LEN = 20

DISPATCH_ARCH_Q16_ID_LLAMA = 1
DISPATCH_ARCH_Q16_ID_MISTRAL = 2
DISPATCH_ARCH_Q16_ID_QWEN2 = 3
DISPATCH_ARCH_Q16_ID_PHI3 = 4


def parse_architecture_id_checked(meta_key: bytes, arch_value: bytes) -> tuple[int, int]:
    if len(meta_key) <= 0 or len(arch_value) <= 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0
    if len(meta_key) != DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE_LEN:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0
    if meta_key != DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0

    if arch_value == b"llama":
        return SAMPLING_Q16_OK, DISPATCH_ARCH_Q16_ID_LLAMA
    if arch_value == b"mistral":
        return SAMPLING_Q16_OK, DISPATCH_ARCH_Q16_ID_MISTRAL
    if arch_value == b"qwen2" or arch_value == b"qwen":
        return SAMPLING_Q16_OK, DISPATCH_ARCH_Q16_ID_QWEN2
    if arch_value == b"phi3":
        return SAMPLING_Q16_OK, DISPATCH_ARCH_Q16_ID_PHI3

    return SAMPLING_Q16_ERR_BAD_PARAM, 0


def try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if lhs == 0 or rhs == 0:
        return SAMPLING_Q16_OK, 0

    if lhs > 0:
        if rhs > 0:
            if lhs > I64_MAX // rhs:
                return SAMPLING_Q16_ERR_OVERFLOW, 0
        else:
            if rhs < I64_MIN // lhs:
                return SAMPLING_Q16_ERR_OVERFLOW, 0
    else:
        if rhs > 0:
            if lhs < I64_MIN // rhs:
                return SAMPLING_Q16_ERR_OVERFLOW, 0
        else:
            if lhs != 0 and rhs < I64_MAX // lhs:
                return SAMPLING_Q16_ERR_OVERFLOW, 0

    return SAMPLING_Q16_OK, lhs * rhs


def dispatch_model_arch_commit_only_preflight_only_reference(
    *,
    meta_key: bytes,
    arch_value: bytes,
    row_count: int,
    lane_count: int,
    out_rows_processed: list[int],
    out_logits_written: list[int],
) -> tuple[int, int, int, int]:
    if out_rows_processed is out_logits_written:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0, 0, 0
    if row_count < 0 or lane_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0, 0, 0

    status, staged_arch_id = parse_architecture_id_checked(meta_key, arch_value)
    if status != SAMPLING_Q16_OK:
        return status, 0, 0, 0

    if staged_arch_id not in {
        DISPATCH_ARCH_Q16_ID_LLAMA,
        DISPATCH_ARCH_Q16_ID_MISTRAL,
        DISPATCH_ARCH_Q16_ID_QWEN2,
        DISPATCH_ARCH_Q16_ID_PHI3,
    }:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0, 0, 0

    status, staged_logits_written = try_mul_i64_checked(row_count, lane_count)
    if status != SAMPLING_Q16_OK:
        return status, staged_arch_id, 0, 0

    status, canonical_arch_id = parse_architecture_id_checked(meta_key, arch_value)
    if status != SAMPLING_Q16_OK:
        return status, staged_arch_id, 0, 0

    if canonical_arch_id not in {
        DISPATCH_ARCH_Q16_ID_LLAMA,
        DISPATCH_ARCH_Q16_ID_MISTRAL,
        DISPATCH_ARCH_Q16_ID_QWEN2,
        DISPATCH_ARCH_Q16_ID_PHI3,
    }:
        return SAMPLING_Q16_ERR_BAD_PARAM, staged_arch_id, 0, 0

    status, canonical_logits_written = try_mul_i64_checked(row_count, lane_count)
    if status != SAMPLING_Q16_OK:
        return status, staged_arch_id, 0, 0

    if staged_arch_id != canonical_arch_id:
        return SAMPLING_Q16_ERR_BAD_PARAM, staged_arch_id, 0, 0
    if staged_logits_written != canonical_logits_written:
        return SAMPLING_Q16_ERR_BAD_PARAM, staged_arch_id, 0, 0

    out_rows_processed[0] = row_count
    out_logits_written[0] = staged_logits_written
    return SAMPLING_Q16_OK, staged_arch_id, row_count, staged_logits_written


def dispatch_model_arch_commit_only_preflight_only_parity_reference(
    *,
    meta_key: bytes,
    arch_value: bytes,
    row_count: int,
    lane_count: int,
    out_rows_processed: list[int],
    out_logits_written: list[int],
    preflight_only_fn=dispatch_model_arch_commit_only_preflight_only_reference,
) -> tuple[int, int]:
    if out_rows_processed is out_logits_written:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0
    if row_count < 0 or lane_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0

    snapshot_meta_key = meta_key
    snapshot_arch_value = arch_value
    snapshot_row_count = row_count
    snapshot_lane_count = lane_count
    snapshot_out_rows_processed = out_rows_processed
    snapshot_out_logits_written = out_logits_written

    staged_rows = [0]
    staged_logits = [0]
    status, staged_arch_id, _, _ = preflight_only_fn(
        meta_key=meta_key,
        arch_value=arch_value,
        row_count=row_count,
        lane_count=lane_count,
        out_rows_processed=staged_rows,
        out_logits_written=staged_logits,
    )
    if status != SAMPLING_Q16_OK:
        return status, staged_arch_id

    if staged_rows[0] < 0 or staged_logits[0] < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, staged_arch_id

    status, canonical_arch_id = parse_architecture_id_checked(snapshot_meta_key, snapshot_arch_value)
    if status != SAMPLING_Q16_OK:
        return status, staged_arch_id
    if canonical_arch_id not in {
        DISPATCH_ARCH_Q16_ID_LLAMA,
        DISPATCH_ARCH_Q16_ID_MISTRAL,
        DISPATCH_ARCH_Q16_ID_QWEN2,
        DISPATCH_ARCH_Q16_ID_PHI3,
    }:
        return SAMPLING_Q16_ERR_BAD_PARAM, staged_arch_id

    status, canonical_logits = try_mul_i64_checked(snapshot_row_count, snapshot_lane_count)
    if status != SAMPLING_Q16_OK:
        return status, staged_arch_id

    if (
        snapshot_meta_key is not meta_key
        or snapshot_arch_value is not arch_value
        or snapshot_row_count != row_count
        or snapshot_lane_count != lane_count
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM, staged_arch_id

    if (
        snapshot_out_rows_processed is not out_rows_processed
        or snapshot_out_logits_written is not out_logits_written
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM, staged_arch_id

    if (
        staged_arch_id != canonical_arch_id
        or staged_rows[0] != snapshot_row_count
        or staged_logits[0] != canonical_logits
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM, staged_arch_id

    out_rows_processed[0] = staged_rows[0]
    out_logits_written[0] = staged_logits[0]
    return SAMPLING_Q16_OK, staged_arch_id


def _extract_function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    index = brace + 1
    while depth:
        ch = source[index]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        index += 1
    return source[brace + 1 : index - 1]


def test_source_contains_iq1111_function_and_composition_calls() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    sig = "I32 InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert sig in source

    body = _extract_function_body(source, sig)
    assert "InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "InferenceDispatchParseArchitectureIdChecked" in body
    assert "InferenceDispatchTryMulI64Checked(" in body
    assert "staged_arch_id != canonical_arch_id" in body
    assert "snapshot_out_rows_processed != out_rows_processed" in body


def test_supported_architectures_publish_expected_tuple() -> None:
    vectors = [
        (b"llama", DISPATCH_ARCH_Q16_ID_LLAMA),
        (b"mistral", DISPATCH_ARCH_Q16_ID_MISTRAL),
        (b"qwen2", DISPATCH_ARCH_Q16_ID_QWEN2),
        (b"qwen", DISPATCH_ARCH_Q16_ID_QWEN2),
        (b"phi3", DISPATCH_ARCH_Q16_ID_PHI3),
    ]

    for arch_tag, expected_arch_id in vectors:
        rows_out = [901]
        logits_out = [902]
        status, arch_id = dispatch_model_arch_commit_only_preflight_only_parity_reference(
            meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
            arch_value=arch_tag,
            row_count=17,
            lane_count=19,
            out_rows_processed=rows_out,
            out_logits_written=logits_out,
        )
        assert status == SAMPLING_Q16_OK
        assert arch_id == expected_arch_id
        assert rows_out[0] == 17
        assert logits_out[0] == 323


def test_unsupported_tag_bad_key_and_negative_geometry_no_publish() -> None:
    rows_out = [71]
    logits_out = [73]

    status, _ = dispatch_model_arch_commit_only_preflight_only_parity_reference(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"gemma",
        row_count=9,
        lane_count=10,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows_out[0] == 71
    assert logits_out[0] == 73

    status, _ = dispatch_model_arch_commit_only_preflight_only_parity_reference(
        meta_key=b"general.arch",
        arch_value=b"llama",
        row_count=9,
        lane_count=10,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows_out[0] == 71
    assert logits_out[0] == 73

    status, _ = dispatch_model_arch_commit_only_preflight_only_parity_reference(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"llama",
        row_count=-1,
        lane_count=10,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows_out[0] == 71
    assert logits_out[0] == 73


def test_overflow_and_alias_safe_snapshot_vectors() -> None:
    rows_out = [111]
    logits_out = [222]

    status, arch_id = dispatch_model_arch_commit_only_preflight_only_parity_reference(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"phi3",
        row_count=1 << 62,
        lane_count=8,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
    )
    assert status == SAMPLING_Q16_ERR_OVERFLOW
    assert arch_id == DISPATCH_ARCH_Q16_ID_PHI3
    assert rows_out[0] == 111
    assert logits_out[0] == 222

    aliased = [17]
    status, _ = dispatch_model_arch_commit_only_preflight_only_parity_reference(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"llama",
        row_count=4,
        lane_count=5,
        out_rows_processed=aliased,
        out_logits_written=aliased,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM


def test_deterministic_dispatch_vectors() -> None:
    rng = random.Random(1111)
    tags = [b"llama", b"mistral", b"qwen2", b"qwen", b"phi3"]

    for _ in range(120):
        rows = rng.randint(0, 2000)
        lanes = rng.randint(0, 2000)
        tag = rng.choice(tags)
        rows_out = [-1]
        logits_out = [-1]

        status, _ = dispatch_model_arch_commit_only_preflight_only_parity_reference(
            meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
            arch_value=tag,
            row_count=rows,
            lane_count=lanes,
            out_rows_processed=rows_out,
            out_logits_written=logits_out,
        )
        assert status == SAMPLING_Q16_OK
        assert rows_out[0] == rows
        assert logits_out[0] == rows * lanes


if __name__ == "__main__":
    test_source_contains_iq1111_function_and_composition_calls()
    test_supported_architectures_publish_expected_tuple()
    test_unsupported_tag_bad_key_and_negative_geometry_no_publish()
    test_overflow_and_alias_safe_snapshot_vectors()
    test_deterministic_dispatch_vectors()
    print("ok")
