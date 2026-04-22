#!/usr/bin/env python3
"""Parity harness for IQ-1094 dispatch-by-architecture forward wrapper."""

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


def byte_span_equals(lhs: bytes, rhs: bytes) -> bool:
    return lhs == rhs


def parse_architecture_id_checked(meta_key: bytes, arch_value: bytes) -> tuple[int, int]:
    if len(meta_key) <= 0 or len(arch_value) <= 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0
    if len(meta_key) != DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE_LEN:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0
    if not byte_span_equals(meta_key, DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE):
        return SAMPLING_Q16_ERR_BAD_PARAM, 0

    if byte_span_equals(arch_value, b"llama"):
        return SAMPLING_Q16_OK, DISPATCH_ARCH_Q16_ID_LLAMA
    if byte_span_equals(arch_value, b"mistral"):
        return SAMPLING_Q16_OK, DISPATCH_ARCH_Q16_ID_MISTRAL
    if byte_span_equals(arch_value, b"qwen2") or byte_span_equals(arch_value, b"qwen"):
        return SAMPLING_Q16_OK, DISPATCH_ARCH_Q16_ID_QWEN2
    if byte_span_equals(arch_value, b"phi3"):
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


def dispatch_model_arch_checked_nopartial_model(
    *,
    meta_key: bytes,
    arch_value: bytes,
    row_count: int,
    lane_count: int,
    out_rows_processed: list[int],
    out_logits_written: list[int],
    forward_status: int,
) -> tuple[int, int]:
    if out_rows_processed is out_logits_written:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0
    if row_count < 0 or lane_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0

    snap_rows = out_rows_processed[0]
    snap_logits = out_logits_written[0]

    status, arch_id = parse_architecture_id_checked(meta_key, arch_value)
    if status != SAMPLING_Q16_OK:
        return status, 0

    if arch_id not in {
        DISPATCH_ARCH_Q16_ID_LLAMA,
        DISPATCH_ARCH_Q16_ID_MISTRAL,
        DISPATCH_ARCH_Q16_ID_QWEN2,
        DISPATCH_ARCH_Q16_ID_PHI3,
    }:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0

    if forward_status != SAMPLING_Q16_OK:
        return forward_status, arch_id

    status, logits_written = try_mul_i64_checked(row_count, lane_count)
    if status != SAMPLING_Q16_OK:
        return status, arch_id

    out_rows_processed[0] = row_count
    out_logits_written[0] = logits_written

    assert out_rows_processed[0] != snap_rows or row_count == snap_rows
    assert out_logits_written[0] != snap_logits or logits_written == snap_logits
    return SAMPLING_Q16_OK, arch_id


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


def test_source_contains_explicit_dispatch_cases_and_qwen_alias() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceForwardDispatchModelArchQ16CheckedNoPartial("
    assert signature in source

    body = _extract_function_body(source, signature)
    assert "switch (arch_id)" in body
    assert "case DISPATCH_ARCH_Q16_ID_LLAMA:" in body
    assert "case DISPATCH_ARCH_Q16_ID_MISTRAL:" in body
    assert "case DISPATCH_ARCH_Q16_ID_QWEN2:" in body
    assert "case DISPATCH_ARCH_Q16_ID_PHI3:" in body
    assert "InferenceForwardQ16CheckedNoPartial(" in body
    assert "DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE_LEN" in source
    assert "(U8 *)\"qwen2\"" in source
    assert "(U8 *)\"qwen\"" in source


def test_arch_metadata_key_requires_exact_length() -> None:
    rows_out = [11]
    logits_out = [22]

    status, _ = dispatch_model_arch_checked_nopartial_model(
        meta_key=b"general.architectur",  # len 19
        arch_value=b"llama",
        row_count=3,
        lane_count=5,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
        forward_status=SAMPLING_Q16_OK,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows_out == [11]
    assert logits_out == [22]

    status, _ = dispatch_model_arch_checked_nopartial_model(
        meta_key=b"general.architectureX",  # len 21
        arch_value=b"llama",
        row_count=3,
        lane_count=5,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
        forward_status=SAMPLING_Q16_OK,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows_out == [11]
    assert logits_out == [22]




def test_negative_geometry_rejected_without_publish() -> None:
    rows_out = [61]
    logits_out = [67]

    status, _ = dispatch_model_arch_checked_nopartial_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"llama",
        row_count=-1,
        lane_count=8,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
        forward_status=SAMPLING_Q16_OK,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows_out == [61]
    assert logits_out == [67]

    status, _ = dispatch_model_arch_checked_nopartial_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"llama",
        row_count=4,
        lane_count=-2,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
        forward_status=SAMPLING_Q16_OK,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows_out == [61]
    assert logits_out == [67]

def test_supported_architectures_publish_atomic_diagnostics() -> None:
    vectors = [
        (b"llama", DISPATCH_ARCH_Q16_ID_LLAMA),
        (b"mistral", DISPATCH_ARCH_Q16_ID_MISTRAL),
        (b"qwen2", DISPATCH_ARCH_Q16_ID_QWEN2),
        (b"phi3", DISPATCH_ARCH_Q16_ID_PHI3),
        (b"qwen", DISPATCH_ARCH_Q16_ID_QWEN2),
    ]

    for tag, expected_arch_id in vectors:
        rows_out = [123]
        logits_out = [456]
        status, arch_id = dispatch_model_arch_checked_nopartial_model(
            meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
            arch_value=tag,
            row_count=7,
            lane_count=13,
            out_rows_processed=rows_out,
            out_logits_written=logits_out,
            forward_status=SAMPLING_Q16_OK,
        )
        assert status == SAMPLING_Q16_OK
        assert arch_id == expected_arch_id
        assert rows_out[0] == 7
        assert logits_out[0] == 91


def test_unsupported_tag_and_bad_key_leave_outputs_unchanged() -> None:
    rows_out = [99]
    logits_out = [77]

    status, _ = dispatch_model_arch_checked_nopartial_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"gemma",
        row_count=3,
        lane_count=5,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
        forward_status=SAMPLING_Q16_OK,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows_out[0] == 99
    assert logits_out[0] == 77

    status, _ = dispatch_model_arch_checked_nopartial_model(
        meta_key=b"general.arch",
        arch_value=b"llama",
        row_count=3,
        lane_count=5,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
        forward_status=SAMPLING_Q16_OK,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows_out[0] == 99
    assert logits_out[0] == 77


def test_forward_failure_and_multiply_overflow_are_no_partial() -> None:
    rows_out = [11]
    logits_out = [22]

    status, arch_id = dispatch_model_arch_checked_nopartial_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"llama",
        row_count=9,
        lane_count=9,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
        forward_status=SAMPLING_Q16_ERR_BAD_PARAM,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert arch_id == DISPATCH_ARCH_Q16_ID_LLAMA
    assert rows_out[0] == 11
    assert logits_out[0] == 22

    status, arch_id = dispatch_model_arch_checked_nopartial_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"phi3",
        row_count=1 << 62,
        lane_count=8,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
        forward_status=SAMPLING_Q16_OK,
    )
    assert status == SAMPLING_Q16_ERR_OVERFLOW
    assert arch_id == DISPATCH_ARCH_Q16_ID_PHI3
    assert rows_out[0] == 11
    assert logits_out[0] == 22


def test_deterministic_dispatch_vectors() -> None:
    rng = random.Random(1094)
    tags = [b"llama", b"mistral", b"qwen2", b"phi3", b"qwen"]
    for _ in range(80):
        rows = rng.randint(0, 1024)
        lanes = rng.randint(0, 1024)
        tag = rng.choice(tags)
        rows_out = [-1]
        logits_out = [-1]

        status, _ = dispatch_model_arch_checked_nopartial_model(
            meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
            arch_value=tag,
            row_count=rows,
            lane_count=lanes,
            out_rows_processed=rows_out,
            out_logits_written=logits_out,
            forward_status=SAMPLING_Q16_OK,
        )
        assert status == SAMPLING_Q16_OK
        assert rows_out[0] == rows
        assert logits_out[0] == rows * lanes


if __name__ == "__main__":
    test_source_contains_explicit_dispatch_cases_and_qwen_alias()
    test_arch_metadata_key_requires_exact_length()
    test_negative_geometry_rejected_without_publish()
    test_supported_architectures_publish_atomic_diagnostics()
    test_unsupported_tag_and_bad_key_leave_outputs_unchanged()
    test_forward_failure_and_multiply_overflow_are_no_partial()
    test_deterministic_dispatch_vectors()
    print("ok")
