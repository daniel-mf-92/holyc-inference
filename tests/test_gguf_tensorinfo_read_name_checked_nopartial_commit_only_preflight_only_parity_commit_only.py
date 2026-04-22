#!/usr/bin/env python3
"""Harness for GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly (IQ-1028)."""

from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_gguf_tensor_info_read_checked_nopartial import (  # noqa: E402
    GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN,
    GGUF_TENSOR_PARSE_ERR_NULL_PTR,
    GGUF_TENSOR_PARSE_ERR_TRUNCATED,
    GGUF_TENSOR_PARSE_OK,
)
from test_gguf_tensorinfo_read_name_checked_nopartial import (  # noqa: E402
    GGUF_TENSOR_MAX_NAME_BYTES,
)
from test_gguf_tensorinfo_read_name_checked_nopartial_commit_only_preflight_only_parity import (  # noqa: E402
    parse_name_checked_nopartial_commit_only_preflight_only_parity,
)

U64_MAX = (1 << 64) - 1
I64_MAX = (1 << 63) - 1


def _try_add_u64(a: int, b: int, out: list[int]) -> bool:
    if a < 0 or b < 0:
        return False
    if a > U64_MAX or b > U64_MAX:
        return False
    total = a + b
    if total > U64_MAX:
        return False
    out[0] = total
    return True


def _encode_name_entry(name: bytes) -> bytes:
    return struct.pack("<Q", len(name)) + name


def parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only(
    buf: bytes | None,
    size: int,
    cursor: int,
    out_name_len: list[int] | None,
    out_required_bytes: list[int] | None,
    out_next_cursor: list[int] | None,
) -> int:
    if buf is None or out_name_len is None or out_required_bytes is None or out_next_cursor is None:
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR

    if out_name_len is out_required_bytes or out_name_len is out_next_cursor or out_required_bytes is out_next_cursor:
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR

    if cursor > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if size > I64_MAX or cursor > I64_MAX:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    snapshot_buf = buf
    snapshot_size = size
    snapshot_cursor = cursor

    staged_name_len = [0]
    staged_required_bytes = [0]
    staged_next_cursor = [0]

    canonical_required_bytes = [0]
    canonical_next_cursor = [0]

    status = parse_name_checked_nopartial_commit_only_preflight_only_parity(
        buf,
        size,
        cursor,
        staged_name_len,
        staged_required_bytes,
        staged_next_cursor,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    if snapshot_buf is not buf or snapshot_size != size or snapshot_cursor != cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if staged_name_len[0] > GGUF_TENSOR_MAX_NAME_BYTES:
        return GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN

    if not _try_add_u64(8, staged_name_len[0], canonical_required_bytes):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if staged_required_bytes[0] != canonical_required_bytes[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if staged_required_bytes[0] > size - cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if not _try_add_u64(cursor, staged_required_bytes[0], canonical_next_cursor):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if staged_next_cursor[0] < cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if canonical_next_cursor[0] > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if canonical_next_cursor[0] != staged_next_cursor[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_name_len[0] = staged_name_len[0]
    out_required_bytes[0] = staged_required_bytes[0]
    out_next_cursor[0] = staged_next_cursor[0]
    return GGUF_TENSOR_PARSE_OK


def explicit_checked_composition(*args) -> int:
    return parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only(*args)


def test_source_contains_iq1021_signature_and_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly("
    assert sig in source
    body = source.rsplit(sig, 1)[1].split("\nI64 ", 1)[0]

    assert "IQ-1028 commit-only diagnostics wrapper for tensor name parity preflight." in source
    assert "status = GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "Bool GGUFTensorByteSpansOverlap(U8 *a_base," in source
    assert "GGUFTensorByteSpansOverlap(out_name_len_ptr, sizeof(U64)," in body
    assert "out_required_bytes_ptr, sizeof(U64))" in body
    assert "if (staged_name_len > GGUF_TENSOR_MAX_NAME_BYTES)" in body
    assert "if (!GGUFTensorTryAddU64(8, staged_name_len, &canonical_required_bytes))" in body
    assert "if (staged_required_bytes != canonical_required_bytes)" in body
    assert "if (staged_next_cursor < cursor)" in body
    assert "if (!GGUFTensorTryAddU64(cursor, staged_required_bytes, &canonical_next_cursor))" in body
    assert "if (canonical_next_cursor != staged_next_cursor)" in body


def test_known_vector_success_and_alias_guard() -> None:
    buf = _encode_name_entry(b"tok_embeddings.weight")

    name_len = [111]
    required = [222]
    next_cursor = [333]

    status = parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        buf,
        len(buf),
        0,
        name_len,
        required,
        next_cursor,
    )
    assert status == GGUF_TENSOR_PARSE_OK
    assert name_len == [len(b"tok_embeddings.weight")]
    assert required == [8 + len(b"tok_embeddings.weight")]
    assert next_cursor == [len(buf)]

    alias = [999]
    status = parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        buf,
        len(buf),
        0,
        alias,
        alias,
        [444],
    )
    assert status == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert alias == [999]


def test_adversarial_length_truncation_cursor_overflow_vectors() -> None:
    out_name = [7]
    out_required = [8]
    out_next = [9]

    tiny = b"\x01\x00\x00"
    status = parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        tiny,
        len(tiny),
        0,
        out_name,
        out_required,
        out_next,
    )
    assert status == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    assert out_name == [7]
    assert out_required == [8]
    assert out_next == [9]

    payload = _encode_name_entry(b"abc")
    status = parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        payload,
        len(payload),
        len(payload) + 1,
        out_name,
        out_required,
        out_next,
    )
    assert status == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    assert out_name == [7]
    assert out_required == [8]
    assert out_next == [9]

    status = parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        payload,
        I64_MAX + 1,
        0,
        out_name,
        out_required,
        out_next,
    )
    assert status == GGUF_TENSOR_PARSE_ERR_TRUNCATED


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260422_1021)

    for _ in range(900):
        name_len = rng.randint(0, 96)
        name = bytes(rng.randint(1, 255) for _ in range(name_len))
        buf = _encode_name_entry(name)

        size = len(buf)
        cursor = 0

        if rng.random() < 0.3 and size > 0:
            size -= rng.randint(1, min(8, size))

        out_a_name = [rng.randint(1, 999)]
        out_a_required = [rng.randint(1, 999)]
        out_a_next = [rng.randint(1, 999)]

        out_b_name = list(out_a_name)
        out_b_required = list(out_a_required)
        out_b_next = list(out_a_next)

        err_a = parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only(
            buf,
            size,
            cursor,
            out_a_name,
            out_a_required,
            out_a_next,
        )
        err_b = explicit_checked_composition(
            buf,
            size,
            cursor,
            out_b_name,
            out_b_required,
            out_b_next,
        )

        assert err_a == err_b
        assert out_a_name == out_b_name
        assert out_a_required == out_b_required
        assert out_a_next == out_b_next


if __name__ == "__main__":
    test_source_contains_iq1021_signature_and_contract()
    test_known_vector_success_and_alias_guard()
    test_adversarial_length_truncation_cursor_overflow_vectors()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
