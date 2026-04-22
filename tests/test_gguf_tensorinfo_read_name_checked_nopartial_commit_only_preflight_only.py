#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnly (IQ-1011)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_gguf_tensorinfo_read_name_checked_nopartial import (
    GGUF_TENSOR_MAX_NAME_BYTES,
    GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN,
    GGUF_TENSOR_PARSE_ERR_NULL_PTR,
    GGUF_TENSOR_PARSE_ERR_TRUNCATED,
    GGUF_TENSOR_PARSE_OK,
    I64_MAX,
    gguf_name_entry,
    parse_name_checked_nopartial,
)
from test_gguf_tensorinfo_read_name_checked_nopartial_commit_only import (
    U64_MAX,
    parse_name_checked_nopartial_commit_only,
)


def u64_add(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None
    if a > U64_MAX - b:
        return None
    return a + b


def parse_name_checked_nopartial_commit_only_preflight_only(
    buf: bytes | None,
    size: int,
    cursor: int,
    out_name_len: list[int] | None,
    out_required_bytes: list[int] | None,
    out_next_cursor: list[int] | None,
) -> int:
    if (
        buf is None
        or out_name_len is None
        or out_required_bytes is None
        or out_next_cursor is None
    ):
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

    status = parse_name_checked_nopartial_commit_only(
        buf,
        size,
        cursor,
        staged_name_len,
        staged_required_bytes,
        staged_next_cursor,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    canonical_name = {}
    canonical_next_cursor = [0]
    status = parse_name_checked_nopartial(
        buf,
        size,
        cursor,
        canonical_name,
        canonical_next_cursor,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    canonical_name_len = canonical_name["len"]
    canonical_required_bytes = u64_add(8, canonical_name_len)
    if canonical_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    computed_end = u64_add(cursor, canonical_required_bytes)
    if computed_end is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if computed_end != canonical_next_cursor[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if snapshot_buf is not buf or snapshot_size != size or snapshot_cursor != cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_name_len[0] != canonical_name_len
        or staged_required_bytes[0] != canonical_required_bytes
        or staged_next_cursor[0] != canonical_next_cursor[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_name_len[0] = staged_name_len[0]
    out_required_bytes[0] = staged_required_bytes[0]
    out_next_cursor[0] = staged_next_cursor[0]
    return GGUF_TENSOR_PARSE_OK


def test_source_contains_iq1011_signature_and_no_partial_publish_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartial(",
        1,
    )[0]

    assert "status = GGUFTensorInfoReadNameCheckedNoPartialCommitOnly(buf," in body
    assert "status = GGUFTensorInfoReadNameCheckedNoPartial(buf," in body
    assert "out_name_len_ptr = (U8 *)out_name_len;" in body
    assert "out_required_bytes_ptr = (U8 *)out_required_bytes;" in body
    assert "if (staged_name_len != canonical_name_len ||" in body
    assert "*out_name_len = staged_name_len;" in body
    assert "*out_required_bytes = staged_required_bytes;" in body
    assert "*out_next_cursor = staged_next_cursor;" in body


def test_null_alias_and_no_partial_publish_on_failure() -> None:
    payload = gguf_name_entry(b"tok_embd")

    out_name_len = [101]
    out_required_bytes = [102]
    out_next_cursor = [103]

    err = parse_name_checked_nopartial_commit_only_preflight_only(
        None,
        len(payload),
        0,
        out_name_len,
        out_required_bytes,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert out_name_len == [101]
    assert out_required_bytes == [102]
    assert out_next_cursor == [103]

    err = parse_name_checked_nopartial_commit_only_preflight_only(
        payload,
        len(payload),
        0,
        out_name_len,
        out_name_len,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert out_name_len == [101]
    assert out_required_bytes == [102]
    assert out_next_cursor == [103]

    bad = gguf_name_entry(b"abc")[:-1]
    err = parse_name_checked_nopartial_commit_only_preflight_only(
        bad,
        len(bad),
        0,
        out_name_len,
        out_required_bytes,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    assert out_name_len == [101]
    assert out_required_bytes == [102]
    assert out_next_cursor == [103]


def test_adversarial_length_truncation_and_cursor_overflow_vectors() -> None:
    out_name_len = [201]
    out_required_bytes = [202]
    out_next_cursor = [203]

    too_long = (GGUF_TENSOR_MAX_NAME_BYTES + 1).to_bytes(8, "little")
    err = parse_name_checked_nopartial_commit_only_preflight_only(
        too_long,
        len(too_long),
        0,
        out_name_len,
        out_required_bytes,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN

    bogus_len = (128).to_bytes(8, "little") + b"abc"
    err = parse_name_checked_nopartial_commit_only_preflight_only(
        bogus_len,
        len(bogus_len),
        0,
        out_name_len,
        out_required_bytes,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED

    payload = gguf_name_entry(b"xyz")
    err = parse_name_checked_nopartial_commit_only_preflight_only(
        payload,
        U64_MAX,
        U64_MAX - 1,
        out_name_len,
        out_required_bytes,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    assert out_name_len == [201]
    assert out_required_bytes == [202]
    assert out_next_cursor == [203]


def test_success_and_randomized_tuple_parity() -> None:
    fixed_name = b"model.layers.0.attn_q.weight"
    payload = gguf_name_entry(fixed_name)

    out_name_len = [0]
    out_required_bytes = [0]
    out_next_cursor = [0]
    err = parse_name_checked_nopartial_commit_only_preflight_only(
        payload,
        len(payload),
        0,
        out_name_len,
        out_required_bytes,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_OK
    assert out_name_len == [len(fixed_name)]
    assert out_required_bytes == [8 + len(fixed_name)]
    assert out_next_cursor == [len(payload)]

    rng = random.Random(202604221011)
    for i in range(1200):
        name_len = rng.randint(0, 192)
        name_raw = bytes(rng.randint(0, 255) for _ in range(name_len))

        prefix_len = rng.randint(0, 20)
        prefix = bytes(rng.randint(0, 255) for _ in range(prefix_len))
        payload = prefix + gguf_name_entry(name_raw)

        out_name_len = [0x10 + i]
        out_required_bytes = [0x20 + i]
        out_next_cursor = [0x30 + i]

        err = parse_name_checked_nopartial_commit_only_preflight_only(
            payload,
            len(payload),
            prefix_len,
            out_name_len,
            out_required_bytes,
            out_next_cursor,
        )

        assert err == GGUF_TENSOR_PARSE_OK
        assert out_name_len == [name_len]
        assert out_required_bytes == [8 + name_len]
        assert out_next_cursor == [len(payload)]


if __name__ == "__main__":
    test_source_contains_iq1011_signature_and_no_partial_publish_contract()
    test_null_alias_and_no_partial_publish_on_failure()
    test_adversarial_length_truncation_and_cursor_overflow_vectors()
    test_success_and_randomized_tuple_parity()
    print("gguf_tensorinfo_read_name_checked_nopartial_commit_only_preflight_only=ok")
