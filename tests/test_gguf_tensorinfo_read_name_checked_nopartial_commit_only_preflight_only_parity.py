#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-1020)."""

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
)
from test_gguf_tensorinfo_read_name_checked_nopartial_commit_only import (
    U64_MAX,
    parse_name_checked_nopartial_commit_only,
)
from test_gguf_tensorinfo_read_name_checked_nopartial_commit_only_preflight_only import (
    parse_name_checked_nopartial_commit_only_preflight_only,
)


def parse_name_checked_nopartial_commit_only_preflight_only_parity(
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

    staged_pre_name_len = [0]
    staged_pre_required_bytes = [0]
    staged_pre_next_cursor = [0]

    staged_commit_name_len = [0]
    staged_commit_required_bytes = [0]
    staged_commit_next_cursor = [0]

    err = parse_name_checked_nopartial_commit_only_preflight_only(
        buf,
        size,
        cursor,
        staged_pre_name_len,
        staged_pre_required_bytes,
        staged_pre_next_cursor,
    )
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    err = parse_name_checked_nopartial_commit_only(
        buf,
        size,
        cursor,
        staged_commit_name_len,
        staged_commit_required_bytes,
        staged_commit_next_cursor,
    )
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    if snapshot_buf is not buf or snapshot_size != size or snapshot_cursor != cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_pre_name_len[0] != staged_commit_name_len[0]
        or staged_pre_required_bytes[0] != staged_commit_required_bytes[0]
        or staged_pre_next_cursor[0] != staged_commit_next_cursor[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_name_len[0] = staged_commit_name_len[0]
    out_required_bytes[0] = staged_commit_required_bytes[0]
    out_next_cursor[0] = staged_commit_next_cursor[0]
    return GGUF_TENSOR_PARSE_OK


def explicit_checked_composition(
    buf: bytes,
    size: int,
    cursor: int,
    out_name_len: list[int],
    out_required_bytes: list[int],
    out_next_cursor: list[int],
) -> int:
    staged_pre_name_len = [0]
    staged_pre_required_bytes = [0]
    staged_pre_next_cursor = [0]

    staged_commit_name_len = [0]
    staged_commit_required_bytes = [0]
    staged_commit_next_cursor = [0]

    err = parse_name_checked_nopartial_commit_only_preflight_only(
        buf,
        size,
        cursor,
        staged_pre_name_len,
        staged_pre_required_bytes,
        staged_pre_next_cursor,
    )
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    err = parse_name_checked_nopartial_commit_only(
        buf,
        size,
        cursor,
        staged_commit_name_len,
        staged_commit_required_bytes,
        staged_commit_next_cursor,
    )
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    if (
        staged_pre_name_len[0] != staged_commit_name_len[0]
        or staged_pre_required_bytes[0] != staged_commit_required_bytes[0]
        or staged_pre_next_cursor[0] != staged_commit_next_cursor[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_name_len[0] = staged_commit_name_len[0]
    out_required_bytes[0] = staged_commit_required_bytes[0]
    out_next_cursor[0] = staged_commit_next_cursor[0]
    return GGUF_TENSOR_PARSE_OK


def test_source_contains_iq1020_signature_and_tuple_parity_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartial(",
        1,
    )[0]

    assert "status = GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "status = GGUFTensorInfoReadNameCheckedNoPartialCommitOnly(" in body
    assert "if (staged_pre_computed_end != staged_pre_next_cursor ||" in body
    assert "staged_commit_computed_end != staged_commit_next_cursor)" in body
    assert "if (staged_pre_name_len != staged_commit_name_len ||" in body
    assert "staged_pre_required_bytes != staged_pre_required_check ||" in body
    assert "staged_commit_required_bytes != staged_commit_required_check)" in body
    assert "out_name_len_ptr = (U8 *)out_name_len;" in body
    assert "*out_name_len = staged_commit_name_len;" in body
    assert "*out_required_bytes = staged_commit_required_bytes;" in body
    assert "*out_next_cursor = staged_commit_next_cursor;" in body


def test_null_alias_and_no_partial_publish_on_failure() -> None:
    payload = gguf_name_entry(b"tok_embd")

    out_name_len = [101]
    out_required_bytes = [102]
    out_next_cursor = [103]

    err = parse_name_checked_nopartial_commit_only_preflight_only_parity(
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

    err = parse_name_checked_nopartial_commit_only_preflight_only_parity(
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
    err = parse_name_checked_nopartial_commit_only_preflight_only_parity(
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
    err = parse_name_checked_nopartial_commit_only_preflight_only_parity(
        too_long,
        len(too_long),
        0,
        out_name_len,
        out_required_bytes,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN

    bogus_len = (128).to_bytes(8, "little") + b"abc"
    err = parse_name_checked_nopartial_commit_only_preflight_only_parity(
        bogus_len,
        len(bogus_len),
        0,
        out_name_len,
        out_required_bytes,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED

    payload = gguf_name_entry(b"xyz")
    err = parse_name_checked_nopartial_commit_only_preflight_only_parity(
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


def test_success_and_randomized_tuple_parity_against_explicit_composition() -> None:
    fixed_name = b"model.layers.0.attn_q.weight"
    payload = gguf_name_entry(fixed_name)

    out_name_len = [0]
    out_required_bytes = [0]
    out_next_cursor = [0]
    err = parse_name_checked_nopartial_commit_only_preflight_only_parity(
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

    rng = random.Random(202604221020)
    for i in range(1400):
        name_len = rng.randint(0, 256)
        name_raw = bytes(rng.randint(0, 255) for _ in range(name_len))

        prefix_len = rng.randint(0, 20)
        prefix = bytes(rng.randint(0, 255) for _ in range(prefix_len))
        payload = prefix + gguf_name_entry(name_raw)

        actual_name_len = [0x10 + i]
        actual_required_bytes = [0x20 + i]
        actual_next_cursor = [0x30 + i]

        expected_name_len = [0]
        expected_required_bytes = [0]
        expected_next_cursor = [0]

        err_actual = parse_name_checked_nopartial_commit_only_preflight_only_parity(
            payload,
            len(payload),
            prefix_len,
            actual_name_len,
            actual_required_bytes,
            actual_next_cursor,
        )
        err_expected = explicit_checked_composition(
            payload,
            len(payload),
            prefix_len,
            expected_name_len,
            expected_required_bytes,
            expected_next_cursor,
        )

        assert err_actual == GGUF_TENSOR_PARSE_OK
        assert err_expected == GGUF_TENSOR_PARSE_OK
        assert actual_name_len == expected_name_len == [name_len]
        assert actual_required_bytes == expected_required_bytes == [8 + name_len]
        assert actual_next_cursor == expected_next_cursor == [len(payload)]


if __name__ == "__main__":
    test_source_contains_iq1020_signature_and_tuple_parity_contract()
    test_null_alias_and_no_partial_publish_on_failure()
    test_adversarial_length_truncation_and_cursor_overflow_vectors()
    test_success_and_randomized_tuple_parity_against_explicit_composition()
    print("gguf_tensorinfo_read_name_checked_nopartial_commit_only_preflight_only_parity=ok")
