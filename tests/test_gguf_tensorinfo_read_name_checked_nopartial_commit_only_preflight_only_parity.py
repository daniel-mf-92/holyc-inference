#!/usr/bin/env python3
"""Parity harness for IQ-1020 tensor-name diagnostics wrapper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_gguf_tensor_info_read_checked_nopartial import (  # noqa: E402
    GGUF_TENSOR_PARSE_ERR_NULL_PTR,
    GGUF_TENSOR_PARSE_ERR_TRUNCATED,
    GGUF_TENSOR_PARSE_OK,
)
from test_gguf_tensorinfo_read_name_checked_nopartial_commit_only import (  # noqa: E402
    parse_name_checked_nopartial_commit_only,
)
from test_gguf_tensorinfo_read_name_checked_nopartial_commit_only_preflight_only import (  # noqa: E402
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
    if buf is None or out_name_len is None or out_required_bytes is None or out_next_cursor is None:
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR

    if out_name_len is out_required_bytes or out_name_len is out_next_cursor or out_required_bytes is out_next_cursor:
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR

    if cursor > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if size > 0x7FFFFFFFFFFFFFFF or cursor > 0x7FFFFFFFFFFFFFFF:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    snapshot_buf = buf
    snapshot_size = size
    snapshot_cursor = cursor

    staged_pre_name_len = [0]
    staged_pre_required_bytes = [0]
    staged_pre_next_cursor = [0]
    staged_pre_computed_end = [0]
    staged_pre_required_check = [0]

    staged_commit_name_len = [0]
    staged_commit_required_bytes = [0]
    staged_commit_next_cursor = [0]
    staged_commit_computed_end = [0]
    staged_commit_required_check = [0]

    status = parse_name_checked_nopartial_commit_only_preflight_only(
        buf,
        size,
        cursor,
        staged_pre_name_len,
        staged_pre_required_bytes,
        staged_pre_next_cursor,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    status = parse_name_checked_nopartial_commit_only(
        buf,
        size,
        cursor,
        staged_commit_name_len,
        staged_commit_required_bytes,
        staged_commit_next_cursor,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    if not _try_add_u64(8, staged_pre_name_len[0], staged_pre_required_check):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if not _try_add_u64(8, staged_commit_name_len[0], staged_commit_required_check):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if not _try_add_u64(cursor, staged_pre_required_bytes[0], staged_pre_computed_end):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if not _try_add_u64(cursor, staged_commit_required_bytes[0], staged_commit_computed_end):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if staged_pre_computed_end[0] > size or staged_commit_computed_end[0] > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if staged_pre_computed_end[0] != staged_pre_next_cursor[0] or staged_commit_computed_end[0] != staged_commit_next_cursor[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if snapshot_buf is not buf or snapshot_size != size or snapshot_cursor != cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_pre_name_len[0] != staged_commit_name_len[0]
        or staged_pre_required_bytes[0] != staged_commit_required_bytes[0]
        or staged_pre_next_cursor[0] != staged_commit_next_cursor[0]
        or staged_pre_required_bytes[0] != staged_pre_required_check[0]
        or staged_commit_required_bytes[0] != staged_commit_required_check[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_name_len[0] = staged_commit_name_len[0]
    out_required_bytes[0] = staged_commit_required_bytes[0]
    out_next_cursor[0] = staged_commit_next_cursor[0]
    return GGUF_TENSOR_PARSE_OK


def _try_add_u64(a: int, b: int, out: list[int]) -> bool:
    if a < 0 or b < 0:
        return False
    if a > 0xFFFFFFFFFFFFFFFF or b > 0xFFFFFFFFFFFFFFFF:
        return False
    total = a + b
    if total > 0xFFFFFFFFFFFFFFFF:
        return False
    out[0] = total
    return True


def explicit_checked_composition(*args) -> int:
    return parse_name_checked_nopartial_commit_only_preflight_only_parity(*args)


def _encode_name_entry(name: bytes) -> bytes:
    return len(name).to_bytes(8, "little") + name


def test_source_contains_iq1020_signature_and_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.rsplit(sig, 1)[1].split("\nI64 ", 1)[0]

    assert "IQ-1020 diagnostics-only parity gate for tensor name tuple decoding." in source
    assert "GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "GGUFTensorInfoReadNameCheckedNoPartialCommitOnly(" in body
    assert "if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)" in body
    assert "staged_pre_name_len != staged_commit_name_len" in body
    assert "staged_pre_required_bytes != staged_commit_required_bytes" in body
    assert "staged_pre_next_cursor != staged_commit_next_cursor" in body
    assert "staged_pre_required_bytes != staged_pre_required_check" in body
    assert "staged_commit_required_bytes != staged_commit_required_check" in body


def test_known_vector_success_and_alias_guard() -> None:
    buf = _encode_name_entry(b"tok_embd.weight")

    name_len = [111]
    required = [222]
    next_cursor = [333]

    status = parse_name_checked_nopartial_commit_only_preflight_only_parity(
        buf,
        len(buf),
        0,
        name_len,
        required,
        next_cursor,
    )
    assert status == GGUF_TENSOR_PARSE_OK
    assert name_len == [len(b"tok_embd.weight")]
    assert required == [8 + len(b"tok_embd.weight")]
    assert next_cursor == [len(buf)]

    alias = [999]
    status = parse_name_checked_nopartial_commit_only_preflight_only_parity(
        buf,
        len(buf),
        0,
        alias,
        alias,
        [444],
    )
    assert status == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert alias == [999]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260422_1020)

    for _ in range(700):
        name_len = rng.randint(0, 96)
        name = bytes(rng.randint(1, 255) for _ in range(name_len))
        buf = _encode_name_entry(name)

        size = len(buf)
        cursor = 0

        if rng.random() < 0.25 and size > 0:
            size -= rng.randint(1, min(8, size))

        out_a_name = [rng.randint(1, 999)]
        out_a_required = [rng.randint(1, 999)]
        out_a_next = [rng.randint(1, 999)]

        out_b_name = list(out_a_name)
        out_b_required = list(out_a_required)
        out_b_next = list(out_a_next)

        err_a = parse_name_checked_nopartial_commit_only_preflight_only_parity(
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
    test_source_contains_iq1020_signature_and_contract()
    test_known_vector_success_and_alias_guard()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
