#!/usr/bin/env python3
"""Harness for IQ-1029 tensor-name diagnostics no-write companion."""

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
from test_gguf_tensorinfo_read_name_checked_nopartial import gguf_name_entry  # noqa: E402
from test_gguf_tensorinfo_read_name_checked_nopartial_commit_only_preflight_only import (  # noqa: E402
    parse_name_checked_nopartial_commit_only_preflight_only,
)
from test_gguf_tensorinfo_read_name_checked_nopartial_commit_only_preflight_only_parity_commit_only import (  # noqa: E402
    parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only,
)


def try_add_u64(a: int, b: int, out: list[int]) -> bool:
    if a < 0 or b < 0:
        return False
    if a > 0xFFFFFFFFFFFFFFFF or b > 0xFFFFFFFFFFFFFFFF:
        return False
    total = a + b
    if total > 0xFFFFFFFFFFFFFFFF:
        return False
    out[0] = total
    return True


def parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
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

    staged_commit_name_len = [0]
    staged_commit_required_bytes = [0]
    staged_commit_next_cursor = [0]
    staged_commit_required_check = [0]
    staged_commit_computed_end = [0]

    canonical_name_len = [0]
    canonical_required_bytes = [0]
    canonical_next_cursor = [0]
    canonical_required_check = [0]
    canonical_computed_end = [0]

    raw_name_len = [0]
    raw_required_bytes = [0]
    raw_computed_end = [0]

    status = parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        buf,
        size,
        cursor,
        staged_commit_name_len,
        staged_commit_required_bytes,
        staged_commit_next_cursor,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    status = parse_name_checked_nopartial_commit_only_preflight_only(
        buf,
        size,
        cursor,
        canonical_name_len,
        canonical_required_bytes,
        canonical_next_cursor,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    if not try_add_u64(8, staged_commit_name_len[0], staged_commit_required_check):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if not try_add_u64(8, canonical_name_len[0], canonical_required_check):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if size < 0 or cursor < 0 or cursor > size or (size - cursor) < 8:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    raw_name_len[0] = int.from_bytes(buf[cursor : cursor + 8], "little", signed=False)
    if raw_name_len[0] > (1 << 20):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if not try_add_u64(8, raw_name_len[0], raw_required_bytes):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if raw_required_bytes[0] > (size - cursor):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if not try_add_u64(cursor, raw_required_bytes[0], raw_computed_end):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if raw_computed_end[0] > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if not try_add_u64(cursor, staged_commit_required_bytes[0], staged_commit_computed_end):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if not try_add_u64(cursor, canonical_required_bytes[0], canonical_computed_end):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if staged_commit_computed_end[0] > size or canonical_computed_end[0] > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if staged_commit_computed_end[0] != staged_commit_next_cursor[0] or canonical_computed_end[0] != canonical_next_cursor[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if snapshot_buf is not buf or snapshot_size != size or snapshot_cursor != cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_commit_name_len[0] != canonical_name_len[0]
        or staged_commit_name_len[0] != raw_name_len[0]
        or staged_commit_required_bytes[0] != canonical_required_bytes[0]
        or staged_commit_required_bytes[0] != raw_required_bytes[0]
        or staged_commit_next_cursor[0] != canonical_next_cursor[0]
        or staged_commit_next_cursor[0] != raw_computed_end[0]
        or staged_commit_required_bytes[0] != staged_commit_required_check[0]
        or canonical_required_bytes[0] != canonical_required_check[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_name_len[0] = staged_commit_name_len[0]
    out_required_bytes[0] = staged_commit_required_bytes[0]
    out_next_cursor[0] = staged_commit_next_cursor[0]
    return GGUF_TENSOR_PARSE_OK


def explicit_checked_composition(*args) -> int:
    return parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(*args)


def test_source_contains_iq1029_signature_and_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
    assert sig in source
    body = source.rsplit(sig, 1)[1].split("\nI64 ", 1)[0]

    assert "IQ-1029 diagnostics-only no-write companion for tensor name parity commit." in source
    assert "GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "GGUFTensorInfoReadNameCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "if (!GGUFTensorTryAddU64(8, staged_commit_name_len, &staged_commit_required_check))" in body
    assert "if (!GGUFTensorTryAddU64(8, canonical_name_len, &canonical_required_check))" in body
    assert "raw_name_len = (U64)buf[cursor]" in body
    assert "staged_commit_name_len != raw_name_len" in body
    assert "staged_commit_required_bytes != raw_required_bytes" in body
    assert "staged_commit_next_cursor != raw_computed_end" in body
    assert "if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)" in body
    assert "staged_commit_name_len != canonical_name_len" in body
    assert "staged_commit_required_bytes != canonical_required_bytes" in body
    assert "staged_commit_next_cursor != canonical_next_cursor" in body
    assert "*out_name_len = staged_commit_name_len;" in body
    assert "*out_required_bytes = staged_commit_required_bytes;" in body
    assert "*out_next_cursor = staged_commit_next_cursor;" in body


def test_known_vector_success_and_alias_guard() -> None:
    payload = gguf_name_entry(b"tok_embd.weight")

    name_len = [111]
    required = [222]
    next_cursor = [333]

    status = parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
        payload,
        len(payload),
        0,
        name_len,
        required,
        next_cursor,
    )
    assert status == GGUF_TENSOR_PARSE_OK
    assert name_len == [len(b"tok_embd.weight")]
    assert required == [8 + len(b"tok_embd.weight")]
    assert next_cursor == [len(payload)]

    alias = [999]
    status = parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
        payload,
        len(payload),
        0,
        alias,
        alias,
        [444],
    )
    assert status == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert alias == [999]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260422_1029)

    for _ in range(900):
        name_len = rng.randint(0, 120)
        name = bytes(rng.randint(1, 255) for _ in range(name_len))
        prefix_len = rng.randint(0, 16)
        prefix = bytes(rng.randint(0, 255) for _ in range(prefix_len))
        payload = prefix + gguf_name_entry(name)

        size = len(payload)
        if rng.random() < 0.28 and size > prefix_len:
            size -= rng.randint(1, min(8, size - prefix_len))

        out_a_name = [rng.randint(1, 999)]
        out_a_required = [rng.randint(1, 999)]
        out_a_next = [rng.randint(1, 999)]

        out_b_name = list(out_a_name)
        out_b_required = list(out_a_required)
        out_b_next = list(out_a_next)

        err_a = parse_name_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
            payload,
            size,
            prefix_len,
            out_a_name,
            out_a_required,
            out_a_next,
        )
        err_b = explicit_checked_composition(
            payload,
            size,
            prefix_len,
            out_b_name,
            out_b_required,
            out_b_next,
        )

        assert err_a == err_b
        assert out_a_name == out_b_name
        assert out_a_required == out_b_required
        assert out_a_next == out_b_next


if __name__ == "__main__":
    test_source_contains_iq1029_signature_and_contract()
    test_known_vector_success_and_alias_guard()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
