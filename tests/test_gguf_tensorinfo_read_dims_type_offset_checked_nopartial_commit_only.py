#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnly (IQ-1002)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial import (
    GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW,
    GGUF_TENSOR_PARSE_ERR_NULL_PTR,
    GGUF_TENSOR_PARSE_ERR_TRUNCATED,
    GGUF_TENSOR_PARSE_OK,
    U64_MAX,
    dims_type_offset_entry,
    parse_dims_type_offset_checked_nopartial,
)


def u64_add(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None
    if a > U64_MAX - b:
        return None
    return a + b


def u64_mul(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None
    if a != 0 and b > U64_MAX // a:
        return None
    return a * b


def parse_dims_type_offset_checked_nopartial_commit_only(
    buf: bytes | None,
    size: int,
    cursor: int,
    out_dim_count: list[int] | None,
    out_dims_cells: list[int] | None,
    out_required_bytes: list[int] | None,
    out_type_value: list[int] | None,
    out_tensor_offset: list[int] | None,
    out_last_dim_index: list[int] | None,
    out_next_cursor: list[int] | None,
) -> int:
    if (
        buf is None
        or out_dim_count is None
        or out_dims_cells is None
        or out_required_bytes is None
        or out_type_value is None
        or out_tensor_offset is None
        or out_last_dim_index is None
        or out_next_cursor is None
    ):
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR
    if cursor > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    snapshot_buf = buf
    snapshot_size = size
    snapshot_cursor = cursor

    parsed: dict = {}
    parsed_next = [0]
    status = parse_dims_type_offset_checked_nopartial(
        buf,
        size,
        cursor,
        parsed,
        parsed_next,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    staged_dim_count = parsed["n_dims"]
    staged_type_value = parsed["ggml_type"]
    staged_tensor_offset = parsed["offset"]
    staged_next_cursor = parsed_next[0]

    staged_dims_cells = staged_dim_count
    staged_last_dim_index = staged_dim_count - 1

    staged_dims_bytes = u64_mul(staged_dims_cells, 8)
    if staged_dims_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW

    staged_required_bytes = u64_add(4, staged_dims_bytes)
    if staged_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW
    staged_required_bytes = u64_add(staged_required_bytes, 4)
    if staged_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    staged_required_bytes = u64_add(staged_required_bytes, 8)
    if staged_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    computed_end = u64_add(cursor, staged_required_bytes)
    if computed_end is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if computed_end != staged_next_cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if snapshot_buf is not buf or snapshot_size != size or snapshot_cursor != cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_dim_count[0] = staged_dim_count
    out_dims_cells[0] = staged_dims_cells
    out_required_bytes[0] = staged_required_bytes
    out_type_value[0] = staged_type_value
    out_tensor_offset[0] = staged_tensor_offset
    out_last_dim_index[0] = staged_last_dim_index
    out_next_cursor[0] = staged_next_cursor
    return GGUF_TENSOR_PARSE_OK


def test_source_contains_iq1002_commit_only_signature_and_publish_tuple() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFTensorParseOne(", 1)[0]

    assert "status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartial(" in body
    assert "staged_dims_cells = staged_dim_count;" in body
    assert "staged_last_dim_index = staged_dim_count - 1;" in body
    assert "GGUFTensorTryMulU64(staged_dims_cells, 8, &staged_dims_bytes)" in body
    assert "if (!GGUFTensorTryAddU64(cursor, staged_required_bytes, &computed_end))" in body
    assert "*out_dim_count = staged_dim_count;" in body
    assert "*out_dims_cells = staged_dims_cells;" in body
    assert "*out_required_bytes = staged_required_bytes;" in body
    assert "*out_type_value = staged_type_value;" in body
    assert "*out_tensor_offset = staged_tensor_offset;" in body
    assert "*out_last_dim_index = staged_last_dim_index;" in body
    assert "*out_next_cursor = staged_next_cursor;" in body


def test_null_and_no_partial_publish_on_failure() -> None:
    payload = dims_type_offset_entry([8, 8], 2, 16)

    out_dim_count = [101]
    out_dims_cells = [102]
    out_required_bytes = [103]
    out_type_value = [104]
    out_tensor_offset = [105]
    out_last_dim_index = [106]
    out_next_cursor = [107]

    err = parse_dims_type_offset_checked_nopartial_commit_only(
        None,
        len(payload),
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert out_dim_count == [101]
    assert out_dims_cells == [102]
    assert out_required_bytes == [103]
    assert out_type_value == [104]
    assert out_tensor_offset == [105]
    assert out_last_dim_index == [106]
    assert out_next_cursor == [107]

    truncated = payload[:-2]
    err = parse_dims_type_offset_checked_nopartial_commit_only(
        truncated,
        len(truncated),
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    assert out_dim_count == [101]
    assert out_dims_cells == [102]
    assert out_required_bytes == [103]
    assert out_type_value == [104]
    assert out_tensor_offset == [105]
    assert out_last_dim_index == [106]
    assert out_next_cursor == [107]


def test_adversarial_dim_type_offset_and_span_vectors() -> None:
    out_dim_count = [201]
    out_dims_cells = [202]
    out_required_bytes = [203]
    out_type_value = [204]
    out_tensor_offset = [205]
    out_last_dim_index = [206]
    out_next_cursor = [207]

    bad_type = dims_type_offset_entry([4, 4], 999, 0)
    err = parse_dims_type_offset_checked_nopartial_commit_only(
        bad_type,
        len(bad_type),
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
        out_next_cursor,
    )
    assert err != GGUF_TENSOR_PARSE_OK
    assert out_dim_count == [201]
    assert out_dims_cells == [202]

    dim_overflow = dims_type_offset_entry([1 << 63, 3], 2, 0)
    err = parse_dims_type_offset_checked_nopartial_commit_only(
        dim_overflow,
        len(dim_overflow),
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW

    trunc_offset = dims_type_offset_entry([2, 2, 2], 8, 42)[:-7]
    err = parse_dims_type_offset_checked_nopartial_commit_only(
        trunc_offset,
        len(trunc_offset),
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED

    # Span-overflow vector: computed cursor end overflows U64.
    tail = dims_type_offset_entry([1], 2, 0)
    err = parse_dims_type_offset_checked_nopartial_commit_only(
        tail,
        U64_MAX,
        U64_MAX - 1,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED


def test_success_and_randomized_tuple_parity() -> None:
    fixed = dims_type_offset_entry([64, 32, 16], 8, 4096)
    out_dim_count = [0]
    out_dims_cells = [0]
    out_required_bytes = [0]
    out_type_value = [0]
    out_tensor_offset = [0]
    out_last_dim_index = [0]
    out_next_cursor = [0]

    err = parse_dims_type_offset_checked_nopartial_commit_only(
        fixed,
        len(fixed),
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_OK
    assert out_dim_count == [3]
    assert out_dims_cells == [3]
    assert out_required_bytes == [4 + 3 * 8 + 4 + 8]
    assert out_type_value == [8]
    assert out_tensor_offset == [4096]
    assert out_last_dim_index == [2]
    assert out_next_cursor == [len(fixed)]

    rng = random.Random(202604221002)
    for i in range(1800):
        n_dims = rng.randint(1, 8)
        dims: list[int] = []
        prod = 1
        for _ in range(n_dims):
            max_dim = max(1, min(1 << 14, U64_MAX // prod))
            dim = rng.randint(1, max_dim)
            dims.append(dim)
            prod *= dim

        ggml_type = rng.choice([2, 8, 12, 14, 30, 35])
        offset = rng.randint(0, 1 << 45)

        prefix_len = rng.randint(0, 31)
        prefix = bytes(rng.randint(0, 255) for _ in range(prefix_len))
        payload = prefix + dims_type_offset_entry(dims, ggml_type, offset)

        out_dim_count_a = [0x11]
        out_dims_cells_a = [0x12]
        out_required_bytes_a = [0x13]
        out_type_value_a = [0x14]
        out_tensor_offset_a = [0x15]
        out_last_dim_index_a = [0x16]
        out_next_cursor_a = [0x17]

        err = parse_dims_type_offset_checked_nopartial_commit_only(
            payload,
            len(payload),
            prefix_len,
            out_dim_count_a,
            out_dims_cells_a,
            out_required_bytes_a,
            out_type_value_a,
            out_tensor_offset_a,
            out_last_dim_index_a,
            out_next_cursor_a,
        )
        assert err == GGUF_TENSOR_PARSE_OK
        assert out_dim_count_a == [n_dims]
        assert out_dims_cells_a == [n_dims]
        assert out_required_bytes_a == [4 + n_dims * 8 + 4 + 8]
        assert out_type_value_a == [ggml_type]
        assert out_tensor_offset_a == [offset]
        assert out_last_dim_index_a == [n_dims - 1]
        assert out_next_cursor_a == [len(payload)]


if __name__ == "__main__":
    test_source_contains_iq1002_commit_only_signature_and_publish_tuple()
    test_null_and_no_partial_publish_on_failure()
    test_adversarial_dim_type_offset_and_span_vectors()
    test_success_and_randomized_tuple_parity()
    print("gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only=ok")
