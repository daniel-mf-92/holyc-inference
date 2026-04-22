#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoReadCheckedNoPartialCommitOnly (IQ-1009)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_gguf_tensor_info_read_checked_nopartial import (
    GGUF_TENSOR_PARSE_ERR_BAD_DIMS,
    GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW,
    GGUF_TENSOR_PARSE_ERR_NULL_PTR,
    GGUF_TENSOR_PARSE_ERR_TRUNCATED,
    GGUF_TENSOR_PARSE_OK,
    KNOWN_TYPES,
    U64_MAX,
    gguf_string_bytes,
    parse_one_checked_nopartial,
    tensor_entry,
    u32,
    u64,
)

I64_MAX = (1 << 63) - 1


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


def parse_one_checked_nopartial_commit_only(
    buf: bytes | None,
    size: int,
    cursor: int,
    out_name_len: list[int] | None,
    out_dim_count: list[int] | None,
    out_dims_cells: list[int] | None,
    out_required_bytes: list[int] | None,
    out_type_value: list[int] | None,
    out_tensor_offset: list[int] | None,
    out_next_cursor: list[int] | None,
) -> int:
    if (
        buf is None
        or out_name_len is None
        or out_dim_count is None
        or out_dims_cells is None
        or out_required_bytes is None
        or out_type_value is None
        or out_tensor_offset is None
        or out_next_cursor is None
    ):
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR
    if cursor > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if size > I64_MAX or cursor > I64_MAX:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    snapshot_buf = buf
    snapshot_size = size
    snapshot_cursor = cursor

    staged: dict = {}
    staged_next_cursor = [0]
    status = parse_one_checked_nopartial(buf, size, cursor, staged, staged_next_cursor)
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    staged_name_len = staged["name_len"]
    staged_dim_count = staged["n_dims"]
    staged_dims_cells = staged_dim_count
    staged_type_value = staged["ggml_type"]
    staged_tensor_offset = staged["offset"]

    staged_required_bytes = u64_add(8, staged_name_len)
    if staged_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    staged_required_bytes = u64_add(staged_required_bytes, 4)
    if staged_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    dims_bytes = u64_mul(staged_dims_cells, 8)
    if dims_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW

    staged_required_bytes = u64_add(staged_required_bytes, dims_bytes)
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
    if computed_end != staged_next_cursor[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if snapshot_buf is not buf or snapshot_size != size or snapshot_cursor != cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_name_len[0] = staged_name_len
    out_dim_count[0] = staged_dim_count
    out_dims_cells[0] = staged_dims_cells
    out_required_bytes[0] = staged_required_bytes
    out_type_value[0] = staged_type_value
    out_tensor_offset[0] = staged_tensor_offset
    out_next_cursor[0] = staged_next_cursor[0]
    return GGUF_TENSOR_PARSE_OK


def parse_one_checked_nopartial_commit_only_explicit_composition(
    buf: bytes | None,
    size: int,
    cursor: int,
) -> tuple[int, dict[str, int]]:
    if buf is None:
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR, {}

    parsed: dict = {}
    next_cursor = [0]
    status = parse_one_checked_nopartial(buf, size, cursor, parsed, next_cursor)
    if status != GGUF_TENSOR_PARSE_OK:
        return status, {}

    name_required = u64_add(8, parsed["name_len"])
    if name_required is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED, {}

    dims_cells = parsed["n_dims"]
    dims_bytes = u64_mul(dims_cells, 8)
    if dims_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW, {}

    dims_required = u64_add(4, dims_bytes)
    if dims_required is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW, {}
    dims_required = u64_add(dims_required, 4)
    if dims_required is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED, {}
    dims_required = u64_add(dims_required, 8)
    if dims_required is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED, {}

    required = u64_add(name_required, dims_required)
    if required is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW, {}

    end_cursor = u64_add(cursor, required)
    if end_cursor is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED, {}
    if end_cursor != next_cursor[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED, {}

    return (
        GGUF_TENSOR_PARSE_OK,
        {
            "name_len": parsed["name_len"],
            "dim_count": parsed["n_dims"],
            "dims_cells": dims_cells,
            "required_bytes": required,
            "type_value": parsed["ggml_type"],
            "tensor_offset": parsed["offset"],
            "next_cursor": next_cursor[0],
        },
    )


def test_source_contains_iq1009_signature_and_publish_tuple() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadCheckedNoPartialCommitOnly(U8 *buf,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFParseTensorInfo(", 1)[0]

    assert "status = GGUFTensorInfoReadCheckedNoPartial(buf," in body
    assert "if (!GGUFTensorTryAddU64(8, staged_name_len, &staged_required_bytes))" in body
    assert "if (!GGUFTensorTryMulU64(staged_dims_cells, 8, &dims_bytes))" in body
    assert "if (!GGUFTensorTryAddU64(cursor, staged_required_bytes, &computed_end))" in body
    assert "*out_name_len = staged_name_len;" in body
    assert "*out_dim_count = staged_dim_count;" in body
    assert "*out_dims_cells = staged_dims_cells;" in body
    assert "*out_required_bytes = staged_required_bytes;" in body
    assert "*out_type_value = staged_type_value;" in body
    assert "*out_tensor_offset = staged_tensor_offset;" in body
    assert "*out_next_cursor = staged_next_cursor;" in body


def test_null_cursor_and_no_partial_publish_on_failure() -> None:
    payload = tensor_entry("w", [8, 8], 2, 16)

    out_name_len = [101]
    out_dim_count = [102]
    out_dims_cells = [103]
    out_required_bytes = [104]
    out_type_value = [105]
    out_tensor_offset = [106]
    out_next_cursor = [107]

    err = parse_one_checked_nopartial_commit_only(
        None,
        len(payload),
        0,
        out_name_len,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert out_name_len == [101]
    assert out_dim_count == [102]
    assert out_dims_cells == [103]
    assert out_required_bytes == [104]
    assert out_type_value == [105]
    assert out_tensor_offset == [106]
    assert out_next_cursor == [107]

    err = parse_one_checked_nopartial_commit_only(
        payload,
        len(payload),
        len(payload) + 1,
        out_name_len,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    assert out_name_len == [101]
    assert out_dim_count == [102]
    assert out_dims_cells == [103]
    assert out_required_bytes == [104]
    assert out_type_value == [105]
    assert out_tensor_offset == [106]
    assert out_next_cursor == [107]


def test_adversarial_name_dim_type_and_span_vectors() -> None:
    out_name_len = [11]
    out_dim_count = [22]
    out_dims_cells = [33]
    out_required_bytes = [44]
    out_type_value = [55]
    out_tensor_offset = [66]
    out_next_cursor = [77]

    too_long = u64((1 << 20) + 1)
    err = parse_one_checked_nopartial_commit_only(
        too_long,
        len(too_long),
        0,
        out_name_len,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED or err == GGUF_TENSOR_PARSE_ERR_NULL_PTR or err != GGUF_TENSOR_PARSE_OK

    bad_dims = gguf_string_bytes(b"bad.dims") + u32(0)
    err = parse_one_checked_nopartial_commit_only(
        bad_dims,
        len(bad_dims),
        0,
        out_name_len,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_DIMS

    unknown_type = gguf_string_bytes(b"bad.type") + u32(1) + u64(128) + u32(123456) + u64(0)
    err = parse_one_checked_nopartial_commit_only(
        unknown_type,
        len(unknown_type),
        0,
        out_name_len,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_next_cursor,
    )
    assert err != GGUF_TENSOR_PARSE_OK

    trunc_offset = tensor_entry("trunc.offset", [32, 64], 8, 1234)[:-5]
    err = parse_one_checked_nopartial_commit_only(
        trunc_offset,
        len(trunc_offset),
        0,
        out_name_len,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED

    assert out_name_len == [11]
    assert out_dim_count == [22]
    assert out_dims_cells == [33]
    assert out_required_bytes == [44]
    assert out_type_value == [55]
    assert out_tensor_offset == [66]
    assert out_next_cursor == [77]


def test_success_and_randomized_parity_against_explicit_composition() -> None:
    first = tensor_entry("tok", [32000, 2048], 2, 0)
    second = tensor_entry("blk.0.attn_q.weight", [2048, 2048], 8, 917504)
    payload = first + second

    out_name_len = [0]
    out_dim_count = [0]
    out_dims_cells = [0]
    out_required_bytes = [0]
    out_type_value = [0]
    out_tensor_offset = [0]
    out_next_cursor = [0]

    err = parse_one_checked_nopartial_commit_only(
        payload,
        len(payload),
        0,
        out_name_len,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_OK
    assert out_name_len[0] == 3
    assert out_dim_count[0] == 2
    assert out_dims_cells[0] == 2
    assert out_required_bytes[0] == len(first)
    assert out_next_cursor[0] == len(first)

    err = parse_one_checked_nopartial_commit_only(
        payload,
        len(payload),
        len(first),
        out_name_len,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_OK
    assert out_required_bytes[0] == len(second)
    assert out_next_cursor[0] == len(payload)

    rng = random.Random(20260422_1009)
    for _ in range(3000):
        name_len = rng.randint(1, 48)
        name_raw = bytes(rng.randint(97, 122) for _ in range(name_len))

        n_dims = rng.randint(1, 8)
        dims: list[int] = []
        prod = 1
        for _d in range(n_dims):
            max_dim = max(1, min(1 << 12, U64_MAX // prod))
            dim = rng.randint(1, max_dim)
            dims.append(dim)
            prod *= dim

        ggml_type = rng.choice(tuple(KNOWN_TYPES))
        offset = rng.randint(0, 1 << 40)

        entry = gguf_string_bytes(name_raw)
        entry += u32(n_dims)
        for dim in dims:
            entry += u64(dim)
        entry += u32(ggml_type)
        entry += u64(offset)

        explicit_status, explicit = parse_one_checked_nopartial_commit_only_explicit_composition(
            entry,
            len(entry),
            0,
        )

        out_name_len = [9991]
        out_dim_count = [9992]
        out_dims_cells = [9993]
        out_required_bytes = [9994]
        out_type_value = [9995]
        out_tensor_offset = [9996]
        out_next_cursor = [9997]

        status = parse_one_checked_nopartial_commit_only(
            entry,
            len(entry),
            0,
            out_name_len,
            out_dim_count,
            out_dims_cells,
            out_required_bytes,
            out_type_value,
            out_tensor_offset,
            out_next_cursor,
        )

        assert status == explicit_status
        assert status == GGUF_TENSOR_PARSE_OK
        assert out_name_len[0] == explicit["name_len"]
        assert out_dim_count[0] == explicit["dim_count"]
        assert out_dims_cells[0] == explicit["dims_cells"]
        assert out_required_bytes[0] == explicit["required_bytes"]
        assert out_type_value[0] == explicit["type_value"]
        assert out_tensor_offset[0] == explicit["tensor_offset"]
        assert out_next_cursor[0] == explicit["next_cursor"]


if __name__ == "__main__":
    test_source_contains_iq1009_signature_and_publish_tuple()
    test_null_cursor_and_no_partial_publish_on_failure()
    test_adversarial_name_dim_type_and_span_vectors()
    test_success_and_randomized_parity_against_explicit_composition()
    print("gguf_tensor_info_read_checked_nopartial_commit_only=ok")
