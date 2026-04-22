#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoReadCheckedNoPartialCommitOnly (IQ-1009)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_gguf_tensor_info_read_checked_nopartial import (  # noqa: E402
    GGUF_TENSOR_PARSE_ERR_BAD_DIMS,
    GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN,
    GGUF_TENSOR_PARSE_ERR_BAD_TYPE,
    GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW,
    GGUF_TENSOR_PARSE_ERR_NULL_PTR,
    GGUF_TENSOR_PARSE_ERR_TRUNCATED,
    GGUF_TENSOR_PARSE_OK,
    U64_MAX,
    gguf_string_bytes,
    parse_one_checked_nopartial,
    tensor_entry,
    u32,
    u64,
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

    if (
        out_name_len is out_dim_count
        or out_name_len is out_dims_cells
        or out_name_len is out_required_bytes
        or out_name_len is out_type_value
        or out_name_len is out_tensor_offset
        or out_name_len is out_next_cursor
        or out_dim_count is out_dims_cells
        or out_dim_count is out_required_bytes
        or out_dim_count is out_type_value
        or out_dim_count is out_tensor_offset
        or out_dim_count is out_next_cursor
        or out_dims_cells is out_required_bytes
        or out_dims_cells is out_type_value
        or out_dims_cells is out_tensor_offset
        or out_dims_cells is out_next_cursor
        or out_required_bytes is out_type_value
        or out_required_bytes is out_tensor_offset
        or out_required_bytes is out_next_cursor
        or out_type_value is out_tensor_offset
        or out_type_value is out_next_cursor
        or out_tensor_offset is out_next_cursor
    ):
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR

    if cursor > size:
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


def explicit_checked_composition(
    buf: bytes,
    size: int,
    cursor: int,
    out_name_len: list[int],
    out_dim_count: list[int],
    out_dims_cells: list[int],
    out_required_bytes: list[int],
    out_type_value: list[int],
    out_tensor_offset: list[int],
    out_next_cursor: list[int],
) -> int:
    staged: dict = {}
    staged_next_cursor = [0]
    status = parse_one_checked_nopartial(buf, size, cursor, staged, staged_next_cursor)
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    dims_cells = staged["n_dims"]
    dims_bytes = u64_mul(dims_cells, 8)
    if dims_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW

    required_bytes = u64_add(8, staged["name_len"])
    if required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    required_bytes = u64_add(required_bytes, 4)
    if required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    required_bytes = u64_add(required_bytes, dims_bytes)
    if required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW
    required_bytes = u64_add(required_bytes, 4)
    if required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    required_bytes = u64_add(required_bytes, 8)
    if required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    computed_end = u64_add(cursor, required_bytes)
    if computed_end is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if computed_end != staged_next_cursor[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_name_len[0] = staged["name_len"]
    out_dim_count[0] = staged["n_dims"]
    out_dims_cells[0] = dims_cells
    out_required_bytes[0] = required_bytes
    out_type_value[0] = staged["ggml_type"]
    out_tensor_offset[0] = staged["offset"]
    out_next_cursor[0] = staged_next_cursor[0]
    return GGUF_TENSOR_PARSE_OK


def test_source_contains_iq1009_signature_and_atomic_publish_tuple() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadCheckedNoPartialCommitOnly(U8 *buf,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFParseTensorInfo(", 1)[0]

    assert "status = GGUFTensorInfoReadCheckedNoPartial(buf," in body
    assert "out_name_len_ptr = (U8 *)out_name_len;" in body
    assert "out_dim_count_ptr = (U8 *)out_dim_count;" in body
    assert "out_type_value_ptr = (U8 *)out_type_value;" in body
    assert "GGUFTensorTryMulU64(staged_dims_cells, 8, &dims_bytes)" in body
    assert "if (!GGUFTensorTryAddU64(cursor, staged_required_bytes, &computed_end))" in body
    assert "*out_name_len = staged_name_len;" in body
    assert "*out_dim_count = staged_dim_count;" in body
    assert "*out_dims_cells = staged_dims_cells;" in body
    assert "*out_required_bytes = staged_required_bytes;" in body
    assert "*out_type_value = staged_type_value;" in body
    assert "*out_tensor_offset = staged_tensor_offset;" in body
    assert "*out_next_cursor = staged_next_cursor;" in body


def test_null_alias_rejection_and_no_partial_publish_on_fail() -> None:
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

    err = parse_one_checked_nopartial_commit_only(
        payload,
        len(payload),
        0,
        out_name_len,
        out_name_len,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_NULL_PTR

    trunc = payload[:-3]
    err = parse_one_checked_nopartial_commit_only(
        trunc,
        len(trunc),
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
    assert out_name_len == [101]
    assert out_dim_count == [102]
    assert out_dims_cells == [103]
    assert out_required_bytes == [104]
    assert out_type_value == [105]
    assert out_tensor_offset == [106]
    assert out_next_cursor == [107]


def test_adversarial_name_dim_type_and_span_overflow_vectors() -> None:
    out_name_len = [201]
    out_dim_count = [202]
    out_dims_cells = [203]
    out_required_bytes = [204]
    out_type_value = [205]
    out_tensor_offset = [206]
    out_next_cursor = [207]

    too_long_hdr = u64((1 << 20) + 1)
    err = parse_one_checked_nopartial_commit_only(
        too_long_hdr,
        len(too_long_hdr),
        0,
        out_name_len,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN

    bad_dims = gguf_string_bytes(b"bad.dims") + u32(9)
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

    dim_overflow = tensor_entry("ovf", [1 << 63, 3], 2, 0)
    err = parse_one_checked_nopartial_commit_only(
        dim_overflow,
        len(dim_overflow),
        0,
        out_name_len,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW

    bad_type = tensor_entry("bad.type", [16, 16], 999, 0)
    err = parse_one_checked_nopartial_commit_only(
        bad_type,
        len(bad_type),
        0,
        out_name_len,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_TYPE

    # Span overflow on cursor + required_bytes.
    tiny = tensor_entry("t", [1], 2, 0)
    err = parse_one_checked_nopartial_commit_only(
        tiny,
        U64_MAX,
        U64_MAX - 1,
        out_name_len,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED


def test_success_and_randomized_tuple_parity() -> None:
    fixed = tensor_entry("blk.0.attn_q.weight", [64, 32, 16], 8, 4096)

    a_name_len = [0]
    a_dim_count = [0]
    a_dims_cells = [0]
    a_required_bytes = [0]
    a_type_value = [0]
    a_tensor_offset = [0]
    a_next_cursor = [0]

    b_name_len = [0]
    b_dim_count = [0]
    b_dims_cells = [0]
    b_required_bytes = [0]
    b_type_value = [0]
    b_tensor_offset = [0]
    b_next_cursor = [0]

    err_a = parse_one_checked_nopartial_commit_only(
        fixed,
        len(fixed),
        0,
        a_name_len,
        a_dim_count,
        a_dims_cells,
        a_required_bytes,
        a_type_value,
        a_tensor_offset,
        a_next_cursor,
    )
    err_b = explicit_checked_composition(
        fixed,
        len(fixed),
        0,
        b_name_len,
        b_dim_count,
        b_dims_cells,
        b_required_bytes,
        b_type_value,
        b_tensor_offset,
        b_next_cursor,
    )
    assert err_a == err_b == GGUF_TENSOR_PARSE_OK
    assert a_name_len == b_name_len
    assert a_dim_count == b_dim_count == [3]
    assert a_dims_cells == b_dims_cells == [3]
    assert a_required_bytes == b_required_bytes == [8 + len("blk.0.attn_q.weight") + 4 + 3 * 8 + 4 + 8]
    assert a_type_value == b_type_value == [8]
    assert a_tensor_offset == b_tensor_offset == [4096]
    assert a_next_cursor == b_next_cursor == [len(fixed)]

    rng = random.Random(202604221009)
    for _ in range(2000):
        name_len = rng.randint(1, 32)
        name = "".join(chr(rng.randint(97, 122)) for _ in range(name_len))
        n_dims = rng.randint(1, 8)

        dims: list[int] = []
        prod = 1
        for _j in range(n_dims):
            max_dim = max(1, min(1 << 12, U64_MAX // prod))
            dim = rng.randint(1, max_dim)
            dims.append(dim)
            prod *= dim

        ggml_type = rng.choice([2, 8, 12, 14, 30, 35])
        offset = rng.randint(0, 1 << 40)

        prefix_len = rng.randint(0, 23)
        prefix = bytes(rng.randint(0, 255) for _ in range(prefix_len))
        payload = prefix + tensor_entry(name, dims, ggml_type, offset)
        cursor = prefix_len

        a_name_len = [0x11]
        a_dim_count = [0x12]
        a_dims_cells = [0x13]
        a_required_bytes = [0x14]
        a_type_value = [0x15]
        a_tensor_offset = [0x16]
        a_next_cursor = [0x17]

        b_name_len = [0x21]
        b_dim_count = [0x22]
        b_dims_cells = [0x23]
        b_required_bytes = [0x24]
        b_type_value = [0x25]
        b_tensor_offset = [0x26]
        b_next_cursor = [0x27]

        err_a = parse_one_checked_nopartial_commit_only(
            payload,
            len(payload),
            cursor,
            a_name_len,
            a_dim_count,
            a_dims_cells,
            a_required_bytes,
            a_type_value,
            a_tensor_offset,
            a_next_cursor,
        )
        err_b = explicit_checked_composition(
            payload,
            len(payload),
            cursor,
            b_name_len,
            b_dim_count,
            b_dims_cells,
            b_required_bytes,
            b_type_value,
            b_tensor_offset,
            b_next_cursor,
        )

        assert err_a == err_b
        if err_a == GGUF_TENSOR_PARSE_OK:
            assert a_name_len == b_name_len
            assert a_dim_count == b_dim_count
            assert a_dims_cells == b_dims_cells
            assert a_required_bytes == b_required_bytes
            assert a_type_value == b_type_value
            assert a_tensor_offset == b_tensor_offset
            assert a_next_cursor == b_next_cursor


if __name__ == "__main__":
    test_source_contains_iq1009_signature_and_atomic_publish_tuple()
    test_null_alias_rejection_and_no_partial_publish_on_fail()
    test_adversarial_name_dim_type_and_span_overflow_vectors()
    test_success_and_randomized_tuple_parity()
    print("gguf_tensor_info_read_checked_nopartial_commit_only=ok")
