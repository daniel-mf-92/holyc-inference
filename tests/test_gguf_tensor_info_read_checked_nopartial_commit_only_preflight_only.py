#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnly (IQ-1010)."""

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
    KNOWN_TYPES,
    gguf_string_bytes,
    parse_one_checked_nopartial,
    tensor_entry,
    u32,
    u64,
)
from test_gguf_tensor_info_read_checked_nopartial_commit_only import (  # noqa: E402
    parse_one_checked_nopartial_commit_only,
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


def parse_one_checked_nopartial_commit_only_preflight_only(
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
    if size > 0x7FFFFFFFFFFFFFFF or cursor > 0x7FFFFFFFFFFFFFFF:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    snapshot_buf = buf
    snapshot_size = size
    snapshot_cursor = cursor

    staged_name_len = [0]
    staged_dim_count = [0]
    staged_dims_cells = [0]
    staged_required_bytes = [0]
    staged_type_value = [0]
    staged_tensor_offset = [0]
    staged_next_cursor = [0]

    status = parse_one_checked_nopartial_commit_only(
        buf,
        size,
        cursor,
        staged_name_len,
        staged_dim_count,
        staged_dims_cells,
        staged_required_bytes,
        staged_type_value,
        staged_tensor_offset,
        staged_next_cursor,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    canonical: dict = {}
    canonical_next_cursor = [0]
    status = parse_one_checked_nopartial(
        buf,
        size,
        cursor,
        canonical,
        canonical_next_cursor,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    canonical_name_len = canonical["name_len"]
    canonical_dim_count = canonical["n_dims"]
    canonical_dims_cells = canonical_dim_count
    canonical_type_value = canonical["ggml_type"]
    canonical_tensor_offset = canonical["offset"]

    canonical_required_bytes = u64_add(8, canonical_name_len)
    if canonical_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    canonical_required_bytes = u64_add(canonical_required_bytes, 4)
    if canonical_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    dims_bytes = u64_mul(canonical_dims_cells, 8)
    if dims_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW
    canonical_required_bytes = u64_add(canonical_required_bytes, dims_bytes)
    if canonical_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW

    canonical_required_bytes = u64_add(canonical_required_bytes, 4)
    if canonical_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    canonical_required_bytes = u64_add(canonical_required_bytes, 8)
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
        or staged_dim_count[0] != canonical_dim_count
        or staged_dims_cells[0] != canonical_dims_cells
        or staged_required_bytes[0] != canonical_required_bytes
        or staged_type_value[0] != canonical_type_value
        or staged_tensor_offset[0] != canonical_tensor_offset
        or staged_next_cursor[0] != canonical_next_cursor[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_name_len[0] = staged_name_len[0]
    out_dim_count[0] = staged_dim_count[0]
    out_dims_cells[0] = staged_dims_cells[0]
    out_required_bytes[0] = staged_required_bytes[0]
    out_type_value[0] = staged_type_value[0]
    out_tensor_offset[0] = staged_tensor_offset[0]
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
    staged_name_len = [0]
    staged_dim_count = [0]
    staged_dims_cells = [0]
    staged_required_bytes = [0]
    staged_type_value = [0]
    staged_tensor_offset = [0]
    staged_next_cursor = [0]

    err = parse_one_checked_nopartial_commit_only(
        buf,
        size,
        cursor,
        staged_name_len,
        staged_dim_count,
        staged_dims_cells,
        staged_required_bytes,
        staged_type_value,
        staged_tensor_offset,
        staged_next_cursor,
    )
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    canonical: dict = {}
    canonical_next_cursor = [0]
    err = parse_one_checked_nopartial(buf, size, cursor, canonical, canonical_next_cursor)
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    canonical_name_len = canonical["name_len"]
    canonical_dim_count = canonical["n_dims"]
    canonical_dims_cells = canonical_dim_count
    canonical_type_value = canonical["ggml_type"]
    canonical_tensor_offset = canonical["offset"]

    canonical_required_bytes = u64_add(8, canonical_name_len)
    if canonical_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    canonical_required_bytes = u64_add(canonical_required_bytes, 4)
    if canonical_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    dims_bytes = u64_mul(canonical_dims_cells, 8)
    if dims_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW
    canonical_required_bytes = u64_add(canonical_required_bytes, dims_bytes)
    if canonical_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW

    canonical_required_bytes = u64_add(canonical_required_bytes, 4)
    if canonical_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    canonical_required_bytes = u64_add(canonical_required_bytes, 8)
    if canonical_required_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    computed_end = u64_add(cursor, canonical_required_bytes)
    if computed_end is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if computed_end != canonical_next_cursor[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_name_len[0] != canonical_name_len
        or staged_dim_count[0] != canonical_dim_count
        or staged_dims_cells[0] != canonical_dims_cells
        or staged_required_bytes[0] != canonical_required_bytes
        or staged_type_value[0] != canonical_type_value
        or staged_tensor_offset[0] != canonical_tensor_offset
        or staged_next_cursor[0] != canonical_next_cursor[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_name_len[0] = staged_name_len[0]
    out_dim_count[0] = staged_dim_count[0]
    out_dims_cells[0] = staged_dims_cells[0]
    out_required_bytes[0] = staged_required_bytes[0]
    out_type_value[0] = staged_type_value[0]
    out_tensor_offset[0] = staged_tensor_offset[0]
    out_next_cursor[0] = staged_next_cursor[0]
    return GGUF_TENSOR_PARSE_OK


def test_source_contains_iq1010_signature_and_tuple_parity_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFParseTensorInfo(", 1)[0]

    assert "status = GGUFTensorInfoReadCheckedNoPartialCommitOnly(buf," in body
    assert "status = GGUFTensorInfoReadCheckedNoPartial(buf," in body
    assert "if (staged_name_len != canonical_name_len ||" in body
    assert "out_name_len_ptr = (U8 *)out_name_len;" in body
    assert "*out_name_len = staged_name_len;" in body
    assert "*out_dim_count = staged_dim_count;" in body
    assert "*out_dims_cells = staged_dims_cells;" in body
    assert "*out_required_bytes = staged_required_bytes;" in body
    assert "*out_type_value = staged_type_value;" in body
    assert "*out_tensor_offset = staged_tensor_offset;" in body
    assert "*out_next_cursor = staged_next_cursor;" in body


def test_null_alias_and_no_partial_publish_on_failure() -> None:
    payload = tensor_entry("tok_embd.weight", [32000, 2048], 2, 0)

    out_name_len = [101]
    out_dim_count = [102]
    out_dims_cells = [103]
    out_required_bytes = [104]
    out_type_value = [105]
    out_tensor_offset = [106]
    out_next_cursor = [107]

    err = parse_one_checked_nopartial_commit_only_preflight_only(
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

    err = parse_one_checked_nopartial_commit_only_preflight_only(
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
    assert out_name_len == [101]
    assert out_dim_count == [102]
    assert out_dims_cells == [103]

    truncated = payload[:-3]
    err = parse_one_checked_nopartial_commit_only_preflight_only(
        truncated,
        len(truncated),
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
    assert out_required_bytes == [104]
    assert out_type_value == [105]
    assert out_tensor_offset == [106]
    assert out_next_cursor == [107]




def test_cross_tuple_alias_and_signed_range_rejection() -> None:
    payload = tensor_entry("blk.0.attn_k.weight", [128, 64], 8, 2048)

    out_name_len = [401]
    out_dim_count = [402]
    out_dims_cells = [403]
    out_required_bytes = [404]
    out_type_value = [405]
    out_tensor_offset = [406]
    out_next_cursor = [407]

    err = parse_one_checked_nopartial_commit_only_preflight_only(
        payload,
        len(payload),
        0,
        out_name_len,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_dim_count,
        out_tensor_offset,
        out_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert out_name_len == [401]
    assert out_dim_count == [402]
    assert out_dims_cells == [403]
    assert out_required_bytes == [404]
    assert out_tensor_offset == [406]
    assert out_next_cursor == [407]

    err = parse_one_checked_nopartial_commit_only_preflight_only(
        payload,
        (1 << 63) + 5,
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

def test_adversarial_name_dim_type_span_overflow_vectors() -> None:
    out_name_len = [201]
    out_dim_count = [202]
    out_dims_cells = [203]
    out_required_bytes = [204]
    out_type_value = [205]
    out_tensor_offset = [206]
    out_next_cursor = [207]

    too_long = u64((1 << 20) + 1)
    err = parse_one_checked_nopartial_commit_only_preflight_only(
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
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN

    bad_dims = gguf_string_bytes(b"bad.dims") + u32(9)
    err = parse_one_checked_nopartial_commit_only_preflight_only(
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

    bad_type = tensor_entry("bad.type", [8, 8], 999, 0)
    err = parse_one_checked_nopartial_commit_only_preflight_only(
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

    overflow_dims = tensor_entry("ovf", [1 << 63, 3], 2, 0)
    err = parse_one_checked_nopartial_commit_only_preflight_only(
        overflow_dims,
        len(overflow_dims),
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

    err = parse_one_checked_nopartial_commit_only_preflight_only(
        tensor_entry("cursor.ovf", [4], 2, 0),
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

    assert out_name_len == [201]
    assert out_dim_count == [202]
    assert out_dims_cells == [203]
    assert out_required_bytes == [204]
    assert out_type_value == [205]
    assert out_tensor_offset == [206]
    assert out_next_cursor == [207]


def test_success_and_randomized_preflight_tuple_parity() -> None:
    fixed = tensor_entry("blk.0.attn_q.weight", [2048, 2048], 8, 917504)
    out_name_len = [0]
    out_dim_count = [0]
    out_dims_cells = [0]
    out_required_bytes = [0]
    out_type_value = [0]
    out_tensor_offset = [0]
    out_next_cursor = [0]

    err = parse_one_checked_nopartial_commit_only_preflight_only(
        fixed,
        len(fixed),
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
    assert out_name_len == [len("blk.0.attn_q.weight")]
    assert out_dim_count == [2]
    assert out_dims_cells == [2]
    assert out_required_bytes == [8 + len("blk.0.attn_q.weight") + 4 + 2 * 8 + 4 + 8]
    assert out_type_value == [8]
    assert out_tensor_offset == [917504]
    assert out_next_cursor == [len(fixed)]

    rng = random.Random(202604221010)
    known_types = tuple(KNOWN_TYPES)
    for i in range(1400):
        name_len = rng.randint(1, 32)
        name_raw = bytes(rng.randint(97, 122) for _ in range(name_len))

        n_dims = rng.randint(1, 8)
        dims: list[int] = []
        product = 1
        for _ in range(n_dims):
            max_dim = max(1, min(512, U64_MAX // product))
            dim = rng.randint(1, max_dim)
            dims.append(dim)
            product *= dim

        ggml_type = rng.choice(known_types)
        offset = rng.randint(0, 1 << 40)

        body = gguf_string_bytes(name_raw)
        body += u32(n_dims)
        for dim in dims:
            body += u64(dim)
        body += u32(ggml_type)
        body += u64(offset)

        prefix_len = rng.randint(0, 24)
        prefix = bytes(rng.randint(0, 255) for _ in range(prefix_len))
        payload = prefix + body

        got_name_len = [0x100 + i]
        got_dim_count = [0x200 + i]
        got_dims_cells = [0x300 + i]
        got_required_bytes = [0x400 + i]
        got_type_value = [0x500 + i]
        got_tensor_offset = [0x600 + i]
        got_next_cursor = [0x700 + i]

        exp_name_len = [0]
        exp_dim_count = [0]
        exp_dims_cells = [0]
        exp_required_bytes = [0]
        exp_type_value = [0]
        exp_tensor_offset = [0]
        exp_next_cursor = [0]

        err = parse_one_checked_nopartial_commit_only_preflight_only(
            payload,
            len(payload),
            prefix_len,
            got_name_len,
            got_dim_count,
            got_dims_cells,
            got_required_bytes,
            got_type_value,
            got_tensor_offset,
            got_next_cursor,
        )
        assert err == GGUF_TENSOR_PARSE_OK

        err = explicit_checked_composition(
            payload,
            len(payload),
            prefix_len,
            exp_name_len,
            exp_dim_count,
            exp_dims_cells,
            exp_required_bytes,
            exp_type_value,
            exp_tensor_offset,
            exp_next_cursor,
        )
        assert err == GGUF_TENSOR_PARSE_OK

        assert got_name_len == exp_name_len
        assert got_dim_count == exp_dim_count
        assert got_dims_cells == exp_dims_cells
        assert got_required_bytes == exp_required_bytes
        assert got_type_value == exp_type_value
        assert got_tensor_offset == exp_tensor_offset
        assert got_next_cursor == exp_next_cursor


if __name__ == "__main__":
    test_source_contains_iq1010_signature_and_tuple_parity_contract()
    test_null_alias_and_no_partial_publish_on_failure()
    test_cross_tuple_alias_and_signed_range_rejection()
    test_adversarial_name_dim_type_span_overflow_vectors()
    test_success_and_randomized_preflight_tuple_parity()
    print("gguf_tensor_info_read_checked_nopartial_commit_only_preflight_only=ok")
