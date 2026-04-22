#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-1015)."""

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
    tensor_entry,
    u32,
    u64,
    gguf_string_bytes,
)
from test_gguf_tensor_info_read_checked_nopartial_commit_only import (  # noqa: E402
    parse_one_checked_nopartial_commit_only,
)
from test_gguf_tensor_info_read_checked_nopartial_commit_only_preflight_only import (  # noqa: E402
    parse_one_checked_nopartial_commit_only_preflight_only,
)


def parse_one_checked_nopartial_commit_only_preflight_only_parity(
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

    staged_pre_name_len = [0]
    staged_pre_dim_count = [0]
    staged_pre_dims_cells = [0]
    staged_pre_required_bytes = [0]
    staged_pre_type_value = [0]
    staged_pre_tensor_offset = [0]
    staged_pre_next_cursor = [0]

    staged_commit_name_len = [0]
    staged_commit_dim_count = [0]
    staged_commit_dims_cells = [0]
    staged_commit_required_bytes = [0]
    staged_commit_type_value = [0]
    staged_commit_tensor_offset = [0]
    staged_commit_next_cursor = [0]

    err = parse_one_checked_nopartial_commit_only_preflight_only(
        buf,
        size,
        cursor,
        staged_pre_name_len,
        staged_pre_dim_count,
        staged_pre_dims_cells,
        staged_pre_required_bytes,
        staged_pre_type_value,
        staged_pre_tensor_offset,
        staged_pre_next_cursor,
    )
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    err = parse_one_checked_nopartial_commit_only(
        buf,
        size,
        cursor,
        staged_commit_name_len,
        staged_commit_dim_count,
        staged_commit_dims_cells,
        staged_commit_required_bytes,
        staged_commit_type_value,
        staged_commit_tensor_offset,
        staged_commit_next_cursor,
    )
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    if snapshot_buf is not buf or snapshot_size != size or snapshot_cursor != cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_pre_name_len[0] != staged_commit_name_len[0]
        or staged_pre_dim_count[0] != staged_commit_dim_count[0]
        or staged_pre_dims_cells[0] != staged_commit_dims_cells[0]
        or staged_pre_required_bytes[0] != staged_commit_required_bytes[0]
        or staged_pre_type_value[0] != staged_commit_type_value[0]
        or staged_pre_tensor_offset[0] != staged_commit_tensor_offset[0]
        or staged_pre_next_cursor[0] != staged_commit_next_cursor[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_name_len[0] = staged_commit_name_len[0]
    out_dim_count[0] = staged_commit_dim_count[0]
    out_dims_cells[0] = staged_commit_dims_cells[0]
    out_required_bytes[0] = staged_commit_required_bytes[0]
    out_type_value[0] = staged_commit_type_value[0]
    out_tensor_offset[0] = staged_commit_tensor_offset[0]
    out_next_cursor[0] = staged_commit_next_cursor[0]
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
    staged_pre_name_len = [0]
    staged_pre_dim_count = [0]
    staged_pre_dims_cells = [0]
    staged_pre_required_bytes = [0]
    staged_pre_type_value = [0]
    staged_pre_tensor_offset = [0]
    staged_pre_next_cursor = [0]

    staged_commit_name_len = [0]
    staged_commit_dim_count = [0]
    staged_commit_dims_cells = [0]
    staged_commit_required_bytes = [0]
    staged_commit_type_value = [0]
    staged_commit_tensor_offset = [0]
    staged_commit_next_cursor = [0]

    err = parse_one_checked_nopartial_commit_only_preflight_only(
        buf,
        size,
        cursor,
        staged_pre_name_len,
        staged_pre_dim_count,
        staged_pre_dims_cells,
        staged_pre_required_bytes,
        staged_pre_type_value,
        staged_pre_tensor_offset,
        staged_pre_next_cursor,
    )
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    err = parse_one_checked_nopartial_commit_only(
        buf,
        size,
        cursor,
        staged_commit_name_len,
        staged_commit_dim_count,
        staged_commit_dims_cells,
        staged_commit_required_bytes,
        staged_commit_type_value,
        staged_commit_tensor_offset,
        staged_commit_next_cursor,
    )
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    if (
        staged_pre_name_len[0] != staged_commit_name_len[0]
        or staged_pre_dim_count[0] != staged_commit_dim_count[0]
        or staged_pre_dims_cells[0] != staged_commit_dims_cells[0]
        or staged_pre_required_bytes[0] != staged_commit_required_bytes[0]
        or staged_pre_type_value[0] != staged_commit_type_value[0]
        or staged_pre_tensor_offset[0] != staged_commit_tensor_offset[0]
        or staged_pre_next_cursor[0] != staged_commit_next_cursor[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_name_len[0] = staged_commit_name_len[0]
    out_dim_count[0] = staged_commit_dim_count[0]
    out_dims_cells[0] = staged_commit_dims_cells[0]
    out_required_bytes[0] = staged_commit_required_bytes[0]
    out_type_value[0] = staged_commit_type_value[0]
    out_tensor_offset[0] = staged_commit_tensor_offset[0]
    out_next_cursor[0] = staged_commit_next_cursor[0]
    return GGUF_TENSOR_PARSE_OK


def test_source_contains_iq1015_signature_and_tuple_parity_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFParseTensorInfo(", 1)[0]

    assert "status = GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "status = GGUFTensorInfoReadCheckedNoPartialCommitOnly(" in body
    assert "if (staged_pre_name_len != staged_commit_name_len ||" in body
    assert "if (staged_pre_dims_cells != staged_pre_dim_count ||" in body
    assert "staged_commit_dims_cells != staged_commit_dim_count)" in body
    assert "staged_pre_computed_end != staged_pre_next_cursor" in body
    assert "staged_commit_computed_end != staged_commit_next_cursor" in body
    assert "out_name_len_ptr = (U8 *)out_name_len;" in body
    assert "*out_name_len = staged_commit_name_len;" in body
    assert "*out_dim_count = staged_commit_dim_count;" in body
    assert "*out_dims_cells = staged_commit_dims_cells;" in body
    assert "*out_required_bytes = staged_commit_required_bytes;" in body
    assert "*out_type_value = staged_commit_type_value;" in body
    assert "*out_tensor_offset = staged_commit_tensor_offset;" in body
    assert "*out_next_cursor = staged_commit_next_cursor;" in body


def test_null_alias_and_no_partial_publish_on_failure() -> None:
    payload = tensor_entry("tok_embd.weight", [32000, 2048], 2, 0)

    out_name_len = [101]
    out_dim_count = [102]
    out_dims_cells = [103]
    out_required_bytes = [104]
    out_type_value = [105]
    out_tensor_offset = [106]
    out_next_cursor = [107]

    err = parse_one_checked_nopartial_commit_only_preflight_only_parity(
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

    err = parse_one_checked_nopartial_commit_only_preflight_only_parity(
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

    truncated = payload[:-5]
    err = parse_one_checked_nopartial_commit_only_preflight_only_parity(
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


def test_adversarial_name_dim_type_span_overflow_vectors() -> None:
    out_name_len = [201]
    out_dim_count = [202]
    out_dims_cells = [203]
    out_required_bytes = [204]
    out_type_value = [205]
    out_tensor_offset = [206]
    out_next_cursor = [207]

    too_long = u64((1 << 20) + 1)
    err = parse_one_checked_nopartial_commit_only_preflight_only_parity(
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
    err = parse_one_checked_nopartial_commit_only_preflight_only_parity(
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
    err = parse_one_checked_nopartial_commit_only_preflight_only_parity(
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
    err = parse_one_checked_nopartial_commit_only_preflight_only_parity(
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

    err = parse_one_checked_nopartial_commit_only_preflight_only_parity(
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


def test_success_and_randomized_tuple_parity() -> None:
    fixed = tensor_entry("blk.0.attn_q.weight", [2048, 2048], 8, 917504)

    got_name_len = [0]
    got_dim_count = [0]
    got_dims_cells = [0]
    got_required_bytes = [0]
    got_type_value = [0]
    got_tensor_offset = [0]
    got_next_cursor = [0]

    exp_name_len = [0]
    exp_dim_count = [0]
    exp_dims_cells = [0]
    exp_required_bytes = [0]
    exp_type_value = [0]
    exp_tensor_offset = [0]
    exp_next_cursor = [0]

    err = parse_one_checked_nopartial_commit_only_preflight_only_parity(
        fixed,
        len(fixed),
        0,
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
        fixed,
        len(fixed),
        0,
        exp_name_len,
        exp_dim_count,
        exp_dims_cells,
        exp_required_bytes,
        exp_type_value,
        exp_tensor_offset,
        exp_next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_OK
    assert got_name_len == exp_name_len == [len("blk.0.attn_q.weight")]
    assert got_dim_count == exp_dim_count == [2]
    assert got_dims_cells == exp_dims_cells == [2]
    assert got_required_bytes == exp_required_bytes == [8 + len("blk.0.attn_q.weight") + 4 + 2 * 8 + 4 + 8]
    assert got_type_value == exp_type_value == [8]
    assert got_tensor_offset == exp_tensor_offset == [917504]
    assert got_next_cursor == exp_next_cursor == [len(fixed)]

    rng = random.Random(202604221015)
    known_types = tuple(KNOWN_TYPES)
    for i in range(1200):
        name_len = rng.randint(1, 40)
        name_raw = bytes(rng.randint(97, 122) for _ in range(name_len))

        n_dims = rng.randint(1, 8)
        dims: list[int] = []
        product = 1
        for _ in range(n_dims):
            max_dim = max(1, min(768, U64_MAX // product))
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

        err = parse_one_checked_nopartial_commit_only_preflight_only_parity(
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


def test_required_bytes_and_next_cursor_invariants_hold() -> None:
    rng = random.Random(20260422_1501)

    for _ in range(350):
        name = "".join(chr(97 + rng.randint(0, 25)) for _ in range(rng.randint(0, 32)))
        n_dims = rng.randint(1, 8)
        dims: list[int] = []
        product = 1
        for _ in range(n_dims):
            max_dim = max(1, min(768, U64_MAX // product))
            dim = rng.randint(1, max_dim)
            dims.append(dim)
            product *= dim
        ggml_type = rng.choice(tuple(KNOWN_TYPES))
        offset = rng.randint(0, 1 << 22)

        buf = tensor_entry(name, dims, ggml_type, offset)

        out_name_len = [0]
        out_dim_count = [0]
        out_dims_cells = [0]
        out_required_bytes = [0]
        out_type_value = [0]
        out_tensor_offset = [0]
        out_next_cursor = [0]

        err = parse_one_checked_nopartial_commit_only_preflight_only_parity(
            buf,
            len(buf),
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
        assert out_dims_cells[0] == out_dim_count[0]
        assert out_next_cursor[0] == out_required_bytes[0]

        expected_required = 8 + len(name.encode("utf-8")) + 4 + (8 * n_dims) + 4 + 8
        assert out_required_bytes[0] == expected_required


if __name__ == "__main__":
    test_source_contains_iq1015_signature_and_tuple_parity_contract()
    test_null_alias_and_no_partial_publish_on_failure()
    test_adversarial_name_dim_type_span_overflow_vectors()
    test_success_and_randomized_tuple_parity()
    test_required_bytes_and_next_cursor_invariants_hold()
    print("gguf_tensor_info_read_checked_nopartial_commit_only_preflight_only_parity=ok")
