#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-1017)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial import (
    GGUF_TENSOR_PARSE_ERR_NULL_PTR,
    GGUF_TENSOR_PARSE_ERR_TRUNCATED,
    GGUF_TENSOR_PARSE_OK,
    U64_MAX,
    dims_type_offset_entry,
)
from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only import (
    parse_dims_type_offset_checked_nopartial_commit_only,
)
from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only_preflight_only import (
    parse_dims_type_offset_checked_nopartial_commit_only_preflight_only,
)


def u64_add(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None
    if a > U64_MAX - b:
        return None
    return a + b


def parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity(
    buf: bytes | None,
    size: int,
    cursor: int,
    out_dim_count: list[int] | None,
    out_dims_cells: list[int] | None,
    out_required_bytes: list[int] | None,
    out_type_value: list[int] | None,
    out_tensor_offset: list[int] | None,
    out_last_dim_index: list[int] | None,
) -> int:
    if (
        buf is None
        or out_dim_count is None
        or out_dims_cells is None
        or out_required_bytes is None
        or out_type_value is None
        or out_tensor_offset is None
        or out_last_dim_index is None
    ):
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR

    if (
        out_dim_count is out_dims_cells
        or out_dim_count is out_required_bytes
        or out_dim_count is out_type_value
        or out_dim_count is out_tensor_offset
        or out_dim_count is out_last_dim_index
        or out_dims_cells is out_required_bytes
        or out_dims_cells is out_type_value
        or out_dims_cells is out_tensor_offset
        or out_dims_cells is out_last_dim_index
        or out_required_bytes is out_type_value
        or out_required_bytes is out_tensor_offset
        or out_required_bytes is out_last_dim_index
        or out_type_value is out_tensor_offset
        or out_type_value is out_last_dim_index
        or out_tensor_offset is out_last_dim_index
    ):
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR

    if cursor > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    snapshot_buf = buf
    snapshot_size = size
    snapshot_cursor = cursor

    staged_pre_dim_count = [0]
    staged_pre_dims_cells = [0]
    staged_pre_required_bytes = [0]
    staged_pre_type_value = [0]
    staged_pre_tensor_offset = [0]
    staged_pre_last_dim_index = [0]

    staged_commit_dim_count = [0]
    staged_commit_dims_cells = [0]
    staged_commit_required_bytes = [0]
    staged_commit_type_value = [0]
    staged_commit_tensor_offset = [0]
    staged_commit_last_dim_index = [0]
    staged_commit_next_cursor = [0]

    status = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only(
        buf,
        size,
        cursor,
        staged_pre_dim_count,
        staged_pre_dims_cells,
        staged_pre_required_bytes,
        staged_pre_type_value,
        staged_pre_tensor_offset,
        staged_pre_last_dim_index,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    status = parse_dims_type_offset_checked_nopartial_commit_only(
        buf,
        size,
        cursor,
        staged_commit_dim_count,
        staged_commit_dims_cells,
        staged_commit_required_bytes,
        staged_commit_type_value,
        staged_commit_tensor_offset,
        staged_commit_last_dim_index,
        staged_commit_next_cursor,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    if snapshot_buf is not buf or snapshot_size != size or snapshot_cursor != cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_pre_dim_count[0] != staged_commit_dim_count[0]
        or staged_pre_dims_cells[0] != staged_commit_dims_cells[0]
        or staged_pre_required_bytes[0] != staged_commit_required_bytes[0]
        or staged_pre_type_value[0] != staged_commit_type_value[0]
        or staged_pre_tensor_offset[0] != staged_commit_tensor_offset[0]
        or staged_pre_last_dim_index[0] != staged_commit_last_dim_index[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_pre_dims_cells[0] != staged_pre_dim_count[0]
        or staged_commit_dims_cells[0] != staged_commit_dim_count[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    staged_commit_computed_end = u64_add(cursor, staged_commit_required_bytes[0])
    if staged_commit_computed_end is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if staged_commit_computed_end != staged_commit_next_cursor[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_dim_count[0] = staged_commit_dim_count[0]
    out_dims_cells[0] = staged_commit_dims_cells[0]
    out_required_bytes[0] = staged_commit_required_bytes[0]
    out_type_value[0] = staged_commit_type_value[0]
    out_tensor_offset[0] = staged_commit_tensor_offset[0]
    out_last_dim_index[0] = staged_commit_last_dim_index[0]
    return GGUF_TENSOR_PARSE_OK


def explicit_checked_composition(
    buf: bytes,
    size: int,
    cursor: int,
    out_dim_count: list[int],
    out_dims_cells: list[int],
    out_required_bytes: list[int],
    out_type_value: list[int],
    out_tensor_offset: list[int],
    out_last_dim_index: list[int],
) -> int:
    staged_pre_dim_count = [0]
    staged_pre_dims_cells = [0]
    staged_pre_required_bytes = [0]
    staged_pre_type_value = [0]
    staged_pre_tensor_offset = [0]
    staged_pre_last_dim_index = [0]

    staged_commit_dim_count = [0]
    staged_commit_dims_cells = [0]
    staged_commit_required_bytes = [0]
    staged_commit_type_value = [0]
    staged_commit_tensor_offset = [0]
    staged_commit_last_dim_index = [0]
    staged_commit_next_cursor = [0]

    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only(
        buf,
        size,
        cursor,
        staged_pre_dim_count,
        staged_pre_dims_cells,
        staged_pre_required_bytes,
        staged_pre_type_value,
        staged_pre_tensor_offset,
        staged_pre_last_dim_index,
    )
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    err = parse_dims_type_offset_checked_nopartial_commit_only(
        buf,
        size,
        cursor,
        staged_commit_dim_count,
        staged_commit_dims_cells,
        staged_commit_required_bytes,
        staged_commit_type_value,
        staged_commit_tensor_offset,
        staged_commit_last_dim_index,
        staged_commit_next_cursor,
    )
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    if (
        staged_pre_dim_count[0] != staged_commit_dim_count[0]
        or staged_pre_dims_cells[0] != staged_commit_dims_cells[0]
        or staged_pre_required_bytes[0] != staged_commit_required_bytes[0]
        or staged_pre_type_value[0] != staged_commit_type_value[0]
        or staged_pre_tensor_offset[0] != staged_commit_tensor_offset[0]
        or staged_pre_last_dim_index[0] != staged_commit_last_dim_index[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_pre_dims_cells[0] != staged_pre_dim_count[0]
        or staged_commit_dims_cells[0] != staged_commit_dim_count[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    computed_end = u64_add(cursor, staged_commit_required_bytes[0])
    if computed_end is None or computed_end != staged_commit_next_cursor[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_dim_count[0] = staged_commit_dim_count[0]
    out_dims_cells[0] = staged_commit_dims_cells[0]
    out_required_bytes[0] = staged_commit_required_bytes[0]
    out_type_value[0] = staged_commit_type_value[0]
    out_tensor_offset[0] = staged_commit_tensor_offset[0]
    out_last_dim_index[0] = staged_commit_last_dim_index[0]
    return GGUF_TENSOR_PARSE_OK


def test_source_contains_iq1017_signature_and_tuple_parity_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFTensorParseOne(", 1)[0]

    assert "status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnly(" in body
    assert "if (staged_pre_dim_count != staged_commit_dim_count ||" in body
    assert "if (staged_pre_dims_cells != staged_pre_dim_count ||" in body
    assert "staged_commit_dims_cells != staged_commit_dim_count)" in body
    assert "if (!GGUFTensorTryAddU64(cursor," in body
    assert "staged_commit_required_bytes," in body
    assert "&staged_commit_computed_end))" in body
    assert "if (staged_commit_computed_end != staged_commit_next_cursor)" in body
    assert "out_dim_count_ptr = (U8 *)out_dim_count;" in body
    assert "*out_dim_count = staged_commit_dim_count;" in body
    assert "*out_dims_cells = staged_commit_dims_cells;" in body
    assert "*out_required_bytes = staged_commit_required_bytes;" in body
    assert "*out_type_value = staged_commit_type_value;" in body
    assert "*out_tensor_offset = staged_commit_tensor_offset;" in body
    assert "*out_last_dim_index = staged_commit_last_dim_index;" in body


def test_known_vector_alias_rejection_and_truncation_no_partial_publish() -> None:
    payload = dims_type_offset_entry([32, 16, 8], 8, 1234)

    out_dim_count = [0x41]
    out_dims_cells = [0x42]
    out_required_bytes = [0x43]
    out_type_value = [0x44]
    out_tensor_offset = [0x45]
    out_last_dim_index = [0x46]

    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity(
        payload,
        len(payload),
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
    )
    assert err == GGUF_TENSOR_PARSE_OK
    assert out_dim_count == [3]
    assert out_dims_cells == [3]
    assert out_required_bytes == [4 + 3 * 8 + 4 + 8]
    assert out_type_value == [8]
    assert out_tensor_offset == [1234]
    assert out_last_dim_index == [2]

    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity(
        payload,
        len(payload),
        0,
        out_dim_count,
        out_dim_count,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_NULL_PTR

    truncated_size = len(payload) - 2
    out_before = (
        out_dim_count[0],
        out_dims_cells[0],
        out_required_bytes[0],
        out_type_value[0],
        out_tensor_offset[0],
        out_last_dim_index[0],
    )
    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity(
        payload,
        truncated_size,
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    assert (
        out_dim_count[0],
        out_dims_cells[0],
        out_required_bytes[0],
        out_type_value[0],
        out_tensor_offset[0],
        out_last_dim_index[0],
    ) == out_before


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(202604221017)
    for _ in range(1800):
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

        cursor = prefix_len
        size = len(payload)
        if rng.random() < 0.2:
            size = max(cursor, size - rng.randint(1, 4))

        out_a_dim_count = [0x51]
        out_a_dims_cells = [0x52]
        out_a_required_bytes = [0x53]
        out_a_type_value = [0x54]
        out_a_tensor_offset = [0x55]
        out_a_last_dim_index = [0x56]

        out_b_dim_count = [0x51]
        out_b_dims_cells = [0x52]
        out_b_required_bytes = [0x53]
        out_b_type_value = [0x54]
        out_b_tensor_offset = [0x55]
        out_b_last_dim_index = [0x56]

        err_a = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity(
            payload,
            size,
            cursor,
            out_a_dim_count,
            out_a_dims_cells,
            out_a_required_bytes,
            out_a_type_value,
            out_a_tensor_offset,
            out_a_last_dim_index,
        )

        err_b = explicit_checked_composition(
            payload,
            size,
            cursor,
            out_b_dim_count,
            out_b_dims_cells,
            out_b_required_bytes,
            out_b_type_value,
            out_b_tensor_offset,
            out_b_last_dim_index,
        )

        assert err_a == err_b
        assert out_a_dim_count == out_b_dim_count
        assert out_a_dims_cells == out_b_dims_cells
        assert out_a_required_bytes == out_b_required_bytes
        assert out_a_type_value == out_b_type_value
        assert out_a_tensor_offset == out_b_tensor_offset
        assert out_a_last_dim_index == out_b_last_dim_index
