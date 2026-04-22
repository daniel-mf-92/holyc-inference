#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnly (IQ-1008)."""

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
from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only import (
    parse_dims_type_offset_checked_nopartial_commit_only,
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


def parse_dims_type_offset_checked_nopartial_commit_only_preflight_only(
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

    staged_dim_count = [0]
    staged_dims_cells = [0]
    staged_required_bytes = [0]
    staged_type_value = [0]
    staged_tensor_offset = [0]
    staged_last_dim_index = [0]
    staged_next_cursor = [0]

    status = parse_dims_type_offset_checked_nopartial_commit_only(
        buf,
        size,
        cursor,
        staged_dim_count,
        staged_dims_cells,
        staged_required_bytes,
        staged_type_value,
        staged_tensor_offset,
        staged_last_dim_index,
        staged_next_cursor,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    canonical: dict = {}
    canonical_next_cursor = [0]
    status = parse_dims_type_offset_checked_nopartial(
        buf,
        size,
        cursor,
        canonical,
        canonical_next_cursor,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    canonical_dim_count = canonical["n_dims"]
    canonical_dims_cells = canonical_dim_count
    canonical_type_value = canonical["ggml_type"]
    canonical_tensor_offset = canonical["offset"]
    canonical_last_dim_index = canonical_dim_count - 1

    canonical_dims_bytes = u64_mul(canonical_dims_cells, 8)
    if canonical_dims_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW

    canonical_required_bytes = u64_add(4, canonical_dims_bytes)
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
        staged_dim_count[0] != canonical_dim_count
        or staged_dims_cells[0] != canonical_dims_cells
        or staged_required_bytes[0] != canonical_required_bytes
        or staged_type_value[0] != canonical_type_value
        or staged_tensor_offset[0] != canonical_tensor_offset
        or staged_last_dim_index[0] != canonical_last_dim_index
        or staged_next_cursor[0] != canonical_next_cursor[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_dim_count[0] = staged_dim_count[0]
    out_dims_cells[0] = staged_dims_cells[0]
    out_required_bytes[0] = staged_required_bytes[0]
    out_type_value[0] = staged_type_value[0]
    out_tensor_offset[0] = staged_tensor_offset[0]
    out_last_dim_index[0] = staged_last_dim_index[0]
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
    staged_dim_count = [0]
    staged_dims_cells = [0]
    staged_required_bytes = [0]
    staged_type_value = [0]
    staged_tensor_offset = [0]
    staged_last_dim_index = [0]
    staged_next_cursor = [0]

    err = parse_dims_type_offset_checked_nopartial_commit_only(
        buf,
        size,
        cursor,
        staged_dim_count,
        staged_dims_cells,
        staged_required_bytes,
        staged_type_value,
        staged_tensor_offset,
        staged_last_dim_index,
        staged_next_cursor,
    )
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    canonical: dict = {}
    canonical_next_cursor = [0]
    err = parse_dims_type_offset_checked_nopartial(
        buf,
        size,
        cursor,
        canonical,
        canonical_next_cursor,
    )
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    canonical_dim_count = canonical["n_dims"]
    canonical_dims_cells = canonical_dim_count
    canonical_type_value = canonical["ggml_type"]
    canonical_tensor_offset = canonical["offset"]
    canonical_last_dim_index = canonical_dim_count - 1

    canonical_dims_bytes = u64_mul(canonical_dims_cells, 8)
    if canonical_dims_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW

    canonical_required_bytes = u64_add(4, canonical_dims_bytes)
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
        staged_dim_count[0] != canonical_dim_count
        or staged_dims_cells[0] != canonical_dims_cells
        or staged_required_bytes[0] != canonical_required_bytes
        or staged_type_value[0] != canonical_type_value
        or staged_tensor_offset[0] != canonical_tensor_offset
        or staged_last_dim_index[0] != canonical_last_dim_index
        or staged_next_cursor[0] != canonical_next_cursor[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_dim_count[0] = staged_dim_count[0]
    out_dims_cells[0] = staged_dims_cells[0]
    out_required_bytes[0] = staged_required_bytes[0]
    out_type_value[0] = staged_type_value[0]
    out_tensor_offset[0] = staged_tensor_offset[0]
    out_last_dim_index[0] = staged_last_dim_index[0]
    return GGUF_TENSOR_PARSE_OK


def test_source_contains_iq1008_signature_and_tuple_parity_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFTensorParseOne(", 1)[0]

    assert "status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnly(" in body
    assert "status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartial(" in body
    assert "if (staged_dim_count != canonical_dim_count ||" in body
    assert "*out_dim_count = staged_dim_count;" in body
    assert "*out_dims_cells = staged_dims_cells;" in body
    assert "*out_required_bytes = staged_required_bytes;" in body
    assert "*out_type_value = staged_type_value;" in body
    assert "*out_tensor_offset = staged_tensor_offset;" in body
    assert "*out_last_dim_index = staged_last_dim_index;" in body


def test_known_vector_success_and_null_rejection() -> None:
    payload = dims_type_offset_entry([64, 32, 16], 8, 4096)

    out_dim_count = [0x11]
    out_dims_cells = [0x12]
    out_required_bytes = [0x13]
    out_type_value = [0x14]
    out_tensor_offset = [0x15]
    out_last_dim_index = [0x16]

    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only(
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
    assert out_tensor_offset == [4096]
    assert out_last_dim_index == [2]

    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only(
        None,
        len(payload),
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_NULL_PTR


def test_adversarial_dim_type_span_overflow_vectors() -> None:
    out_dim_count = [901]
    out_dims_cells = [902]
    out_required_bytes = [903]
    out_type_value = [904]
    out_tensor_offset = [905]
    out_last_dim_index = [906]

    bad_type = dims_type_offset_entry([4, 4], 999, 0)
    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only(
        bad_type,
        len(bad_type),
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
    )
    assert err != GGUF_TENSOR_PARSE_OK
    assert out_dim_count == [901]
    assert out_dims_cells == [902]

    dim_overflow = dims_type_offset_entry([1 << 63, 3], 2, 0)
    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only(
        dim_overflow,
        len(dim_overflow),
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW

    truncated = dims_type_offset_entry([2, 2, 2], 8, 42)[:-7]
    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only(
        truncated,
        len(truncated),
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED

    tail = dims_type_offset_entry([1], 2, 0)
    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only(
        tail,
        U64_MAX,
        U64_MAX - 1,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    assert out_required_bytes == [903]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(202604221008)
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

        cursor = prefix_len

        out_dim_count_a = [0x21]
        out_dims_cells_a = [0x22]
        out_required_bytes_a = [0x23]
        out_type_value_a = [0x24]
        out_tensor_offset_a = [0x25]
        out_last_dim_index_a = [0x26]

        out_dim_count_b = [0x21]
        out_dims_cells_b = [0x22]
        out_required_bytes_b = [0x23]
        out_type_value_b = [0x24]
        out_tensor_offset_b = [0x25]
        out_last_dim_index_b = [0x26]

        size = len(payload)
        if rng.random() < 0.2:
            size = max(cursor, size - rng.randint(1, 4))

        err_a = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only(
            payload,
            size,
            cursor,
            out_dim_count_a,
            out_dims_cells_a,
            out_required_bytes_a,
            out_type_value_a,
            out_tensor_offset_a,
            out_last_dim_index_a,
        )

        err_b = explicit_checked_composition(
            payload,
            size,
            cursor,
            out_dim_count_b,
            out_dims_cells_b,
            out_required_bytes_b,
            out_type_value_b,
            out_tensor_offset_b,
            out_last_dim_index_b,
        )

        assert err_a == err_b
        assert out_dim_count_a == out_dim_count_b
        assert out_dims_cells_a == out_dims_cells_b
        assert out_required_bytes_a == out_required_bytes_b
        assert out_type_value_a == out_type_value_b
        assert out_tensor_offset_a == out_tensor_offset_b
        assert out_last_dim_index_a == out_last_dim_index_b
