#!/usr/bin/env python3
"""Harness for GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly (IQ-1025)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial import (
    GGUF_TENSOR_PARSE_ERR_BAD_DIM_COUNT,
    GGUF_TENSOR_PARSE_ERR_NULL_PTR,
    GGUF_TENSOR_PARSE_ERR_TRUNCATED,
    GGUF_TENSOR_PARSE_OK,
    U64_MAX,
    dims_type_offset_entry,
)
from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity import (
    parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity,
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
    if a == 0 or b == 0:
        return 0
    if a > U64_MAX // b:
        return None
    return a * b


def parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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

    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity(
        buf,
        size,
        cursor,
        staged_dim_count,
        staged_dims_cells,
        staged_required_bytes,
        staged_type_value,
        staged_tensor_offset,
        staged_last_dim_index,
    )
    if err != GGUF_TENSOR_PARSE_OK:
        return err

    if snapshot_buf is not buf or snapshot_size != size or snapshot_cursor != cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if staged_dim_count[0] == 0:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIM_COUNT
    if staged_dims_cells[0] != staged_dim_count[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if staged_last_dim_index[0] != staged_dim_count[0] - 1:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    dims_bytes = u64_mul(staged_dims_cells[0], 8)
    if dims_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    required = u64_add(4, dims_bytes)
    if required is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    required = u64_add(required, 4)
    if required is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    required = u64_add(required, 8)
    if required is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if staged_required_bytes[0] != required:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    next_cursor = u64_add(cursor, staged_required_bytes[0])
    if next_cursor is None or next_cursor > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_dim_count[0] = staged_dim_count[0]
    out_dims_cells[0] = staged_dims_cells[0]
    out_required_bytes[0] = staged_required_bytes[0]
    out_type_value[0] = staged_type_value[0]
    out_tensor_offset[0] = staged_tensor_offset[0]
    out_last_dim_index[0] = staged_last_dim_index[0]
    return GGUF_TENSOR_PARSE_OK


def explicit_checked_composition(*args):
    return parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        *args
    )


def test_source_contains_iq1025_signature_and_commit_only_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFTensorParseOne(", 1)[0]

    assert "status = GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "if (!staged_dim_count)" in body
    assert "if (staged_dims_cells != staged_dim_count)" in body
    assert "if (staged_last_dim_index != (U64)(staged_dim_count - 1))" in body
    assert "if (!GGUFTensorTryMulU64(staged_dims_cells, 8, &canonical_dims_bytes))" in body
    assert "if (!GGUFTensorTryAddU64(cursor, staged_required_bytes, &canonical_next_cursor))" in body
    assert "if (canonical_next_cursor > size)" in body
    assert "*out_dim_count = staged_dim_count;" in body
    assert "*out_dims_cells = staged_dims_cells;" in body
    assert "*out_required_bytes = staged_required_bytes;" in body
    assert "*out_type_value = staged_type_value;" in body
    assert "*out_tensor_offset = staged_tensor_offset;" in body
    assert "*out_last_dim_index = staged_last_dim_index;" in body


def test_known_vector_success_and_no_partial_on_error() -> None:
    payload = dims_type_offset_entry([64, 32, 16], 8, 777)

    out_dim_count = [0x41]
    out_dims_cells = [0x42]
    out_required_bytes = [0x43]
    out_type_value = [0x44]
    out_tensor_offset = [0x45]
    out_last_dim_index = [0x46]

    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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
    assert out_tensor_offset == [777]
    assert out_last_dim_index == [2]

    before = (
        out_dim_count[0],
        out_dims_cells[0],
        out_required_bytes[0],
        out_type_value[0],
        out_tensor_offset[0],
        out_last_dim_index[0],
    )
    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        payload,
        len(payload) - 1,
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
    ) == before


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(202604221025)

    for i in range(800):
        dim_count = rng.randint(1, 4)
        dims = [rng.randint(1, 256) for _ in range(dim_count)]
        ggml_type = rng.choice([0, 1, 2, 8, 16])
        offset = rng.randint(0, 1 << 20)
        payload = dims_type_offset_entry(dims, ggml_type, offset)

        cursor = rng.randint(0, 2)
        prefix = bytes(rng.getrandbits(8) for _ in range(cursor))
        buf = prefix + payload
        size = len(buf)

        if rng.random() < 0.25:
            size = max(cursor, size - rng.randint(1, min(5, max(1, size - cursor))))

        out_a = [[0x11], [0x12], [0x13], [0x14], [0x15], [0x16]]
        out_b = [[0x11], [0x12], [0x13], [0x14], [0x15], [0x16]]

        got = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only(
            buf,
            size,
            cursor,
            out_a[0],
            out_a[1],
            out_a[2],
            out_a[3],
            out_a[4],
            out_a[5],
        )
        want = explicit_checked_composition(
            buf,
            size,
            cursor,
            out_b[0],
            out_b[1],
            out_b[2],
            out_b[3],
            out_b[4],
            out_b[5],
        )

        assert got == want
        for slot_a, slot_b in zip(out_a, out_b):
            assert slot_a == slot_b


if __name__ == "__main__":
    test_source_contains_iq1025_signature_and_commit_only_contract()
    test_known_vector_success_and_no_partial_on_error()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
