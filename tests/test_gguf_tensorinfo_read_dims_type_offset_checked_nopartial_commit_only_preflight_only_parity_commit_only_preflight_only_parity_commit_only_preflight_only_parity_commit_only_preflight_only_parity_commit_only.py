#!/usr/bin/env python3
"""Harness for IQ-1095 dims/type/offset parity chain commit-only wrapper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial import (
    GGUF_TENSOR_PARSE_ERR_BAD_DIMS,
    GGUF_TENSOR_PARSE_ERR_BAD_TYPE,
    GGUF_TENSOR_PARSE_ERR_NULL_PTR,
    GGUF_TENSOR_PARSE_ERR_TRUNCATED,
    GGUF_TENSOR_PARSE_OK,
    KNOWN_TYPES,
    U64_MAX,
    dims_type_offset_entry,
)
from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only import (
    parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only,
)
from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only import (
    parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only,
)


def u64_add(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None
    if a > U64_MAX - b:
        return None
    return a + b


def parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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
    if size > 0x7FFFFFFFFFFFFFFF or cursor > 0x7FFFFFFFFFFFFFFF:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    snapshot_buf = buf
    snapshot_size = size
    snapshot_cursor = cursor

    staged_parity_dim_count = [0]
    staged_parity_dims_cells = [0]
    staged_parity_required_bytes = [0]
    staged_parity_type_value = [0]
    staged_parity_tensor_offset = [0]
    staged_parity_last_dim_index = [0]

    staged_commit_dim_count = [0]
    staged_commit_dims_cells = [0]
    staged_commit_required_bytes = [0]
    staged_commit_type_value = [0]
    staged_commit_tensor_offset = [0]
    staged_commit_last_dim_index = [0]

    status = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        buf,
        size,
        cursor,
        staged_parity_dim_count,
        staged_parity_dims_cells,
        staged_parity_required_bytes,
        staged_parity_type_value,
        staged_parity_tensor_offset,
        staged_parity_last_dim_index,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    status = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        buf,
        size,
        cursor,
        staged_commit_dim_count,
        staged_commit_dims_cells,
        staged_commit_required_bytes,
        staged_commit_type_value,
        staged_commit_tensor_offset,
        staged_commit_last_dim_index,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    parity_end = u64_add(cursor, staged_parity_required_bytes[0])
    commit_end = u64_add(cursor, staged_commit_required_bytes[0])
    if parity_end is None or commit_end is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if parity_end > size or commit_end > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if snapshot_buf is not buf or snapshot_size != size or snapshot_cursor != cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_parity_dim_count[0] != staged_commit_dim_count[0]
        or staged_parity_dims_cells[0] != staged_commit_dims_cells[0]
        or staged_parity_required_bytes[0] != staged_commit_required_bytes[0]
        or staged_parity_type_value[0] != staged_commit_type_value[0]
        or staged_parity_tensor_offset[0] != staged_commit_tensor_offset[0]
        or staged_parity_last_dim_index[0] != staged_commit_last_dim_index[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if staged_parity_dim_count[0] == 0 or staged_commit_dim_count[0] == 0:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS
    if staged_parity_dim_count[0] > 8 or staged_commit_dim_count[0] > 8:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS

    if staged_parity_dims_cells[0] != staged_parity_dim_count[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if staged_commit_dims_cells[0] != staged_commit_dim_count[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if staged_parity_last_dim_index[0] != staged_parity_dim_count[0] - 1:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if staged_commit_last_dim_index[0] != staged_commit_dim_count[0] - 1:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_parity_type_value[0] not in KNOWN_TYPES
        or staged_commit_type_value[0] not in KNOWN_TYPES
    ):
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE

    if parity_end != commit_end:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_dim_count[0] = staged_parity_dim_count[0]
    out_dims_cells[0] = staged_parity_dims_cells[0]
    out_required_bytes[0] = staged_parity_required_bytes[0]
    out_type_value[0] = staged_parity_type_value[0]
    out_tensor_offset[0] = staged_parity_tensor_offset[0]
    out_last_dim_index[0] = staged_parity_last_dim_index[0]
    return GGUF_TENSOR_PARSE_OK


def explicit_checked_composition(*args) -> int:
    return parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        *args
    )


def test_source_contains_iq1095_signature_and_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFTensorParseOne(", 1)[0]

    assert "IQ-1095 commit-only diagnostics wrapper" in source
    assert "GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    assert "GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "if (!GGUFTensorTryAddU64(cursor," in body
    assert "if (staged_parity_computed_end > size ||" in body
    assert "if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)" in body
    assert "if (staged_parity_dim_count != staged_commit_dim_count ||" in body
    assert "staged_parity_required_bytes != staged_commit_required_bytes" in body
    assert "staged_parity_tensor_offset != staged_commit_tensor_offset" in body
    assert "if (!GGUFTensorTypeKnown(staged_parity_type_value) ||" in body
    assert "*out_dim_count = staged_parity_dim_count;" in body


def test_known_vector_success_and_alias_guard() -> None:
    payload = dims_type_offset_entry([64, 32, 16], 8, 777)

    out_dim_count = [0x41]
    out_dims_cells = [0x42]
    out_required_bytes = [0x43]
    out_type_value = [0x44]
    out_tensor_offset = [0x45]
    out_last_dim_index = [0x46]

    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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

    alias = [999]
    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        payload,
        len(payload),
        0,
        alias,
        alias,
        [1],
        [2],
        [3],
        [4],
    )
    assert err == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert alias == [999]


def test_adversarial_dim_type_span_vectors() -> None:
    out_dim_count = [0x41]
    out_dims_cells = [0x42]
    out_required_bytes = [0x43]
    out_type_value = [0x44]
    out_tensor_offset = [0x45]
    out_last_dim_index = [0x46]

    # Bad dims: zero n_dims
    payload = dims_type_offset_entry([], 8, 123)
    err = explicit_checked_composition(
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
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_DIMS

    # Bad type: unknown ggml_type
    payload = dims_type_offset_entry([7], 99, 456)
    err = explicit_checked_composition(
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
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_TYPE

    # Truncated dim payload.
    payload = dims_type_offset_entry([11, 22, 33], 8, 789)
    payload = payload[:-9]
    err = explicit_checked_composition(
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
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED


def test_randomized_vectors_match_reference_chain() -> None:
    rng = random.Random(1095)

    for _ in range(300):
        n_dims = rng.randint(1, 8)
        dims = [rng.randint(1, 4) for _ in range(n_dims)]
        ggml_type = rng.choice(sorted(KNOWN_TYPES))
        offset = rng.randint(0, (1 << 48) - 1)

        payload = dims_type_offset_entry(dims, ggml_type, offset)
        size = len(payload)
        cursor = 0

        out_dim_count = [0]
        out_dims_cells = [0]
        out_required_bytes = [0]
        out_type_value = [0]
        out_tensor_offset = [0]
        out_last_dim_index = [0]

        err = explicit_checked_composition(
            payload,
            size,
            cursor,
            out_dim_count,
            out_dims_cells,
            out_required_bytes,
            out_type_value,
            out_tensor_offset,
            out_last_dim_index,
        )
        assert err == GGUF_TENSOR_PARSE_OK
        assert out_dim_count == [n_dims]
        assert out_dims_cells == [n_dims]
        assert out_required_bytes == [4 + 8 * n_dims + 4 + 8]
        assert out_type_value == [ggml_type]
        assert out_tensor_offset == [offset]
        assert out_last_dim_index == [n_dims - 1]


if __name__ == "__main__":
    test_source_contains_iq1095_signature_and_contract()
    test_known_vector_success_and_alias_guard()
    test_adversarial_dim_type_span_vectors()
    test_randomized_vectors_match_reference_chain()
    print("ok")
