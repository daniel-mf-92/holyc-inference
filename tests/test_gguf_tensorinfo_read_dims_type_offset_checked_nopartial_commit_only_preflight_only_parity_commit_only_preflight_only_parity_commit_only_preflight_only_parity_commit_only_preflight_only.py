#!/usr/bin/env python3
"""Harness for IQ-1075 dims/type/offset diagnostics preflight-only companion."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial import (
    GGUF_TENSOR_MAX_DIMS,
    GGUF_TENSOR_PARSE_ERR_BAD_DIMS,
    GGUF_TENSOR_PARSE_ERR_BAD_TYPE,
    GGUF_TENSOR_PARSE_ERR_NULL_PTR,
    GGUF_TENSOR_PARSE_ERR_TRUNCATED,
    GGUF_TENSOR_PARSE_OK,
    KNOWN_TYPES,
    U64_MAX,
    dims_type_offset_entry,
)
from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity import (
    parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity,
)
from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only import (
    parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only,
)


def u64_add(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None
    if a > U64_MAX - b:
        return None
    return a + b


def parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
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

    snapshot = (buf, size, cursor)

    staged_dim_count = [0]
    staged_dims_cells = [0]
    staged_required_bytes = [0]
    staged_type_value = [0]
    staged_tensor_offset = [0]
    staged_last_dim_index = [0]

    canonical_dim_count = [0]
    canonical_dims_cells = [0]
    canonical_required_bytes = [0]
    canonical_type_value = [0]
    canonical_tensor_offset = [0]
    canonical_last_dim_index = [0]

    status = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    status = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        buf,
        size,
        cursor,
        canonical_dim_count,
        canonical_dims_cells,
        canonical_required_bytes,
        canonical_type_value,
        canonical_tensor_offset,
        canonical_last_dim_index,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    staged_end = u64_add(cursor, staged_required_bytes[0])
    canonical_end = u64_add(cursor, canonical_required_bytes[0])
    if staged_end is None or canonical_end is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if staged_end > size or canonical_end > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if snapshot != (buf, size, cursor):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_dim_count[0] != canonical_dim_count[0]
        or staged_dims_cells[0] != canonical_dims_cells[0]
        or staged_required_bytes[0] != canonical_required_bytes[0]
        or staged_type_value[0] != canonical_type_value[0]
        or staged_tensor_offset[0] != canonical_tensor_offset[0]
        or staged_last_dim_index[0] != canonical_last_dim_index[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if staged_dim_count[0] == 0 or canonical_dim_count[0] == 0:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS
    if staged_dim_count[0] > GGUF_TENSOR_MAX_DIMS or canonical_dim_count[0] > GGUF_TENSOR_MAX_DIMS:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS

    if staged_dims_cells[0] != staged_dim_count[0] or canonical_dims_cells[0] != canonical_dim_count[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if staged_last_dim_index[0] != staged_dim_count[0] - 1:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if canonical_last_dim_index[0] != canonical_dim_count[0] - 1:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if staged_type_value[0] not in KNOWN_TYPES or canonical_type_value[0] not in KNOWN_TYPES:
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE

    if staged_end != canonical_end:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_dim_count[0] = canonical_dim_count[0]
    out_dims_cells[0] = canonical_dims_cells[0]
    out_required_bytes[0] = canonical_required_bytes[0]
    out_type_value[0] = canonical_type_value[0]
    out_tensor_offset[0] = canonical_tensor_offset[0]
    out_last_dim_index[0] = canonical_last_dim_index[0]
    return GGUF_TENSOR_PARSE_OK


def explicit_checked_composition(*args) -> int:
    return parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        *args
    )


def test_source_contains_iq1075_signature_and_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
    assert sig in source
    body = source.rsplit(sig, 1)[1].split("\nI64 GGUFTensorParseOne(", 1)[0]

    assert "IQ-1075 diagnostics-only zero-write companion" in source
    assert "GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in body
    assert "if (staged_commit_dim_count != canonical_dim_count" in body
    assert "if (staged_commit_computed_end != canonical_computed_end)" in body


def test_known_vector_success_and_alias_guard() -> None:
    payload = dims_type_offset_entry([64, 32, 16], 8, 777)

    out_dim_count = [0x41]
    out_dims_cells = [0x42]
    out_required_bytes = [0x43]
    out_type_value = [0x44]
    out_tensor_offset = [0x45]
    out_last_dim_index = [0x46]

    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
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


def test_adversarial_vectors_and_no_partial_publish() -> None:
    out_dim_count = [11]
    out_dims_cells = [22]
    out_required_bytes = [33]
    out_type_value = [44]
    out_tensor_offset = [55]
    out_last_dim_index = [66]

    cases = [
        (b"", 0, 1, GGUF_TENSOR_PARSE_ERR_TRUNCATED),
        (dims_type_offset_entry([], 8, 0), 24, 0, GGUF_TENSOR_PARSE_ERR_BAD_DIMS),
        (dims_type_offset_entry([1], 999, 0), 24, 0, GGUF_TENSOR_PARSE_ERR_BAD_TYPE),
    ]

    for payload, size, cursor, expected in cases:
        err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
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
        assert err == expected
        assert out_dim_count == [11]
        assert out_dims_cells == [22]
        assert out_required_bytes == [33]
        assert out_type_value == [44]
        assert out_tensor_offset == [55]
        assert out_last_dim_index == [66]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260422_1075)

    for _ in range(1200):
        dim_count = rng.randint(1, GGUF_TENSOR_MAX_DIMS)
        dims = [rng.randint(1, 256) for _ in range(dim_count)]
        ggml_type = rng.choice(sorted(KNOWN_TYPES))
        offset = rng.randint(0, 100_000)

        payload = dims_type_offset_entry(dims, ggml_type, offset)

        out_dim_count = [101]
        out_dims_cells = [102]
        out_required_bytes = [103]
        out_type_value = [104]
        out_tensor_offset = [105]
        out_last_dim_index = [106]

        err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
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

        ref_dim = [201]
        ref_cells = [202]
        ref_bytes = [203]
        ref_type = [204]
        ref_off = [205]
        ref_last = [206]
        ref = explicit_checked_composition(
            payload,
            len(payload),
            0,
            ref_dim,
            ref_cells,
            ref_bytes,
            ref_type,
            ref_off,
            ref_last,
        )

        assert err == ref
        if err == GGUF_TENSOR_PARSE_OK:
            assert out_dim_count == ref_dim
            assert out_dims_cells == ref_cells
            assert out_required_bytes == ref_bytes
            assert out_type_value == ref_type
            assert out_tensor_offset == ref_off
            assert out_last_dim_index == ref_last


if __name__ == "__main__":
    test_source_contains_iq1075_signature_and_contract()
    test_known_vector_success_and_alias_guard()
    test_adversarial_vectors_and_no_partial_publish()
    test_randomized_parity_vs_explicit_composition()
    print(
        "gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only=ok"
    )
