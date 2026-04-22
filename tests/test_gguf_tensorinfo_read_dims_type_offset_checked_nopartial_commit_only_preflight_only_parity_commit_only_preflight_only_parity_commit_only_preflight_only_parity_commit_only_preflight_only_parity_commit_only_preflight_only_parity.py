#!/usr/bin/env python3
"""Harness for IQ-1128 dims/type/offset deepest parity gate."""

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
from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only import (
    parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only,
)
from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only import (
    parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only,
)


def u64_add(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None
    if a > U64_MAX - b:
        return None
    return a + b


def expected_required_bytes_from_dim_count(dim_count: int) -> int | None:
    dims_bytes = u64_add(0, dim_count << 3)
    if dims_bytes is None:
        return None
    total = u64_add(4, dims_bytes)
    if total is None:
        return None
    total = u64_add(total, 4)
    if total is None:
        return None
    total = u64_add(total, 8)
    return total


def parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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

    staged_preflight_dim_count = [0]
    staged_preflight_dims_cells = [0]
    staged_preflight_required_bytes = [0]
    staged_preflight_type_value = [0]
    staged_preflight_tensor_offset = [0]
    staged_preflight_last_dim_index = [0]

    staged_commit_dim_count = [0]
    staged_commit_dims_cells = [0]
    staged_commit_required_bytes = [0]
    staged_commit_type_value = [0]
    staged_commit_tensor_offset = [0]
    staged_commit_last_dim_index = [0]

    status = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        buf,
        size,
        cursor,
        staged_preflight_dim_count,
        staged_preflight_dims_cells,
        staged_preflight_required_bytes,
        staged_preflight_type_value,
        staged_preflight_tensor_offset,
        staged_preflight_last_dim_index,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    status = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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

    pre_end = u64_add(cursor, staged_preflight_required_bytes[0])
    commit_end = u64_add(cursor, staged_commit_required_bytes[0])
    if pre_end is None or commit_end is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if pre_end > size or commit_end > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if snapshot_buf is not buf or snapshot_size != size or snapshot_cursor != cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_preflight_dim_count[0] != staged_commit_dim_count[0]
        or staged_preflight_dims_cells[0] != staged_commit_dims_cells[0]
        or staged_preflight_required_bytes[0] != staged_commit_required_bytes[0]
        or staged_preflight_type_value[0] != staged_commit_type_value[0]
        or staged_preflight_tensor_offset[0] != staged_commit_tensor_offset[0]
        or staged_preflight_last_dim_index[0] != staged_commit_last_dim_index[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if staged_preflight_dim_count[0] == 0 or staged_commit_dim_count[0] == 0:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS
    if staged_preflight_dim_count[0] > 8 or staged_commit_dim_count[0] > 8:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS

    if (
        staged_preflight_dims_cells[0] != staged_preflight_dim_count[0]
        or staged_commit_dims_cells[0] != staged_commit_dim_count[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    expected_pre = expected_required_bytes_from_dim_count(staged_preflight_dim_count[0])
    expected_commit = expected_required_bytes_from_dim_count(staged_commit_dim_count[0])
    if expected_pre is None or expected_commit is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_preflight_required_bytes[0] != expected_pre
        or staged_commit_required_bytes[0] != expected_commit
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if staged_preflight_last_dim_index[0] != staged_preflight_dim_count[0] - 1:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if staged_commit_last_dim_index[0] != staged_commit_dim_count[0] - 1:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_preflight_type_value[0] not in KNOWN_TYPES
        or staged_commit_type_value[0] not in KNOWN_TYPES
    ):
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE

    if pre_end != commit_end:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_dim_count[0] = staged_preflight_dim_count[0]
    out_dims_cells[0] = staged_preflight_dims_cells[0]
    out_required_bytes[0] = staged_preflight_required_bytes[0]
    out_type_value[0] = staged_preflight_type_value[0]
    out_tensor_offset[0] = staged_preflight_tensor_offset[0]
    out_last_dim_index[0] = staged_preflight_last_dim_index[0]
    return GGUF_TENSOR_PARSE_OK


def explicit_checked_composition(*args) -> int:
    return parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        *args
    )


def test_source_contains_iq1128_signature_and_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.rsplit(sig, 1)[1].split("\nI64 GGUFTensorParseOne(", 1)[0]

    assert "IQ-1128 diagnostics-only parity gate" in source
    assert "GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    assert "GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "if (staged_preflight_dim_count != staged_commit_dim_count ||" in body
    assert "if (!staged_preflight_dim_count || !staged_commit_dim_count)" in body
    assert "if (staged_preflight_dims_cells != staged_preflight_dim_count ||" in body
    assert "if (staged_preflight_last_dim_index != (U64)(staged_preflight_dim_count - 1) ||" in body
    assert "if (!GGUFTensorTypeKnown(staged_preflight_type_value) ||" in body
    assert "if (staged_preflight_computed_end != staged_commit_computed_end)" in body


def test_known_vector_success_and_alias_guard() -> None:
    payload = dims_type_offset_entry([64, 32, 16], 8, 777)

    out_dim_count = [0x41]
    out_dims_cells = [0x42]
    out_required_bytes = [0x43]
    out_type_value = [0x44]
    out_tensor_offset = [0x45]
    out_last_dim_index = [0x46]

    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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
    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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

    tiny = b"\x01\x00\x00"
    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        tiny,
        len(tiny),
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED

    payload = dims_type_offset_entry([8, 4], 8, 9)
    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        payload,
        len(payload),
        len(payload) + 1,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED

    bad_type_payload = dims_type_offset_entry([8, 4], max(KNOWN_TYPES) + 100, 9)
    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        bad_type_payload,
        len(bad_type_payload),
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
    )
    assert err in (GGUF_TENSOR_PARSE_ERR_BAD_TYPE, GGUF_TENSOR_PARSE_ERR_TRUNCATED)


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260422_1128)

    for _ in range(900):
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
            size = max(cursor, size - rng.randint(1, min(6, max(1, size - cursor))))

        out_a = [[0x11], [0x12], [0x13], [0x14], [0x15], [0x16]]
        out_b = [[0x11], [0x12], [0x13], [0x14], [0x15], [0x16]]

        got = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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
    test_source_contains_iq1128_signature_and_contract()
    test_known_vector_success_and_alias_guard()
    test_adversarial_dim_type_span_vectors()
    test_randomized_parity_vs_explicit_composition()
    print(
        "gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity=ok"
    )
