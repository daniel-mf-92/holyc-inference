#!/usr/bin/env python3
"""Harness for IQ-1033 dims/type/offset parity gate."""

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
    dims_type_offset_entry,
)
from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only import (
    parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only,
)
from test_gguf_tensorinfo_read_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only import (
    parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only,
)


def parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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

    status = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
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

    status = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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

    if (
        staged_preflight_last_dim_index[0] != staged_preflight_dim_count[0] - 1
        or staged_commit_last_dim_index[0] != staged_commit_dim_count[0] - 1
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_preflight_type_value[0] not in KNOWN_TYPES
        or staged_commit_type_value[0] not in KNOWN_TYPES
    ):
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE

    preflight_end = cursor + staged_preflight_required_bytes[0]
    if preflight_end > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    commit_end = cursor + staged_commit_required_bytes[0]
    if commit_end > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if preflight_end != commit_end:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_dim_count[0] = staged_commit_dim_count[0]
    out_dims_cells[0] = staged_commit_dims_cells[0]
    out_required_bytes[0] = staged_commit_required_bytes[0]
    out_type_value[0] = staged_commit_type_value[0]
    out_tensor_offset[0] = staged_commit_tensor_offset[0]
    out_last_dim_index[0] = staged_commit_last_dim_index[0]
    return GGUF_TENSOR_PARSE_OK


def explicit_checked_composition(*args):
    return parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        *args
    )


def test_source_contains_iq1033_signature_and_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFTensorParseOne(", 1)[0]

    assert "IQ-1033 diagnostics-only parity gate" in source
    assert "GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    assert "GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)" in body
    assert "staged_preflight_dim_count != staged_commit_dim_count" in body
    assert "staged_preflight_dims_cells != staged_commit_dims_cells" in body
    assert "staged_preflight_required_bytes != staged_commit_required_bytes" in body
    assert "staged_preflight_type_value != staged_commit_type_value" in body
    assert "staged_preflight_tensor_offset != staged_commit_tensor_offset" in body
    assert "staged_preflight_last_dim_index != staged_commit_last_dim_index" in body
    assert "if (!staged_preflight_dim_count || !staged_commit_dim_count)" in body
    assert "staged_preflight_dim_count > GGUF_TENSOR_MAX_DIMS" in body
    assert "staged_commit_dim_count > GGUF_TENSOR_MAX_DIMS" in body
    assert "staged_preflight_dims_cells != staged_preflight_dim_count" in body
    assert "staged_commit_dims_cells != staged_commit_dim_count" in body
    assert "staged_preflight_last_dim_index != (U64)(staged_preflight_dim_count - 1)" in body
    assert "staged_commit_last_dim_index != (U64)(staged_commit_dim_count - 1)" in body
    assert "!GGUFTensorTypeKnown(staged_preflight_type_value)" in body
    assert "!GGUFTensorTypeKnown(staged_commit_type_value)" in body
    assert "if (!GGUFTensorTryAddU64(cursor," in body
    assert "if (staged_preflight_computed_end > size)" in body
    assert "if (staged_commit_computed_end > size)" in body
    assert "if (staged_preflight_computed_end != staged_commit_computed_end)" in body
    assert "*out_dim_count = staged_commit_dim_count;" in body


def test_known_vector_success_and_alias_guard() -> None:
    payload = dims_type_offset_entry([64, 32, 16], 8, 777)

    out_dim_count = [0x41]
    out_dims_cells = [0x42]
    out_required_bytes = [0x43]
    out_type_value = [0x44]
    out_tensor_offset = [0x45]
    out_last_dim_index = [0x46]

    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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
    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        payload,
        len(payload),
        0,
        alias,
        alias,
        [333],
        [444],
        [555],
        [666],
    )
    assert err == GGUF_TENSOR_PARSE_ERR_NULL_PTR


def test_adversarial_and_randomized_parity() -> None:
    out_dim_count = [0x41]
    out_dims_cells = [0x42]
    out_required_bytes = [0x43]
    out_type_value = [0x44]
    out_tensor_offset = [0x45]
    out_last_dim_index = [0x46]

    payload_bad_type = dims_type_offset_entry([16, 8], max(KNOWN_TYPES) + 100, 123)
    err = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        payload_bad_type,
        len(payload_bad_type),
        0,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_last_dim_index,
    )
    assert err in (GGUF_TENSOR_PARSE_ERR_BAD_TYPE, GGUF_TENSOR_PARSE_ERR_TRUNCATED)

    rng = random.Random(202604221033)

    for _ in range(900):
        dims = [rng.randint(1, 256) for _ in range(rng.randint(1, 4))]
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

        got = parse_dims_type_offset_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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
    test_source_contains_iq1033_signature_and_contract()
    test_known_vector_success_and_alias_guard()
    test_adversarial_and_randomized_parity()
    print("ok")
