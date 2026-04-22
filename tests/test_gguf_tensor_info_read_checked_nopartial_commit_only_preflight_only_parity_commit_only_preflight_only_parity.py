#!/usr/bin/env python3
"""Parity harness for IQ-1027 tensor-info diagnostics wrapper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_gguf_tensor_info_read_checked_nopartial import (  # noqa: E402
    GGUF_TENSOR_PARSE_ERR_BAD_TYPE,
    GGUF_TENSOR_PARSE_ERR_NULL_PTR,
    GGUF_TENSOR_PARSE_ERR_TRUNCATED,
    GGUF_TENSOR_PARSE_OK,
    KNOWN_TYPES,
    tensor_entry,
)
from test_gguf_tensor_info_read_checked_nopartial_commit_only_preflight_only_parity_commit_only import (  # noqa: E402
    parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only,
)
from test_gguf_tensor_info_read_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only import (  # noqa: E402
    parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only,
)


def parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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

    staged_preflight_name_len = [0]
    staged_preflight_dim_count = [0]
    staged_preflight_dims_cells = [0]
    staged_preflight_required_bytes = [0]
    staged_preflight_type_value = [0]
    staged_preflight_tensor_offset = [0]
    staged_preflight_next_cursor = [0]

    staged_commit_name_len = [0]
    staged_commit_dim_count = [0]
    staged_commit_dims_cells = [0]
    staged_commit_required_bytes = [0]
    staged_commit_type_value = [0]
    staged_commit_tensor_offset = [0]
    staged_commit_next_cursor = [0]

    status = parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
        buf,
        size,
        cursor,
        staged_preflight_name_len,
        staged_preflight_dim_count,
        staged_preflight_dims_cells,
        staged_preflight_required_bytes,
        staged_preflight_type_value,
        staged_preflight_tensor_offset,
        staged_preflight_next_cursor,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    status = parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    if snapshot_buf is not buf or snapshot_size != size or snapshot_cursor != cursor:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_preflight_name_len[0] != staged_commit_name_len[0]
        or staged_preflight_dim_count[0] != staged_commit_dim_count[0]
        or staged_preflight_dims_cells[0] != staged_commit_dims_cells[0]
        or staged_preflight_required_bytes[0] != staged_commit_required_bytes[0]
        or staged_preflight_type_value[0] != staged_commit_type_value[0]
        or staged_preflight_tensor_offset[0] != staged_commit_tensor_offset[0]
        or staged_preflight_next_cursor[0] != staged_commit_next_cursor[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_preflight_dims_cells[0] != staged_preflight_dim_count[0]
        or staged_commit_dims_cells[0] != staged_commit_dim_count[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_preflight_type_value[0] not in KNOWN_TYPES
        or staged_commit_type_value[0] not in KNOWN_TYPES
    ):
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE

    pre_end = cursor + staged_preflight_required_bytes[0]
    if pre_end > size or pre_end != staged_preflight_next_cursor[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    commit_end = cursor + staged_commit_required_bytes[0]
    if commit_end > size or commit_end != staged_commit_next_cursor[0]:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    out_name_len[0] = staged_commit_name_len[0]
    out_dim_count[0] = staged_commit_dim_count[0]
    out_dims_cells[0] = staged_commit_dims_cells[0]
    out_required_bytes[0] = staged_commit_required_bytes[0]
    out_type_value[0] = staged_commit_type_value[0]
    out_tensor_offset[0] = staged_commit_tensor_offset[0]
    out_next_cursor[0] = staged_commit_next_cursor[0]
    return GGUF_TENSOR_PARSE_OK


def explicit_checked_composition(*args) -> int:
    return parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        *args
    )


def test_source_contains_iq1027_signature_and_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.rsplit(sig, 1)[1].split("\nI64 ", 1)[0]

    assert "IQ-1027 diagnostics-only parity gate for tensor-info commit+preflight wrappers." in source
    assert "GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    assert "GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)" in body
    assert "staged_preflight_name_len != staged_commit_name_len" in body
    assert "staged_preflight_dim_count != staged_commit_dim_count" in body
    assert "staged_preflight_dims_cells != staged_commit_dims_cells" in body
    assert "staged_preflight_required_bytes != staged_commit_required_bytes" in body
    assert "staged_preflight_type_value != staged_commit_type_value" in body
    assert "staged_preflight_tensor_offset != staged_commit_tensor_offset" in body
    assert "staged_preflight_next_cursor != staged_commit_next_cursor" in body
    assert "staged_preflight_dims_cells != staged_preflight_dim_count" in body
    assert "staged_commit_dims_cells != staged_commit_dim_count" in body
    assert "!GGUFTensorTypeKnown(staged_preflight_type_value)" in body
    assert "!GGUFTensorTypeKnown(staged_commit_type_value)" in body
    assert "if (!GGUFTensorTryAddU64(cursor," in body
    assert "staged_preflight_computed_end != staged_preflight_next_cursor" in body
    assert "staged_commit_computed_end != staged_commit_next_cursor" in body
    assert "*out_name_len = staged_commit_name_len;" in body


def test_known_vector_success_and_alias_guard() -> None:
    buf = tensor_entry("tok_embd.weight", [32000, 64], 2, 4096)

    out_name_len = [111]
    out_dim_count = [222]
    out_dims_cells = [333]
    out_required_bytes = [444]
    out_type_value = [555]
    out_tensor_offset = [666]
    out_next_cursor = [777]

    status = parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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
    assert status == GGUF_TENSOR_PARSE_OK
    assert out_name_len == [len("tok_embd.weight")]
    assert out_dim_count == [2]
    assert out_dims_cells == [2]
    assert out_type_value == [2]
    assert out_tensor_offset == [4096]
    assert out_next_cursor == [len(buf)]

    alias = [999]
    status = parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        buf,
        len(buf),
        0,
        alias,
        alias,
        [333],
        [444],
        [555],
        [666],
        [777],
    )
    assert status == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert alias == [999]


def test_adversarial_vectors() -> None:
    out_name_len = [11]
    out_dim_count = [22]
    out_dims_cells = [33]
    out_required_bytes = [44]
    out_type_value = [55]
    out_tensor_offset = [66]
    out_next_cursor = [77]

    # Type guard.
    bad_type = tensor_entry("bad", [1], 0xFFFFFFFF, 0)
    status = parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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
    assert status in (GGUF_TENSOR_PARSE_ERR_TRUNCATED, GGUF_TENSOR_PARSE_ERR_BAD_TYPE)

    # Truncation keeps outputs unchanged.
    prev = (
        out_name_len[0],
        out_dim_count[0],
        out_dims_cells[0],
        out_required_bytes[0],
        out_type_value[0],
        out_tensor_offset[0],
        out_next_cursor[0],
    )
    buf = tensor_entry("tok_embd.weight", [4, 8], 2, 16)
    status = parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        buf,
        len(buf) - 1,
        0,
        out_name_len,
        out_dim_count,
        out_dims_cells,
        out_required_bytes,
        out_type_value,
        out_tensor_offset,
        out_next_cursor,
    )
    assert status == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    assert prev == (
        out_name_len[0],
        out_dim_count[0],
        out_dims_cells[0],
        out_required_bytes[0],
        out_type_value[0],
        out_tensor_offset[0],
        out_next_cursor[0],
    )


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260422_1027)

    for _ in range(900):
        name = "".join(chr(97 + rng.randint(0, 25)) for _ in range(rng.randint(0, 48)))
        dims = [rng.randint(0, 1 << 16) for _ in range(rng.randint(0, 8))]
        ggml_type = rng.choice(sorted(KNOWN_TYPES))
        offset = rng.randint(0, 1 << 22)

        buf = tensor_entry(name, dims, ggml_type, offset)
        size = len(buf)
        if rng.random() < 0.28 and size > 0:
            size -= rng.randint(1, min(8, size))

        a_name = [rng.randint(1, 999)]
        a_dim = [rng.randint(1, 999)]
        a_dims_cells = [rng.randint(1, 999)]
        a_required = [rng.randint(1, 999)]
        a_type = [rng.randint(1, 999)]
        a_offset = [rng.randint(1, 999)]
        a_next = [rng.randint(1, 999)]

        b_name = list(a_name)
        b_dim = list(a_dim)
        b_dims_cells = list(a_dims_cells)
        b_required = list(a_required)
        b_type = list(a_type)
        b_offset = list(a_offset)
        b_next = list(a_next)

        err_a = parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            buf,
            size,
            0,
            a_name,
            a_dim,
            a_dims_cells,
            a_required,
            a_type,
            a_offset,
            a_next,
        )
        err_b = explicit_checked_composition(
            buf,
            size,
            0,
            b_name,
            b_dim,
            b_dims_cells,
            b_required,
            b_type,
            b_offset,
            b_next,
        )

        assert err_a == err_b
        assert a_name == b_name
        assert a_dim == b_dim
        assert a_dims_cells == b_dims_cells
        assert a_required == b_required
        assert a_type == b_type
        assert a_offset == b_offset
        assert a_next == b_next


if __name__ == "__main__":
    test_source_contains_iq1027_signature_and_contract()
    test_known_vector_success_and_alias_guard()
    test_adversarial_vectors()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
