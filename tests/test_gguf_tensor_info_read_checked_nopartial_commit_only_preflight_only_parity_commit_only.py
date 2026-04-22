#!/usr/bin/env python3
"""Parity commit-only harness for IQ-1018 tensor-info diagnostics wrapper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_gguf_tensor_info_read_checked_nopartial import (  # noqa: E402
    GGUF_TENSOR_PARSE_ERR_NULL_PTR,
    GGUF_TENSOR_PARSE_ERR_TRUNCATED,
    GGUF_TENSOR_PARSE_OK,
    KNOWN_TYPES,
    tensor_entry,
)
from test_gguf_tensor_info_read_checked_nopartial_commit_only import (  # noqa: E402
    parse_one_checked_nopartial_commit_only,
)
from test_gguf_tensor_info_read_checked_nopartial_commit_only_preflight_only_parity import (  # noqa: E402
    parse_one_checked_nopartial_commit_only_preflight_only_parity,
)


def parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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

    staged_parity_name_len = [0]
    staged_parity_dim_count = [0]
    staged_parity_dims_cells = [0]
    staged_parity_required_bytes = [0]
    staged_parity_type_value = [0]
    staged_parity_tensor_offset = [0]
    staged_parity_next_cursor = [0]

    staged_commit_name_len = [0]
    staged_commit_dim_count = [0]
    staged_commit_dims_cells = [0]
    staged_commit_required_bytes = [0]
    staged_commit_type_value = [0]
    staged_commit_tensor_offset = [0]
    staged_commit_next_cursor = [0]

    status = parse_one_checked_nopartial_commit_only_preflight_only_parity(
        buf,
        size,
        cursor,
        staged_parity_name_len,
        staged_parity_dim_count,
        staged_parity_dims_cells,
        staged_parity_required_bytes,
        staged_parity_type_value,
        staged_parity_tensor_offset,
        staged_parity_next_cursor,
    )
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    status = parse_one_checked_nopartial_commit_only(
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
        staged_parity_name_len[0] != staged_commit_name_len[0]
        or staged_parity_dim_count[0] != staged_commit_dim_count[0]
        or staged_parity_dims_cells[0] != staged_commit_dims_cells[0]
        or staged_parity_required_bytes[0] != staged_commit_required_bytes[0]
        or staged_parity_type_value[0] != staged_commit_type_value[0]
        or staged_parity_tensor_offset[0] != staged_commit_tensor_offset[0]
        or staged_parity_next_cursor[0] != staged_commit_next_cursor[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_parity_dims_cells[0] != staged_parity_dim_count[0]
        or staged_commit_dims_cells[0] != staged_commit_dim_count[0]
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if (
        staged_parity_type_value[0] not in KNOWN_TYPES
        or staged_commit_type_value[0] not in KNOWN_TYPES
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    staged_parity_computed_end = cursor + staged_parity_required_bytes[0]
    if (
        staged_parity_computed_end != staged_parity_next_cursor[0]
        or staged_parity_computed_end > size
    ):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    staged_commit_computed_end = cursor + staged_commit_required_bytes[0]
    if (
        staged_commit_computed_end != staged_commit_next_cursor[0]
        or staged_commit_computed_end > size
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


def explicit_checked_composition(*args) -> int:
    return parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only(*args)


def test_source_contains_iq1018_signature_and_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly("
    assert sig in source
    body = source.rsplit(sig, 1)[1].split("\nI64 ", 1)[0]

    assert "IQ-1018 commit-only diagnostics wrapper" in source
    assert "GGUFTensorInfoReadCheckedNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "GGUFTensorInfoReadCheckedNoPartialCommitOnly(" in body
    assert "snapshot_buf = buf;" in body
    assert "snapshot_size = size;" in body
    assert "snapshot_cursor = cursor;" in body
    assert "if (snapshot_buf != buf || snapshot_size != size || snapshot_cursor != cursor)" in body
    assert "staged_parity_name_len != staged_commit_name_len" in body
    assert "staged_parity_dim_count != staged_commit_dim_count" in body
    assert "staged_parity_dims_cells != staged_commit_dims_cells" in body
    assert "staged_parity_required_bytes != staged_commit_required_bytes" in body
    assert "staged_parity_type_value != staged_commit_type_value" in body
    assert "staged_parity_tensor_offset != staged_commit_tensor_offset" in body
    assert "staged_parity_next_cursor != staged_commit_next_cursor" in body
    assert "staged_parity_dims_cells != staged_parity_dim_count" in body
    assert "staged_commit_dims_cells != staged_commit_dim_count" in body
    assert "!GGUFTensorTypeKnown(staged_parity_type_value)" in body
    assert "!GGUFTensorTypeKnown(staged_commit_type_value)" in body
    assert "GGUFTensorTryAddU64(cursor," in body
    assert "staged_parity_computed_end != staged_parity_next_cursor" in body
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

    status = parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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

    fail_name = [111]
    fail_dim = [222]
    fail_dims_cells = [333]
    fail_required = [444]
    fail_type = [555]
    fail_offset = [666]
    fail_next = [777]
    status = parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        buf,
        len(buf),
        0,
        fail_name,
        fail_name,
        fail_dims_cells,
        fail_required,
        fail_type,
        fail_offset,
        fail_next,
    )
    assert status == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert fail_name == [111]
    assert fail_dim == [222]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260422_1018)

    for _ in range(700):
        name_len = rng.randint(0, 40)
        name = "".join(chr(97 + rng.randint(0, 25)) for _ in range(name_len))

        n_dims = rng.randint(0, 8)
        dims = [rng.randint(0, 1 << 16) for _ in range(n_dims)]

        ggml_type = rng.choice(sorted(KNOWN_TYPES))
        offset = rng.randint(0, 1 << 24)

        buf = tensor_entry(name, dims, ggml_type, offset)
        size = len(buf)
        cursor = 0

        if rng.random() < 0.25 and size > 0:
            size -= rng.randint(1, min(6, size))

        out_a_name = [rng.randint(1, 999)]
        out_a_dim = [rng.randint(1, 999)]
        out_a_dims_cells = [rng.randint(1, 999)]
        out_a_required = [rng.randint(1, 999)]
        out_a_type = [rng.randint(1, 999)]
        out_a_offset = [rng.randint(1, 999)]
        out_a_next = [rng.randint(1, 999)]

        out_b_name = list(out_a_name)
        out_b_dim = list(out_a_dim)
        out_b_dims_cells = list(out_a_dims_cells)
        out_b_required = list(out_a_required)
        out_b_type = list(out_a_type)
        out_b_offset = list(out_a_offset)
        out_b_next = list(out_a_next)

        err_a = parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only(
            buf,
            size,
            cursor,
            out_a_name,
            out_a_dim,
            out_a_dims_cells,
            out_a_required,
            out_a_type,
            out_a_offset,
            out_a_next,
        )
        err_b = explicit_checked_composition(
            buf,
            size,
            cursor,
            out_b_name,
            out_b_dim,
            out_b_dims_cells,
            out_b_required,
            out_b_type,
            out_b_offset,
            out_b_next,
        )

        assert err_a == err_b
        assert out_a_name == out_b_name
        assert out_a_dim == out_b_dim
        assert out_a_dims_cells == out_b_dims_cells
        assert out_a_required == out_b_required
        assert out_a_type == out_b_type
        assert out_a_offset == out_b_offset
        assert out_a_next == out_b_next


def test_adversarial_subcall_output_invariants_are_enforced() -> None:
    buf = tensor_entry("x", [1], min(KNOWN_TYPES), 0)

    original_parity = globals()[
        "parse_one_checked_nopartial_commit_only_preflight_only_parity"
    ]
    original_commit = globals()["parse_one_checked_nopartial_commit_only"]

    out_name_len = [0]
    out_dim_count = [0]
    out_dims_cells = [0]
    out_required_bytes = [0]
    out_type_value = [0]
    out_tensor_offset = [0]
    out_next_cursor = [0]

    def parity_stub(*_args):
        _args[3][0] = 1
        _args[4][0] = 1
        _args[5][0] = 1
        _args[6][0] = len(buf)
        _args[7][0] = min(KNOWN_TYPES)
        _args[8][0] = 0
        _args[9][0] = len(buf)
        return GGUF_TENSOR_PARSE_OK

    def commit_stub_bad_dims(*_args):
        _args[3][0] = 1
        _args[4][0] = 1
        _args[5][0] = 2
        _args[6][0] = len(buf)
        _args[7][0] = min(KNOWN_TYPES)
        _args[8][0] = 0
        _args[9][0] = len(buf)
        return GGUF_TENSOR_PARSE_OK

    try:
        globals()["parse_one_checked_nopartial_commit_only_preflight_only_parity"] = (
            parity_stub
        )
        globals()["parse_one_checked_nopartial_commit_only"] = commit_stub_bad_dims
        status = parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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
        assert status == GGUF_TENSOR_PARSE_ERR_TRUNCATED

        def commit_stub_bad_type(*_args):
            _args[3][0] = 1
            _args[4][0] = 1
            _args[5][0] = 1
            _args[6][0] = len(buf)
            _args[7][0] = max(KNOWN_TYPES) + 999
            _args[8][0] = 0
            _args[9][0] = len(buf)
            return GGUF_TENSOR_PARSE_OK

        globals()["parse_one_checked_nopartial_commit_only"] = commit_stub_bad_type
        status = parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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
        assert status == GGUF_TENSOR_PARSE_ERR_TRUNCATED

        def commit_stub_bad_span(*_args):
            _args[3][0] = 1
            _args[4][0] = 1
            _args[5][0] = 1
            _args[6][0] = len(buf) + 1
            _args[7][0] = min(KNOWN_TYPES)
            _args[8][0] = 0
            _args[9][0] = len(buf)
            return GGUF_TENSOR_PARSE_OK

        globals()["parse_one_checked_nopartial_commit_only"] = commit_stub_bad_span
        status = parse_one_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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
        assert status == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    finally:
        globals()["parse_one_checked_nopartial_commit_only_preflight_only_parity"] = (
            original_parity
        )
        globals()["parse_one_checked_nopartial_commit_only"] = original_commit


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
