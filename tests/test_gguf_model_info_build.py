#!/usr/bin/env python3
"""Harness for GGUFModelInfoBuildCheckedNoPartial (WS2-05)."""

from __future__ import annotations

from pathlib import Path

GGUF_INFO_OK = 0
GGUF_INFO_ERR_NULL_PTR = 1
GGUF_INFO_ERR_BAD_PARAM = 2
GGUF_INFO_ERR_OVERFLOW = 3
GGUF_INFO_ERR_OUT_OF_BOUNDS = 4
GGUF_INFO_ERR_OVERLAP = 5

U64_MAX = (1 << 64) - 1


def _u64_add(lhs: int, rhs: int) -> int | None:
    if lhs < 0 or rhs < 0:
        return None
    if lhs > U64_MAX - rhs:
        return None
    return lhs + rhs


def gguf_model_info_build_ref(
    *,
    tensor_data_base: int,
    file_nbytes: int,
    tensor_offsets: list[int] | None,
    tensor_sizes: list[int] | None,
    out_row_capacity: int,
) -> tuple[int, list[tuple[int, int, int, int]] | None, int | None, int | None]:
    if tensor_offsets is None or tensor_sizes is None:
        return GGUF_INFO_ERR_NULL_PTR, None, None, None

    tensor_count = len(tensor_offsets)
    if len(tensor_sizes) != tensor_count:
        return GGUF_INFO_ERR_BAD_PARAM, None, None, None
    if out_row_capacity < tensor_count:
        return GGUF_INFO_ERR_BAD_PARAM, None, None, None
    if tensor_data_base > file_nbytes:
        return GGUF_INFO_ERR_BAD_PARAM, None, None, None

    prev_end = tensor_data_base
    total_payload = 0
    last_end = tensor_data_base

    for idx in range(tensor_count):
        rel_offset = tensor_offsets[idx]
        byte_count = tensor_sizes[idx]

        abs_start = _u64_add(tensor_data_base, rel_offset)
        if abs_start is None:
            return GGUF_INFO_ERR_OVERFLOW, None, None, None
        if abs_start < prev_end:
            return GGUF_INFO_ERR_OVERLAP, None, None, None

        abs_end = _u64_add(abs_start, byte_count)
        if abs_end is None:
            return GGUF_INFO_ERR_OVERFLOW, None, None, None
        if abs_end > file_nbytes:
            return GGUF_INFO_ERR_OUT_OF_BOUNDS, None, None, None

        next_total = _u64_add(total_payload, byte_count)
        if next_total is None:
            return GGUF_INFO_ERR_OVERFLOW, None, None, None

        total_payload = next_total
        prev_end = abs_end
        last_end = abs_end

    rows: list[tuple[int, int, int, int]] = []
    for idx in range(tensor_count):
        rel_offset = tensor_offsets[idx]
        byte_count = tensor_sizes[idx]
        abs_start = tensor_data_base + rel_offset
        abs_end = abs_start + byte_count
        rows.append((rel_offset, byte_count, abs_start, abs_end))

    return GGUF_INFO_OK, rows, total_payload, last_end


def test_source_contains_ws2_05_entrypoint_and_guards() -> None:
    source = Path("src/gguf/model_info.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFModelInfoBuildCheckedNoPartial("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "if (out_row_capacity < tensor_count)" in body
    assert "if (abs_start < prev_end)" in body
    assert "if (abs_end > file_nbytes)" in body
    assert "*out_total_payload = staged_total_payload;" in body


def test_success_rows_and_summary() -> None:
    rc, rows, total, last_end = gguf_model_info_build_ref(
        tensor_data_base=64,
        file_nbytes=220,
        tensor_offsets=[0, 18, 68],
        tensor_sizes=[18, 34, 40],
        out_row_capacity=3,
    )

    assert rc == GGUF_INFO_OK
    assert rows == [
        (0, 18, 64, 82),
        (18, 34, 82, 116),
        (68, 40, 132, 172),
    ]
    assert total == 92
    assert last_end == 172


def test_overlap_rejected() -> None:
    rc, rows, total, last_end = gguf_model_info_build_ref(
        tensor_data_base=32,
        file_nbytes=200,
        tensor_offsets=[0, 10],
        tensor_sizes=[18, 34],
        out_row_capacity=2,
    )

    assert rc == GGUF_INFO_ERR_OVERLAP
    assert rows is None and total is None and last_end is None


def test_bounds_and_capacity_guards() -> None:
    rc, _, _, _ = gguf_model_info_build_ref(
        tensor_data_base=100,
        file_nbytes=90,
        tensor_offsets=[0],
        tensor_sizes=[8],
        out_row_capacity=1,
    )
    assert rc == GGUF_INFO_ERR_BAD_PARAM

    rc, _, _, _ = gguf_model_info_build_ref(
        tensor_data_base=64,
        file_nbytes=96,
        tensor_offsets=[40],
        tensor_sizes=[2],
        out_row_capacity=1,
    )
    assert rc == GGUF_INFO_ERR_OUT_OF_BOUNDS

    rc, _, _, _ = gguf_model_info_build_ref(
        tensor_data_base=64,
        file_nbytes=200,
        tensor_offsets=[0, 18],
        tensor_sizes=[18, 34],
        out_row_capacity=1,
    )
    assert rc == GGUF_INFO_ERR_BAD_PARAM
