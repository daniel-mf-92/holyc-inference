#!/usr/bin/env python3
"""Parity harness for IQ-1147 Q8_0DotRowsQ16CheckedNoPartialCommitOnlyPreflightOnly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_q8_0_dot import (
    Q8_0_ERR_BAD_DST_LEN,
    Q8_0_ERR_NULL_PTR,
    Q8_0_ERR_OVERFLOW,
    Q8_0_OK,
    Q8_0_VALUES_PER_BLOCK,
    half_bits,
    pack_signed,
)
from test_q8_0_dot_rows_q16_checked_nopartial import (
    q8_0_dot_rows_q16_checked_nopartial,
)

Q8_0_I64_MAX = 0x7FFFFFFFFFFFFFFF


def q8_0_dot_rows_q16_checked_nopartial_commit_only(
    lhs_blocks: list[tuple[int, bytes]] | None,
    lhs_block_capacity: int,
    left_stride_blocks: int,
    rhs_blocks: list[tuple[int, bytes]] | None,
    rhs_block_capacity: int,
    right_stride_blocks: int,
    rows: int,
    blocks_per_row: int,
    out_rows_q16: list[int] | None,
    out_stride: int,
    out_capacity: int,
) -> tuple[int, int, int]:
    if lhs_blocks is None or rhs_blocks is None or out_rows_q16 is None:
        return Q8_0_ERR_NULL_PTR, 0, 0

    if lhs_block_capacity < 0 or rhs_block_capacity < 0 or out_capacity < 0:
        return Q8_0_ERR_BAD_DST_LEN, 0, 0
    if left_stride_blocks < 0 or right_stride_blocks < 0 or out_stride < 0:
        return Q8_0_ERR_BAD_DST_LEN, 0, 0
    if rows < 0 or blocks_per_row < 0:
        return Q8_0_ERR_BAD_DST_LEN, 0, 0

    if rows == 0:
        return Q8_0_OK, 0, 0

    if out_stride == 0:
        return Q8_0_ERR_BAD_DST_LEN, 0, 0

    last_row = rows - 1

    out_last_index = last_row * out_stride
    if out_last_index > Q8_0_I64_MAX:
        return Q8_0_ERR_OVERFLOW, 0, 0

    required_out_cells = out_last_index + 1
    if required_out_cells > Q8_0_I64_MAX:
        return Q8_0_ERR_OVERFLOW, 0, 0
    if required_out_cells > out_capacity:
        return Q8_0_ERR_BAD_DST_LEN, 0, 0

    if blocks_per_row == 0:
        for row_idx in range(rows):
            out_rows_q16[row_idx * out_stride] = 0
        return Q8_0_OK, 0, 0

    left_last_row_base = last_row * left_stride_blocks
    right_last_row_base = last_row * right_stride_blocks

    required_left_blocks = left_last_row_base + blocks_per_row
    required_right_blocks = right_last_row_base + blocks_per_row

    if required_left_blocks > Q8_0_I64_MAX or required_right_blocks > Q8_0_I64_MAX:
        return Q8_0_ERR_OVERFLOW, 0, 0

    if required_left_blocks > lhs_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN, 0, 0
    if required_right_blocks > rhs_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN, 0, 0

    stage_rows = [0] * rows
    err = q8_0_dot_rows_q16_checked_nopartial(
        lhs_blocks,
        lhs_block_capacity,
        left_stride_blocks,
        rhs_blocks,
        rhs_block_capacity,
        right_stride_blocks,
        rows,
        blocks_per_row,
        stage_rows,
        rows,
    )
    if err != Q8_0_OK:
        return err, 0, 0

    for row_idx in range(rows):
        out_rows_q16[row_idx * out_stride] = stage_rows[row_idx]

    return Q8_0_OK, required_left_blocks, required_right_blocks


def q8_0_dot_rows_q16_checked_nopartial_commit_only_preflight_only(
    lhs_blocks: list[tuple[int, bytes]] | None,
    lhs_block_capacity: int,
    left_stride_blocks: int,
    rhs_blocks: list[tuple[int, bytes]] | None,
    rhs_block_capacity: int,
    right_stride_blocks: int,
    rows: int,
    blocks_per_row: int,
    out_rows_q16: list[int] | None,
    out_stride: int,
    out_capacity: int,
) -> tuple[int, int, int]:
    if lhs_blocks is None or rhs_blocks is None or out_rows_q16 is None:
        return Q8_0_ERR_NULL_PTR, 0, 0

    if rows < 0:
        return Q8_0_ERR_BAD_DST_LEN, 0, 0
    if rows == 0:
        return Q8_0_OK, 0, 0

    if out_stride <= 0:
        return Q8_0_ERR_BAD_DST_LEN, 0, 0

    last_row = rows - 1
    out_last_index = last_row * out_stride
    if out_last_index > Q8_0_I64_MAX:
        return Q8_0_ERR_OVERFLOW, 0, 0

    required_out_cells = out_last_index + 1
    if required_out_cells > Q8_0_I64_MAX:
        return Q8_0_ERR_OVERFLOW, 0, 0
    if required_out_cells > out_capacity:
        return Q8_0_ERR_BAD_DST_LEN, 0, 0

    stage_commit = [0] * required_out_cells
    stage_canonical = [0] * rows

    err, staged_left_blocks, staged_right_blocks = q8_0_dot_rows_q16_checked_nopartial_commit_only(
        lhs_blocks,
        lhs_block_capacity,
        left_stride_blocks,
        rhs_blocks,
        rhs_block_capacity,
        right_stride_blocks,
        rows,
        blocks_per_row,
        stage_commit,
        out_stride,
        required_out_cells,
    )
    if err != Q8_0_OK:
        return err, 0, 0

    err = q8_0_dot_rows_q16_checked_nopartial(
        lhs_blocks,
        lhs_block_capacity,
        left_stride_blocks,
        rhs_blocks,
        rhs_block_capacity,
        right_stride_blocks,
        rows,
        blocks_per_row,
        stage_canonical,
        rows,
    )
    if err != Q8_0_OK:
        return err, 0, 0

    if blocks_per_row == 0:
        canonical_left_blocks = 0
        canonical_right_blocks = 0
    else:
        canonical_left_blocks = (rows - 1) * left_stride_blocks + blocks_per_row
        canonical_right_blocks = (rows - 1) * right_stride_blocks + blocks_per_row

    if (
        staged_left_blocks != canonical_left_blocks
        or staged_right_blocks != canonical_right_blocks
    ):
        return Q8_0_ERR_OVERFLOW, 0, 0

    for row_idx in range(rows):
        if stage_commit[row_idx * out_stride] != stage_canonical[row_idx]:
            return Q8_0_ERR_OVERFLOW, 0, 0

    return Q8_0_OK, staged_left_blocks, staged_right_blocks


def make_block(rng: random.Random) -> tuple[int, bytes]:
    return (
        half_bits(rng.uniform(-2.0, 2.0)),
        pack_signed([rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)]),
    )


def test_source_contains_iq1147_functions() -> None:
    source = Path("src/quant/q8_0_dot.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotRowsQ16CheckedNoPartialCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "status = Q8_0DotRowsQ16CheckedNoPartialCommitOnly(" in body
    assert "status = Q8_0DotRowsQ16CheckedNoPartial(" in body
    assert "if (staged_required_left_blocks != canonical_required_left_blocks ||" in body


def test_null_and_domain_guards() -> None:
    rng = random.Random(114701)
    lhs_blocks = [make_block(rng) for _ in range(12)]
    rhs_blocks = [make_block(rng) for _ in range(12)]
    out = [111] * 32

    err, _, _ = q8_0_dot_rows_q16_checked_nopartial_commit_only_preflight_only(
        None,
        len(lhs_blocks),
        3,
        rhs_blocks,
        len(rhs_blocks),
        3,
        3,
        2,
        out,
        2,
        len(out),
    )
    assert err == Q8_0_ERR_NULL_PTR

    err, _, _ = q8_0_dot_rows_q16_checked_nopartial_commit_only_preflight_only(
        lhs_blocks,
        len(lhs_blocks),
        3,
        rhs_blocks,
        len(rhs_blocks),
        3,
        3,
        2,
        out,
        0,
        len(out),
    )
    assert err == Q8_0_ERR_BAD_DST_LEN


def test_success_reports_tuple_and_no_output_writes() -> None:
    rng = random.Random(114702)
    rows = 5
    blocks_per_row = 3
    left_stride = 4
    right_stride = 5
    out_stride = 3

    lhs_capacity = rows * left_stride
    rhs_capacity = rows * right_stride

    lhs_blocks = [make_block(rng) for _ in range(lhs_capacity)]
    rhs_blocks = [make_block(rng) for _ in range(rhs_capacity)]

    out = [777777] * (rows * out_stride + 5)
    before = out[:]

    err, required_left, required_right = q8_0_dot_rows_q16_checked_nopartial_commit_only_preflight_only(
        lhs_blocks,
        lhs_capacity,
        left_stride,
        rhs_blocks,
        rhs_capacity,
        right_stride,
        rows,
        blocks_per_row,
        out,
        out_stride,
        len(out),
    )

    assert err == Q8_0_OK
    assert required_left == (rows - 1) * left_stride + blocks_per_row
    assert required_right == (rows - 1) * right_stride + blocks_per_row
    assert out == before


def test_strided_capacity_underflow_fails_without_writes() -> None:
    rng = random.Random(114703)
    rows = 4
    out_stride = 5

    lhs_blocks = [make_block(rng) for _ in range(24)]
    rhs_blocks = [make_block(rng) for _ in range(24)]

    out = [9090] * 18
    before = out[:]

    err, req_l, req_r = q8_0_dot_rows_q16_checked_nopartial_commit_only_preflight_only(
        lhs_blocks,
        len(lhs_blocks),
        4,
        rhs_blocks,
        len(rhs_blocks),
        4,
        rows,
        2,
        out,
        out_stride,
        15,
    )

    assert err == Q8_0_ERR_BAD_DST_LEN
    assert req_l == 0
    assert req_r == 0
    assert out == before


def test_overflow_geometry_guard() -> None:
    rng = random.Random(114704)
    lhs_blocks = [make_block(rng) for _ in range(4)]
    rhs_blocks = [make_block(rng) for _ in range(4)]
    out = [1, 2, 3]

    err, req_l, req_r = q8_0_dot_rows_q16_checked_nopartial_commit_only_preflight_only(
        lhs_blocks,
        len(lhs_blocks),
        1,
        rhs_blocks,
        len(rhs_blocks),
        1,
        2,
        1,
        out,
        Q8_0_I64_MAX,
        len(out),
    )

    assert err == Q8_0_ERR_OVERFLOW
    assert req_l == 0
    assert req_r == 0


def test_randomized_parity_strided_windows() -> None:
    rng = random.Random(114705)

    for _ in range(280):
        rows = rng.randint(1, 8)
        blocks_per_row = rng.randint(1, 4)
        left_stride = blocks_per_row + rng.randint(0, 4)
        right_stride = blocks_per_row + rng.randint(0, 4)
        out_stride = rng.randint(1, 4)

        lhs_capacity = rows * left_stride
        rhs_capacity = rows * right_stride

        lhs_blocks = [make_block(rng) for _ in range(lhs_capacity)]
        rhs_blocks = [make_block(rng) for _ in range(rhs_capacity)]

        out = [4444] * (rows * out_stride + 5)
        before = out[:]

        err, req_left, req_right = q8_0_dot_rows_q16_checked_nopartial_commit_only_preflight_only(
            lhs_blocks,
            lhs_capacity,
            left_stride,
            rhs_blocks,
            rhs_capacity,
            right_stride,
            rows,
            blocks_per_row,
            out,
            out_stride,
            len(out),
        )

        assert err == Q8_0_OK
        assert req_left == (rows - 1) * left_stride + blocks_per_row
        assert req_right == (rows - 1) * right_stride + blocks_per_row
        assert out == before


def run() -> None:
    test_source_contains_iq1147_functions()
    test_null_and_domain_guards()
    test_success_reports_tuple_and_no_output_writes()
    test_strided_capacity_underflow_fails_without_writes()
    test_overflow_geometry_guard()
    test_randomized_parity_strided_windows()
    print("q8_0_dot_rows_q16_checked_nopartial_commit_only_preflight_only_checks=ok")


if __name__ == "__main__":
    run()
