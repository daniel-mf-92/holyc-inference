#!/usr/bin/env python3
"""Parity harness for IQ-1141 Q8_0DotRowsQ16CheckedNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_q8_0_dot import (
    Q8_0_ERR_BAD_DST_LEN,
    Q8_0_ERR_NULL_PTR,
    Q8_0_ERR_OVERFLOW,
    Q8_0_I64_MAX,
    Q8_0_OK,
    Q8_0_VALUES_PER_BLOCK,
    dot_product_blocks_q16_accumulate_checked,
    half_bits,
    pack_signed,
)


def q8_0_dot_rows_q16_checked_nopartial(
    lhs_blocks: list[tuple[int, bytes]] | None,
    lhs_block_capacity: int,
    lhs_row_stride_blocks: int,
    rhs_blocks: list[tuple[int, bytes]] | None,
    rhs_block_capacity: int,
    rhs_row_stride_blocks: int,
    row_count: int,
    blocks_per_row: int,
    out_rows_q16: list[int] | None,
    out_row_capacity: int,
) -> int:
    if lhs_blocks is None or rhs_blocks is None or out_rows_q16 is None:
        return Q8_0_ERR_NULL_PTR

    if lhs_block_capacity < 0 or rhs_block_capacity < 0 or out_row_capacity < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if lhs_row_stride_blocks < 0 or rhs_row_stride_blocks < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if row_count < 0 or blocks_per_row < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if row_count > out_row_capacity:
        return Q8_0_ERR_BAD_DST_LEN

    if row_count == 0:
        return Q8_0_OK

    if blocks_per_row == 0:
        for row_idx in range(row_count):
            out_rows_q16[row_idx] = 0
        return Q8_0_OK

    last_row = row_count - 1
    lhs_last_row_base = last_row * lhs_row_stride_blocks
    rhs_last_row_base = last_row * rhs_row_stride_blocks

    lhs_required_blocks = lhs_last_row_base + blocks_per_row
    rhs_required_blocks = rhs_last_row_base + blocks_per_row

    if lhs_required_blocks > Q8_0_I64_MAX:
        return Q8_0_ERR_OVERFLOW
    if rhs_required_blocks > Q8_0_I64_MAX:
        return Q8_0_ERR_OVERFLOW

    if lhs_required_blocks > lhs_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN
    if rhs_required_blocks > rhs_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN

    staged_rows_q16 = [0] * row_count

    for row_idx in range(row_count):
        lhs_row_base = row_idx * lhs_row_stride_blocks
        rhs_row_base = row_idx * rhs_row_stride_blocks

        lhs_slice = lhs_blocks[lhs_row_base : lhs_row_base + blocks_per_row]
        rhs_slice = rhs_blocks[rhs_row_base : rhs_row_base + blocks_per_row]
        if len(lhs_slice) != blocks_per_row or len(rhs_slice) != blocks_per_row:
            return Q8_0_ERR_BAD_DST_LEN

        err, row_dot_q16 = dot_product_blocks_q16_accumulate_checked(lhs_slice, rhs_slice, 0)
        if err != Q8_0_OK:
            return err

        staged_rows_q16[row_idx] = row_dot_q16

    for row_idx in range(row_count):
        out_rows_q16[row_idx] = staged_rows_q16[row_idx]

    return Q8_0_OK


def make_block(rng: random.Random) -> tuple[int, bytes]:
    return (
        half_bits(rng.uniform(-2.0, 2.0)),
        pack_signed([rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)]),
    )


def explicit_expected_rows(
    lhs_blocks: list[tuple[int, bytes]],
    lhs_row_stride_blocks: int,
    rhs_blocks: list[tuple[int, bytes]],
    rhs_row_stride_blocks: int,
    row_count: int,
    blocks_per_row: int,
) -> tuple[int, list[int]]:
    out: list[int] = []
    for row_idx in range(row_count):
        lhs_row_base = row_idx * lhs_row_stride_blocks
        rhs_row_base = row_idx * rhs_row_stride_blocks

        lhs_slice = lhs_blocks[lhs_row_base : lhs_row_base + blocks_per_row]
        rhs_slice = rhs_blocks[rhs_row_base : rhs_row_base + blocks_per_row]
        if len(lhs_slice) != blocks_per_row or len(rhs_slice) != blocks_per_row:
            return Q8_0_ERR_BAD_DST_LEN, []

        err, row_q16 = dot_product_blocks_q16_accumulate_checked(lhs_slice, rhs_slice, 0)
        if err != Q8_0_OK:
            return err, []
        out.append(row_q16)

    return Q8_0_OK, out


def test_source_contains_iq1141_function() -> None:
    source = Path("src/quant/q8_0_dot.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotRowsQ16CheckedNoPartial("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "staged_rows_q16 = MAlloc(stage_bytes);" in body
    assert "status = Q8_0DotProductBlocksQ16AccumulateChecked(" in body
    assert "out_rows_q16[row_index] = staged_rows_q16[row_index];" in body


def test_null_and_domain_guards() -> None:
    rng = random.Random(114101)
    lhs_blocks = [make_block(rng) for _ in range(8)]
    rhs_blocks = [make_block(rng) for _ in range(8)]
    out = [7, 9]

    assert (
        q8_0_dot_rows_q16_checked_nopartial(
            None,
            len(lhs_blocks),
            2,
            rhs_blocks,
            len(rhs_blocks),
            2,
            2,
            2,
            out,
            len(out),
        )
        == Q8_0_ERR_NULL_PTR
    )

    assert (
        q8_0_dot_rows_q16_checked_nopartial(
            lhs_blocks,
            len(lhs_blocks),
            -1,
            rhs_blocks,
            len(rhs_blocks),
            2,
            2,
            2,
            out,
            len(out),
        )
        == Q8_0_ERR_BAD_DST_LEN
    )

    assert (
        q8_0_dot_rows_q16_checked_nopartial(
            lhs_blocks,
            len(lhs_blocks),
            2,
            rhs_blocks,
            len(rhs_blocks),
            2,
            3,
            2,
            out,
            len(out),
        )
        == Q8_0_ERR_BAD_DST_LEN
    )


def test_zero_block_rows_write_zero() -> None:
    rng = random.Random(114102)
    lhs_blocks = [make_block(rng) for _ in range(4)]
    rhs_blocks = [make_block(rng) for _ in range(4)]
    out = [111, 222, 333]

    err = q8_0_dot_rows_q16_checked_nopartial(
        lhs_blocks,
        len(lhs_blocks),
        2,
        rhs_blocks,
        len(rhs_blocks),
        2,
        2,
        0,
        out,
        len(out),
    )
    assert err == Q8_0_OK
    assert out[:2] == [0, 0]
    assert out[2] == 333


def test_preflight_failure_preserves_output() -> None:
    rng = random.Random(114103)
    lhs_blocks = [make_block(rng) for _ in range(10)]
    rhs_blocks = [make_block(rng) for _ in range(10)]

    out = [12345, -67890, 11111]
    before = out[:]

    err = q8_0_dot_rows_q16_checked_nopartial(
        lhs_blocks,
        len(lhs_blocks),
        4,
        rhs_blocks,
        len(rhs_blocks),
        4,
        3,
        3,
        out,
        len(out),
    )
    assert err == Q8_0_ERR_BAD_DST_LEN
    assert out == before


def test_random_strided_parity() -> None:
    rng = random.Random(114104)

    for _ in range(320):
        row_count = rng.randint(1, 8)
        blocks_per_row = rng.randint(1, 5)
        lhs_stride = blocks_per_row + rng.randint(0, 3)
        rhs_stride = blocks_per_row + rng.randint(0, 3)

        lhs_capacity = row_count * lhs_stride
        rhs_capacity = row_count * rhs_stride
        lhs_blocks = [make_block(rng) for _ in range(lhs_capacity)]
        rhs_blocks = [make_block(rng) for _ in range(rhs_capacity)]

        expected_err, expected_rows = explicit_expected_rows(
            lhs_blocks,
            lhs_stride,
            rhs_blocks,
            rhs_stride,
            row_count,
            blocks_per_row,
        )
        assert expected_err == Q8_0_OK

        out = [0] * row_count
        got_err = q8_0_dot_rows_q16_checked_nopartial(
            lhs_blocks,
            lhs_capacity,
            lhs_stride,
            rhs_blocks,
            rhs_capacity,
            rhs_stride,
            row_count,
            blocks_per_row,
            out,
            row_count,
        )
        assert got_err == Q8_0_OK
        assert out == expected_rows


def test_reports_accumulator_overflow_and_no_partial() -> None:
    # Extreme blocks to force checked per-row accumulation overflow.
    max_pos = pack_signed([127] * Q8_0_VALUES_PER_BLOCK)
    max_neg = pack_signed([-128] * Q8_0_VALUES_PER_BLOCK)

    lhs_blocks = [
        (half_bits(65504.0), max_pos),
        (half_bits(65504.0), max_pos),
    ]
    rhs_blocks = [
        (half_bits(65504.0), max_neg),
        (half_bits(65504.0), max_neg),
    ]

    out = [777]
    err = q8_0_dot_rows_q16_checked_nopartial(
        lhs_blocks,
        len(lhs_blocks),
        2,
        rhs_blocks,
        len(rhs_blocks),
        2,
        1,
        2,
        out,
        len(out),
    )

    assert err == Q8_0_ERR_OVERFLOW
    assert out == [777]


def run() -> None:
    test_source_contains_iq1141_function()
    test_null_and_domain_guards()
    test_zero_block_rows_write_zero()
    test_preflight_failure_preserves_output()
    test_random_strided_parity()
    test_reports_accumulator_overflow_and_no_partial()
    print("q8_0_dot_rows_q16_checked_nopartial_checks=ok")


if __name__ == "__main__":
    run()
