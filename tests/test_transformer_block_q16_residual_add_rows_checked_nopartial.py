#!/usr/bin/env python3
"""Parity harness for TransformerBlockQ16ResidualAddRowsCheckedNoPartial."""

from __future__ import annotations

import random
from pathlib import Path

BLOCK_Q16_OK = 0
BLOCK_Q16_ERR_NULL_PTR = 1
BLOCK_Q16_ERR_BAD_PARAM = 2
BLOCK_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


def try_add_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if rhs > 0 and lhs > I64_MAX - rhs:
        return BLOCK_Q16_ERR_OVERFLOW, 0
    if rhs < 0 and lhs < I64_MIN - rhs:
        return BLOCK_Q16_ERR_OVERFLOW, 0
    return BLOCK_Q16_OK, lhs + rhs


def transformer_block_q16_residual_add_rows_checked_nopartial(
    base_q16,
    base_capacity: int,
    base_row_stride_q16: int,
    base_lane_stride_q16: int,
    residual_q16,
    residual_capacity: int,
    residual_row_stride_q16: int,
    residual_lane_stride_q16: int,
    out_q16,
    out_capacity: int,
    out_row_stride_q16: int,
    out_lane_stride_q16: int,
    row_count: int,
    lane_count: int,
) -> int:
    if base_q16 is None or residual_q16 is None or out_q16 is None:
        return BLOCK_Q16_ERR_NULL_PTR

    if base_capacity < 0 or residual_capacity < 0 or out_capacity < 0:
        return BLOCK_Q16_ERR_BAD_PARAM
    if base_row_stride_q16 < 0 or base_lane_stride_q16 < 0:
        return BLOCK_Q16_ERR_BAD_PARAM
    if residual_row_stride_q16 < 0 or residual_lane_stride_q16 < 0:
        return BLOCK_Q16_ERR_BAD_PARAM
    if out_row_stride_q16 < 0 or out_lane_stride_q16 < 0:
        return BLOCK_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return BLOCK_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        return BLOCK_Q16_OK

    if base_lane_stride_q16 < 1 or residual_lane_stride_q16 < 1 or out_lane_stride_q16 < 1:
        return BLOCK_Q16_ERR_BAD_PARAM
    if row_count > 1 and (
        base_row_stride_q16 < 1
        or residual_row_stride_q16 < 1
        or out_row_stride_q16 < 1
    ):
        return BLOCK_Q16_ERR_BAD_PARAM

    required_base_cells = (row_count - 1) * base_row_stride_q16 + (lane_count - 1) * base_lane_stride_q16 + 1
    required_residual_cells = (row_count - 1) * residual_row_stride_q16 + (lane_count - 1) * residual_lane_stride_q16 + 1
    required_out_cells = (row_count - 1) * out_row_stride_q16 + (lane_count - 1) * out_lane_stride_q16 + 1

    if required_base_cells > base_capacity:
        return BLOCK_Q16_ERR_BAD_PARAM
    if required_residual_cells > residual_capacity:
        return BLOCK_Q16_ERR_BAD_PARAM
    if required_out_cells > out_capacity:
        return BLOCK_Q16_ERR_BAD_PARAM

    for row_index in range(row_count):
        base_row_base = row_index * base_row_stride_q16
        residual_row_base = row_index * residual_row_stride_q16
        for lane_index in range(lane_count):
            base_index = base_row_base + lane_index * base_lane_stride_q16
            residual_index = residual_row_base + lane_index * residual_lane_stride_q16
            err, _ = try_add_i64_checked(base_q16[base_index], residual_q16[residual_index])
            if err != BLOCK_Q16_OK:
                return err

    for row_index in range(row_count):
        base_row_base = row_index * base_row_stride_q16
        residual_row_base = row_index * residual_row_stride_q16
        out_row_base = row_index * out_row_stride_q16
        for lane_index in range(lane_count):
            base_index = base_row_base + lane_index * base_lane_stride_q16
            residual_index = residual_row_base + lane_index * residual_lane_stride_q16
            out_index = out_row_base + lane_index * out_lane_stride_q16
            _, lane_sum_q16 = try_add_i64_checked(base_q16[base_index], residual_q16[residual_index])
            out_q16[out_index] = lane_sum_q16

    return BLOCK_Q16_OK


def explicit_checked_composition(*args, **kwargs) -> int:
    return transformer_block_q16_residual_add_rows_checked_nopartial(*args, **kwargs)


def test_source_contains_signature_and_no_partial_contract() -> None:
    source = Path("src/model/block.HC").read_text(encoding="utf-8")
    signature = "I32 TransformerBlockQ16ResidualAddRowsCheckedNoPartial("
    assert signature in source

    body = source.split(signature, 1)[1]
    assert "Full preflight over all rows/lanes before touching out." in body
    assert "Commit pass runs only after full preflight succeeds." in body


def test_known_vector_strided_2d_matches_composition() -> None:
    row_count = 4
    lane_count = 5

    base_row_stride = 17
    base_lane_stride = 3
    residual_row_stride = 19
    residual_lane_stride = 2
    out_row_stride = 23
    out_lane_stride = 4

    req_base = (row_count - 1) * base_row_stride + (lane_count - 1) * base_lane_stride + 1
    req_residual = (row_count - 1) * residual_row_stride + (lane_count - 1) * residual_lane_stride + 1
    req_out = (row_count - 1) * out_row_stride + (lane_count - 1) * out_lane_stride + 1

    base = [0] * req_base
    residual = [0] * req_residual

    for row in range(row_count):
        for lane in range(lane_count):
            base[row * base_row_stride + lane * base_lane_stride] = ((row * 31 - lane * 7) << 10)
            residual[row * residual_row_stride + lane * residual_lane_stride] = ((lane * 11 - row * 3) << 9)

    out_a = [0x5151] * req_out
    out_b = out_a.copy()

    err_a = transformer_block_q16_residual_add_rows_checked_nopartial(
        base,
        len(base),
        base_row_stride,
        base_lane_stride,
        residual,
        len(residual),
        residual_row_stride,
        residual_lane_stride,
        out_a,
        len(out_a),
        out_row_stride,
        out_lane_stride,
        row_count,
        lane_count,
    )
    err_b = explicit_checked_composition(
        base,
        len(base),
        base_row_stride,
        base_lane_stride,
        residual,
        len(residual),
        residual_row_stride,
        residual_lane_stride,
        out_b,
        len(out_b),
        out_row_stride,
        out_lane_stride,
        row_count,
        lane_count,
    )

    assert err_a == err_b == BLOCK_Q16_OK
    assert out_a == out_b


def test_error_and_overflow_paths_keep_out_unchanged() -> None:
    out = [0x1234] * 64
    before = out.copy()

    err = transformer_block_q16_residual_add_rows_checked_nopartial(
        None,
        0,
        0,
        1,
        [0],
        1,
        0,
        1,
        out,
        len(out),
        0,
        1,
        1,
        1,
    )
    assert err == BLOCK_Q16_ERR_NULL_PTR
    assert out == before

    err = transformer_block_q16_residual_add_rows_checked_nopartial(
        [0],
        -1,
        0,
        1,
        [0],
        1,
        0,
        1,
        out,
        len(out),
        0,
        1,
        1,
        1,
    )
    assert err == BLOCK_Q16_ERR_BAD_PARAM
    assert out == before

    base = [0] * 16
    residual = [0] * 16
    base[0] = I64_MAX
    residual[0] = 1

    err = transformer_block_q16_residual_add_rows_checked_nopartial(
        base,
        len(base),
        8,
        1,
        residual,
        len(residual),
        8,
        1,
        out,
        len(out),
        8,
        1,
        2,
        1,
    )
    assert err == BLOCK_Q16_ERR_OVERFLOW
    assert out == before


def test_randomized_parity_vs_composition() -> None:
    random.seed(742)

    for _ in range(250):
        row_count = random.randint(0, 9)
        lane_count = random.randint(0, 10)

        base_row_stride = random.randint(1, 6)
        base_lane_stride = random.randint(1, 5)
        residual_row_stride = random.randint(1, 6)
        residual_lane_stride = random.randint(1, 5)
        out_row_stride = random.randint(1, 6)
        out_lane_stride = random.randint(1, 5)

        req_base = 0 if row_count == 0 or lane_count == 0 else (row_count - 1) * base_row_stride + (lane_count - 1) * base_lane_stride + 1
        req_residual = 0 if row_count == 0 or lane_count == 0 else (row_count - 1) * residual_row_stride + (lane_count - 1) * residual_lane_stride + 1
        req_out = 0 if row_count == 0 or lane_count == 0 else (row_count - 1) * out_row_stride + (lane_count - 1) * out_lane_stride + 1

        base_capacity = req_base + random.randint(0, 3)
        residual_capacity = req_residual + random.randint(0, 3)
        out_capacity = req_out + random.randint(0, 3)

        base = [random.randint(-(1 << 20), (1 << 20)) for _ in range(max(1, base_capacity))]
        residual = [random.randint(-(1 << 20), (1 << 20)) for _ in range(max(1, residual_capacity))]

        if row_count and lane_count and random.randint(0, 8) == 0:
            row = random.randint(0, row_count - 1)
            lane = random.randint(0, lane_count - 1)
            base[row * base_row_stride + lane * base_lane_stride] = I64_MAX
            residual[row * residual_row_stride + lane * residual_lane_stride] = 1

        out_a = [0x7777] * max(1, out_capacity)
        out_b = out_a.copy()

        err_a = transformer_block_q16_residual_add_rows_checked_nopartial(
            base,
            base_capacity,
            base_row_stride,
            base_lane_stride,
            residual,
            residual_capacity,
            residual_row_stride,
            residual_lane_stride,
            out_a,
            out_capacity,
            out_row_stride,
            out_lane_stride,
            row_count,
            lane_count,
        )
        err_b = explicit_checked_composition(
            base,
            base_capacity,
            base_row_stride,
            base_lane_stride,
            residual,
            residual_capacity,
            residual_row_stride,
            residual_lane_stride,
            out_b,
            out_capacity,
            out_row_stride,
            out_lane_stride,
            row_count,
            lane_count,
        )

        assert err_a == err_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_signature_and_no_partial_contract()
    test_known_vector_strided_2d_matches_composition()
    test_error_and_overflow_paths_keep_out_unchanged()
    test_randomized_parity_vs_composition()
    print("ok")
