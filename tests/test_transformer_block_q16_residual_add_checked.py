#!/usr/bin/env python3
"""Parity harness for TransformerBlockQ16ResidualAddChecked."""

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


def transformer_block_q16_residual_add_checked(
    base_q16,
    base_capacity: int,
    base_stride_q16: int,
    residual_q16,
    residual_capacity: int,
    residual_stride_q16: int,
    out_q16,
    out_capacity: int,
    out_stride_q16: int,
    lane_count: int,
) -> int:
    if base_q16 is None or residual_q16 is None or out_q16 is None:
        return BLOCK_Q16_ERR_NULL_PTR

    if base_capacity < 0 or residual_capacity < 0 or out_capacity < 0:
        return BLOCK_Q16_ERR_BAD_PARAM
    if base_stride_q16 < 0 or residual_stride_q16 < 0 or out_stride_q16 < 0:
        return BLOCK_Q16_ERR_BAD_PARAM
    if lane_count < 0:
        return BLOCK_Q16_ERR_BAD_PARAM

    if lane_count == 0:
        return BLOCK_Q16_OK

    if base_stride_q16 < 1 or residual_stride_q16 < 1 or out_stride_q16 < 1:
        return BLOCK_Q16_ERR_BAD_PARAM

    required_base_cells = (lane_count - 1) * base_stride_q16 + 1
    required_residual_cells = (lane_count - 1) * residual_stride_q16 + 1
    required_out_cells = (lane_count - 1) * out_stride_q16 + 1

    if required_base_cells > base_capacity:
        return BLOCK_Q16_ERR_BAD_PARAM
    if required_residual_cells > residual_capacity:
        return BLOCK_Q16_ERR_BAD_PARAM
    if required_out_cells > out_capacity:
        return BLOCK_Q16_ERR_BAD_PARAM

    for lane_index in range(lane_count):
        base_index = lane_index * base_stride_q16
        residual_index = lane_index * residual_stride_q16
        err, _ = try_add_i64_checked(base_q16[base_index], residual_q16[residual_index])
        if err != BLOCK_Q16_OK:
            return err

    for lane_index in range(lane_count):
        base_index = lane_index * base_stride_q16
        residual_index = lane_index * residual_stride_q16
        out_index = lane_index * out_stride_q16
        _, lane_sum_q16 = try_add_i64_checked(base_q16[base_index], residual_q16[residual_index])
        out_q16[out_index] = lane_sum_q16

    return BLOCK_Q16_OK


def explicit_checked_composition(
    base_q16,
    base_capacity: int,
    base_stride_q16: int,
    residual_q16,
    residual_capacity: int,
    residual_stride_q16: int,
    out_q16,
    out_capacity: int,
    out_stride_q16: int,
    lane_count: int,
) -> int:
    if base_q16 is None or residual_q16 is None or out_q16 is None:
        return BLOCK_Q16_ERR_NULL_PTR

    if base_capacity < 0 or residual_capacity < 0 or out_capacity < 0:
        return BLOCK_Q16_ERR_BAD_PARAM
    if base_stride_q16 < 0 or residual_stride_q16 < 0 or out_stride_q16 < 0:
        return BLOCK_Q16_ERR_BAD_PARAM
    if lane_count < 0:
        return BLOCK_Q16_ERR_BAD_PARAM

    if lane_count == 0:
        return BLOCK_Q16_OK

    if base_stride_q16 < 1 or residual_stride_q16 < 1 or out_stride_q16 < 1:
        return BLOCK_Q16_ERR_BAD_PARAM

    req_base = (lane_count - 1) * base_stride_q16 + 1
    req_residual = (lane_count - 1) * residual_stride_q16 + 1
    req_out = (lane_count - 1) * out_stride_q16 + 1

    if req_base > base_capacity or req_residual > residual_capacity or req_out > out_capacity:
        return BLOCK_Q16_ERR_BAD_PARAM

    staged = [0] * lane_count
    for lane_index in range(lane_count):
        base_index = lane_index * base_stride_q16
        residual_index = lane_index * residual_stride_q16
        err, lane_sum = try_add_i64_checked(base_q16[base_index], residual_q16[residual_index])
        if err != BLOCK_Q16_OK:
            return err
        staged[lane_index] = lane_sum

    for lane_index in range(lane_count):
        out_q16[lane_index * out_stride_q16] = staged[lane_index]

    return BLOCK_Q16_OK


def test_source_contains_residual_add_signature_and_preflight_loop() -> None:
    source = Path("src/model/block.HC").read_text(encoding="utf-8")
    signature = "I32 TransformerBlockQ16ResidualAddChecked("
    assert signature in source

    body = source.split(signature, 1)[1]
    assert "if (!base_q16 || !residual_q16 || !out_q16)" in body
    assert "Preflight overflow for every lane before touching out buffer" in body
    assert "for (lane_index = 0; lane_index < lane_count; lane_index++)" in body


def test_known_vector_strided_output_matches_composition() -> None:
    lane_count = 6
    base_stride = 2
    residual_stride = 3
    out_stride = 2

    base = [0] * ((lane_count - 1) * base_stride + 1)
    residual = [0] * ((lane_count - 1) * residual_stride + 1)
    for lane in range(lane_count):
        base[lane * base_stride] = (lane - 11) << 12
        residual[lane * residual_stride] = (17 - lane * 2) << 11

    out_a = [0x7171] * ((lane_count - 1) * out_stride + 1)
    out_b = out_a.copy()

    err_a = transformer_block_q16_residual_add_checked(
        base,
        len(base),
        base_stride,
        residual,
        len(residual),
        residual_stride,
        out_a,
        len(out_a),
        out_stride,
        lane_count,
    )
    err_b = explicit_checked_composition(
        base,
        len(base),
        base_stride,
        residual,
        len(residual),
        residual_stride,
        out_b,
        len(out_b),
        out_stride,
        lane_count,
    )

    assert err_a == err_b == BLOCK_Q16_OK
    assert out_a == out_b


def test_error_paths_and_overflow_keep_out_unchanged() -> None:
    out = [0x1234] * 8
    before = out.copy()

    err = transformer_block_q16_residual_add_checked(
        None,
        0,
        1,
        [0],
        1,
        1,
        out,
        len(out),
        1,
        1,
    )
    assert err == BLOCK_Q16_ERR_NULL_PTR
    assert out == before

    err = transformer_block_q16_residual_add_checked(
        [0],
        -1,
        1,
        [0],
        1,
        1,
        out,
        len(out),
        1,
        1,
    )
    assert err == BLOCK_Q16_ERR_BAD_PARAM
    assert out == before

    base = [I64_MAX, 0, 10]
    residual = [1, 0, 10]
    err = transformer_block_q16_residual_add_checked(
        base,
        len(base),
        1,
        residual,
        len(residual),
        1,
        out,
        len(out),
        1,
        3,
    )
    assert err == BLOCK_Q16_ERR_OVERFLOW
    assert out == before


def test_randomized_parity_vs_explicit_checked_composition() -> None:
    random.seed(741)

    for _ in range(250):
        lane_count = random.randint(0, 16)
        base_stride = random.randint(1, 4)
        residual_stride = random.randint(1, 4)
        out_stride = random.randint(1, 4)

        req_base = 0 if lane_count == 0 else (lane_count - 1) * base_stride + 1
        req_residual = 0 if lane_count == 0 else (lane_count - 1) * residual_stride + 1
        req_out = 0 if lane_count == 0 else (lane_count - 1) * out_stride + 1

        base_capacity = req_base + random.randint(0, 2)
        residual_capacity = req_residual + random.randint(0, 2)
        out_capacity = req_out + random.randint(0, 2)

        base = [random.randint(-(1 << 20), (1 << 20)) for _ in range(max(1, base_capacity))]
        residual = [random.randint(-(1 << 20), (1 << 20)) for _ in range(max(1, residual_capacity))]

        if lane_count and random.randint(0, 7) == 0:
            idx = random.randint(0, lane_count - 1)
            base_idx = idx * base_stride
            residual_idx = idx * residual_stride
            if base_idx < len(base) and residual_idx < len(residual):
                base[base_idx] = I64_MAX
                residual[residual_idx] = 1

        out_a = [0x5A5A] * max(1, out_capacity)
        out_b = out_a.copy()

        err_a = transformer_block_q16_residual_add_checked(
            base,
            base_capacity,
            base_stride,
            residual,
            residual_capacity,
            residual_stride,
            out_a,
            out_capacity,
            out_stride,
            lane_count,
        )
        err_b = explicit_checked_composition(
            base,
            base_capacity,
            base_stride,
            residual,
            residual_capacity,
            residual_stride,
            out_b,
            out_capacity,
            out_stride,
            lane_count,
        )

        assert err_a == err_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_residual_add_signature_and_preflight_loop()
    test_known_vector_strided_output_matches_composition()
    test_error_paths_and_overflow_keep_out_unchanged()
    test_randomized_parity_vs_explicit_checked_composition()
    print("ok")
