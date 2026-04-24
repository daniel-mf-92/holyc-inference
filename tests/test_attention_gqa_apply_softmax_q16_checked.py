#!/usr/bin/env python3
"""Reference checks for GQAAttentionApplySoftmaxQ16Checked semantics (IQ-1368)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_ERR_OVERFLOW,
    ATTN_Q16_OK,
)

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)
ATTN_Q16_SHIFT = 16
GQA_SOFTMAX_EXP_MAX_Q16 = 655360
GQA_SOFTMAX_EXP_MIN_Q16 = -655360
GQA_SOFTMAX_ONE_Q16 = 1 << ATTN_Q16_SHIFT


def try_add_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    value = lhs + rhs
    if value < I64_MIN or value > I64_MAX:
        return (ATTN_Q16_ERR_OVERFLOW, 0)
    return (ATTN_Q16_OK, value)


def try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    value = lhs * rhs
    if value < I64_MIN or value > I64_MAX:
        return (ATTN_Q16_ERR_OVERFLOW, 0)
    return (ATTN_Q16_OK, value)


def fpq16_exp_approx_q16(x_q16: int) -> int:
    # Integer-only deterministic stand-in for HolyC FPQ16Exp.
    x_q16 = max(GQA_SOFTMAX_EXP_MIN_Q16, min(GQA_SOFTMAX_EXP_MAX_Q16, x_q16))
    if x_q16 >= 0:
        return GQA_SOFTMAX_ONE_Q16 + (x_q16 // 2)

    neg = -x_q16
    den = GQA_SOFTMAX_ONE_Q16 + neg
    if den <= 0:
        return 1
    return max(1, (GQA_SOFTMAX_ONE_Q16 * GQA_SOFTMAX_ONE_Q16) // den)


def gqa_attention_apply_softmax_q16_checked(
    scores_q32,
    scores_capacity: int,
    query_rows: int,
    key_rows: int,
    head_groups: int,
    row_stride: int,
    out_probs_q16,
    out_capacity: int,
) -> int:
    if scores_q32 is None or out_probs_q16 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if scores_capacity < 0 or out_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_rows < 0 or key_rows < 0 or head_groups <= 0 or row_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_rows > 0 and (query_rows % head_groups) != 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if key_rows > row_stride:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_rows == 0 or key_rows == 0:
        return ATTN_Q16_OK

    err, required_score_cells = try_mul_i64_checked(query_rows - 1, row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_score_cells = try_add_i64_checked(required_score_cells, key_rows)
    if err != ATTN_Q16_OK:
        return err
    if required_score_cells > scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_out_cells = try_mul_i64_checked(query_rows, key_rows)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    staged = [0] * required_out_cells

    for row_index in range(query_rows):
        err, row_base = try_mul_i64_checked(row_index, row_stride)
        if err != ATTN_Q16_OK:
            return err

        row = scores_q32[row_base : row_base + key_rows]
        row_max_q32 = max(row)

        exp_sum_q16 = 0
        best_index = 0
        best_value = -1
        for col_index, score_q32 in enumerate(row):
            shifted_q16 = (score_q32 - row_max_q32) >> ATTN_Q16_SHIFT
            shifted_q16 = max(GQA_SOFTMAX_EXP_MIN_Q16, min(GQA_SOFTMAX_EXP_MAX_Q16, shifted_q16))

            exp_lane_q16 = fpq16_exp_approx_q16(shifted_q16)
            if exp_lane_q16 <= 0:
                return ATTN_Q16_ERR_BAD_PARAM

            staged[row_index * key_rows + col_index] = exp_lane_q16
            err, exp_sum_q16 = try_add_i64_checked(exp_sum_q16, exp_lane_q16)
            if err != ATTN_Q16_OK:
                return err

            if exp_lane_q16 > best_value:
                best_value = exp_lane_q16
                best_index = col_index

        if exp_sum_q16 <= 0:
            return ATTN_Q16_ERR_BAD_PARAM

        prob_sum_q16 = 0
        for col_index in range(key_rows):
            exp_lane_q16 = staged[row_index * key_rows + col_index]
            err, numerator_q16 = try_mul_i64_checked(exp_lane_q16, GQA_SOFTMAX_ONE_Q16)
            if err != ATTN_Q16_OK:
                return err
            err, numerator_q16 = try_add_i64_checked(numerator_q16, exp_sum_q16 // 2)
            if err != ATTN_Q16_OK:
                return err

            prob_q16 = numerator_q16 // exp_sum_q16
            if prob_q16 < 0:
                prob_q16 = 0

            staged[row_index * key_rows + col_index] = prob_q16
            err, prob_sum_q16 = try_add_i64_checked(prob_sum_q16, prob_q16)
            if err != ATTN_Q16_OK:
                return err

        remainder_q16 = GQA_SOFTMAX_ONE_Q16 - prob_sum_q16
        if remainder_q16 > 0:
            idx = row_index * key_rows + best_index
            err, staged[idx] = try_add_i64_checked(staged[idx], remainder_q16)
            if err != ATTN_Q16_OK:
                return err
        elif remainder_q16 < 0:
            remainder_q16 = -remainder_q16
            for offset in range(key_rows):
                if remainder_q16 == 0:
                    break
                idx = row_index * key_rows + ((best_index + offset) % key_rows)
                if staged[idx] == 0:
                    continue
                if staged[idx] > remainder_q16:
                    staged[idx] -= remainder_q16
                    remainder_q16 = 0
                else:
                    remainder_q16 -= staged[idx]
                    staged[idx] = 0

            if remainder_q16 != 0:
                return ATTN_Q16_ERR_BAD_PARAM

    for i in range(required_out_cells):
        out_probs_q16[i] = staged[i]

    return ATTN_Q16_OK


def explicit_softmax_composition(*args, **kwargs) -> int:
    return gqa_attention_apply_softmax_q16_checked(*args, **kwargs)


def test_fixed_vector_reference() -> None:
    query_rows = 4
    key_rows = 3
    head_groups = 2
    row_stride = 3

    scores_q32 = [
        9 << 16,
        7 << 16,
        4 << 16,
        12 << 16,
        6 << 16,
        1 << 16,
        2 << 16,
        2 << 16,
        2 << 16,
        5 << 16,
        6 << 16,
        7 << 16,
    ]

    out_a = [0] * (query_rows * key_rows)
    out_b = [0] * (query_rows * key_rows)

    err_a = gqa_attention_apply_softmax_q16_checked(
        scores_q32,
        len(scores_q32),
        query_rows,
        key_rows,
        head_groups,
        row_stride,
        out_a,
        len(out_a),
    )
    err_b = explicit_softmax_composition(
        scores_q32,
        len(scores_q32),
        query_rows,
        key_rows,
        head_groups,
        row_stride,
        out_b,
        len(out_b),
    )

    assert err_a == err_b == ATTN_Q16_OK
    assert out_a == out_b
    for row in range(query_rows):
        row_sum = sum(out_a[row * key_rows : (row + 1) * key_rows])
        assert row_sum == GQA_SOFTMAX_ONE_Q16


def test_error_contract_and_no_partial() -> None:
    seed = [3333] * 8
    out = seed.copy()

    err = gqa_attention_apply_softmax_q16_checked(
        [1 << 16, 2 << 16, 3 << 16],
        3,
        1,
        2,
        2,
        2,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == seed

    out = seed.copy()
    err = gqa_attention_apply_softmax_q16_checked(
        [1 << 16],
        1,
        1,
        2,
        1,
        1,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == seed

    out = seed.copy()
    err = gqa_attention_apply_softmax_q16_checked(
        None,
        0,
        0,
        0,
        1,
        0,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert out == seed


def test_randomized_row_sum_and_parity() -> None:
    rng = random.Random(20260425_1368)

    for _ in range(180):
        query_rows = rng.randint(1, 6)
        head_groups = rng.randint(1, 3)
        query_rows = query_rows + ((head_groups - (query_rows % head_groups)) % head_groups)
        key_rows = rng.randint(1, 6)
        row_stride = key_rows + rng.randint(0, 2)

        scores = [rng.randint(-(40 << 16), (40 << 16)) for _ in range(query_rows * row_stride)]
        out_capacity = query_rows * key_rows
        out_a = [rng.randint(0, 99) for _ in range(out_capacity)]
        out_b = out_a.copy()

        err_a = gqa_attention_apply_softmax_q16_checked(
            scores,
            len(scores),
            query_rows,
            key_rows,
            head_groups,
            row_stride,
            out_a,
            out_capacity,
        )
        err_b = explicit_softmax_composition(
            scores,
            len(scores),
            query_rows,
            key_rows,
            head_groups,
            row_stride,
            out_b,
            out_capacity,
        )

        assert err_a == err_b == ATTN_Q16_OK
        assert out_a == out_b

        for row in range(query_rows):
            row_vals = out_a[row * key_rows : (row + 1) * key_rows]
            assert all(v >= 0 for v in row_vals)
            assert sum(row_vals) == GQA_SOFTMAX_ONE_Q16


def test_source_contains_symbol_and_core_ops() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    sig = "I32 GQAAttentionApplySoftmaxQ16Checked("
    assert sig in source

    body = source.split(sig, 1)[1]
    assert "row_max_q32" in body
    assert "shifted_q16" in body
    assert "FPQ16Exp(shifted_q16)" in body


if __name__ == "__main__":
    test_fixed_vector_reference()
    test_error_contract_and_no_partial()
    test_randomized_row_sum_and_parity()
    test_source_contains_symbol_and_core_ops()
    print("attention_gqa_apply_softmax_q16_checked=ok")
