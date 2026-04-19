#!/usr/bin/env python3
"""Parity harness for AttentionQ16ApplyScoreScaleCheckedDefaultStride."""

from __future__ import annotations

import random
from pathlib import Path

ATTN_Q16_OK = 0
ATTN_Q16_ERR_NULL_PTR = 1
ATTN_Q16_ERR_BAD_PARAM = 2
ATTN_Q16_ERR_OVERFLOW = 3

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
U64_MAX = (1 << 64) - 1
Q16_SHIFT = 16


def abs_to_u64(x: int) -> int:
    if x >= 0:
        return x
    return (-(x + 1)) + 1


def apply_sign_from_mag_checked(mag: int, is_negative: bool) -> tuple[int, int]:
    if is_negative:
        if mag > (1 << 63):
            return ATTN_Q16_ERR_OVERFLOW, 0
        if mag == (1 << 63):
            return ATTN_Q16_OK, I64_MIN
        return ATTN_Q16_OK, -mag

    if mag > I64_MAX:
        return ATTN_Q16_ERR_OVERFLOW, 0
    return ATTN_Q16_OK, mag


def try_add_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if rhs > 0 and lhs > I64_MAX - rhs:
        return ATTN_Q16_ERR_OVERFLOW, 0
    if rhs < 0 and lhs < I64_MIN - rhs:
        return ATTN_Q16_ERR_OVERFLOW, 0
    out = lhs + rhs
    if out < I64_MIN or out > I64_MAX:
        return ATTN_Q16_ERR_OVERFLOW, 0
    return ATTN_Q16_OK, out


def try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    abs_lhs = abs_to_u64(lhs)
    abs_rhs = abs_to_u64(rhs)
    if abs_lhs and abs_rhs and abs_lhs > (U64_MAX // abs_rhs):
        return ATTN_Q16_ERR_OVERFLOW, 0
    mag = abs_lhs * abs_rhs
    return apply_sign_from_mag_checked(mag, (lhs < 0) ^ (rhs < 0))


def q32_scale_by_q16_checked(value_q32: int, scale_q16: int) -> tuple[int, int]:
    abs_value = abs_to_u64(value_q32)
    abs_scale = abs_to_u64(scale_q16)
    if abs_value and abs_scale and abs_value > (U64_MAX // abs_scale):
        return ATTN_Q16_ERR_OVERFLOW, 0

    abs_prod = abs_value * abs_scale
    round_bias = 1 << (Q16_SHIFT - 1)
    if abs_prod > U64_MAX - round_bias:
        rounded = U64_MAX >> Q16_SHIFT
    else:
        rounded = (abs_prod + round_bias) >> Q16_SHIFT

    return apply_sign_from_mag_checked(rounded, (value_q32 < 0) ^ (scale_q16 < 0))


def attention_q16_apply_score_scale_checked(
    in_scores_q32,
    in_scores_capacity: int,
    token_count: int,
    in_score_stride: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_score_stride: int,
) -> int:
    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if token_count < 0 or in_score_stride < 0 or out_score_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if token_count == 0:
        return ATTN_Q16_OK

    if in_score_stride < 1 or out_score_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_in_cells = try_mul_i64_checked(token_count - 1, in_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_in_cells = try_add_i64_checked(required_in_cells, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_in_cells > in_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_out_cells = try_mul_i64_checked(token_count - 1, out_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    for token_index in range(token_count):
        err, in_base = try_mul_i64_checked(token_index, in_score_stride)
        if err != ATTN_Q16_OK:
            return err
        err, out_base = try_mul_i64_checked(token_index, out_score_stride)
        if err != ATTN_Q16_OK:
            return err
        err, scaled = q32_scale_by_q16_checked(in_scores_q32[in_base], score_scale_q16)
        if err != ATTN_Q16_OK:
            return err
        out_scores_q32[out_base] = scaled

    return ATTN_Q16_OK


def attention_q16_apply_score_scale_checked_default_stride(
    in_scores_q32,
    in_scores_capacity: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR
    if in_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    default_stride = token_count
    return attention_q16_apply_score_scale_checked(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        default_stride,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_stride,
    )


def explicit_default_stride_composition(
    in_scores_q32,
    in_scores_capacity: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    stride = token_count
    return attention_q16_apply_score_scale_checked(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        stride,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        stride,
    )


def test_source_contains_default_stride_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 AttentionQ16ApplyScoreScaleCheckedDefaultStride(" in source
    body = source.split("I32 AttentionQ16ApplyScoreScaleCheckedDefaultStride(", 1)[1]
    assert "default_score_stride = token_count;" in body
    assert "return AttentionQ16ApplyScoreScaleChecked(" in body


def test_known_vectors_match_explicit_default_composition() -> None:
    token_count = 4
    scale_q16 = 23170

    stride = token_count
    in_capacity = 1 + (token_count - 1) * stride
    out_capacity = 1 + (token_count - 1) * stride

    in_scores = [0] * in_capacity
    seeds = [333333333, -222222222, 111111111, -98765432]
    for i, value in enumerate(seeds):
        in_scores[i * stride] = value

    out_new = [777] * out_capacity
    out_ref = out_new.copy()

    err_new = attention_q16_apply_score_scale_checked_default_stride(
        in_scores,
        in_capacity,
        token_count,
        scale_q16,
        out_new,
        out_capacity,
    )
    err_ref = explicit_default_stride_composition(
        in_scores,
        in_capacity,
        token_count,
        scale_q16,
        out_ref,
        out_capacity,
    )

    assert err_new == err_ref == ATTN_Q16_OK
    assert out_new == out_ref


def test_adversarial_contracts() -> None:
    assert (
        attention_q16_apply_score_scale_checked_default_stride(None, 0, 0, 1 << 16, [0], 1)
        == ATTN_Q16_ERR_NULL_PTR
    )
    assert (
        attention_q16_apply_score_scale_checked_default_stride([0], 1, 0, 1 << 16, None, 0)
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_apply_score_scale_checked_default_stride([0], -1, 0, 1 << 16, [0], 1)
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_checked_default_stride([0], 1, -1, 1 << 16, [0], 1)
        == ATTN_Q16_ERR_BAD_PARAM
    )

    token_count = 3
    stride = token_count
    in_scores = [11, 0, 0, -22, 0, 0, 33]
    out_scores = [0] * 7

    # Bad source capacity for required default-stride span.
    assert (
        attention_q16_apply_score_scale_checked_default_stride(
            in_scores,
            6,
            token_count,
            1 << 16,
            out_scores,
            len(out_scores),
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    # Bad destination capacity for required default-stride span.
    assert (
        attention_q16_apply_score_scale_checked_default_stride(
            in_scores,
            len(in_scores),
            token_count,
            1 << 16,
            out_scores,
            6,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    # Scaling overflow parity from checked core.
    assert (
        attention_q16_apply_score_scale_checked_default_stride(
            [I64_MAX],
            1,
            1,
            I64_MAX,
            [0],
            1,
        )
        == ATTN_Q16_ERR_OVERFLOW
    )


def test_randomized_default_stride_parity() -> None:
    rng = random.Random(20260419_548)

    for _ in range(500):
        token_count = rng.randint(0, 40)
        scale_q16 = rng.randint(-(1 << 16), (1 << 16))
        stride = token_count

        if token_count == 0:
            in_capacity = rng.randint(0, 3)
            out_capacity = rng.randint(0, 3)
        else:
            need = 1 + (token_count - 1) * stride
            in_capacity = max(0, need + rng.randint(-3, 3))
            out_capacity = max(0, need + rng.randint(-3, 3))

        in_scores = [0] * max(1, in_capacity)
        for idx in range(token_count):
            base = idx * stride
            if base < len(in_scores):
                in_scores[base] = rng.randint(-(1 << 30), (1 << 30))

        out_new = [0x5AA5] * max(1, out_capacity)
        out_ref = out_new.copy()

        err_new = attention_q16_apply_score_scale_checked_default_stride(
            in_scores,
            in_capacity,
            token_count,
            scale_q16,
            out_new,
            out_capacity,
        )
        err_ref = explicit_default_stride_composition(
            in_scores,
            in_capacity,
            token_count,
            scale_q16,
            out_ref,
            out_capacity,
        )

        assert err_new == err_ref
        assert out_new == out_ref


def run() -> None:
    test_source_contains_default_stride_helper()
    test_known_vectors_match_explicit_default_composition()
    test_adversarial_contracts()
    test_randomized_default_stride_parity()
    print("attention_q16_apply_score_scale_checked_default_stride=ok")


if __name__ == "__main__":
    run()
