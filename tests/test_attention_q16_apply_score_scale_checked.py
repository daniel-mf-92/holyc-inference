#!/usr/bin/env python3
"""Parity checks for AttentionQ16ApplyScoreScaleChecked."""

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

        err, scaled_q32 = q32_scale_by_q16_checked(in_scores_q32[in_base], score_scale_q16)
        if err != ATTN_Q16_OK:
            return err
        out_scores_q32[out_base] = scaled_q32

    return ATTN_Q16_OK


def scalar_reference(scores_q32: list[int], token_count: int, score_stride: int, scale_q16: int) -> list[int]:
    out: list[int] = []
    for token_index in range(token_count):
        base = token_index * score_stride
        err, scaled_q32 = q32_scale_by_q16_checked(scores_q32[base], scale_q16)
        assert err == ATTN_Q16_OK
        out.append(scaled_q32)
    return out


def test_source_contains_helper_symbol() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 AttentionQ16ApplyScoreScaleChecked(" in source


def test_known_vectors_match_reference() -> None:
    token_count = 5
    in_stride = 3
    out_stride = 2
    scale_q16 = 23170  # about 0.35355 in Q16

    in_scores = [0] * (1 + (token_count - 1) * in_stride)
    values = [987654321, -111111111, 222222222, -333333333, 444444444]
    for i, v in enumerate(values):
        in_scores[i * in_stride] = v

    out_scores = [777] * (1 + (token_count - 1) * out_stride)
    err = attention_q16_apply_score_scale_checked(
        in_scores,
        len(in_scores),
        token_count,
        in_stride,
        scale_q16,
        out_scores,
        len(out_scores),
        out_stride,
    )
    assert err == ATTN_Q16_OK

    expected = scalar_reference(in_scores, token_count, in_stride, scale_q16)
    for i in range(token_count):
        assert out_scores[i * out_stride] == expected[i]


def test_randomized_parity() -> None:
    random.seed(0xA7712)

    for _ in range(250):
        token_count = random.randint(0, 40)
        in_stride = random.randint(1, 6)
        out_stride = random.randint(1, 6)
        scale_q16 = random.randint(-(1 << 16), (1 << 16))

        in_cap = 0 if token_count == 0 else 1 + (token_count - 1) * in_stride
        out_cap = 0 if token_count == 0 else 1 + (token_count - 1) * out_stride

        in_scores = [0] * max(in_cap, 1)
        out_scores = [0x1234] * max(out_cap, 1)

        for i in range(token_count):
            in_scores[i * in_stride] = random.randint(-(1 << 30), (1 << 30))

        err = attention_q16_apply_score_scale_checked(
            in_scores,
            in_cap,
            token_count,
            in_stride,
            scale_q16,
            out_scores,
            out_cap,
            out_stride,
        )
        assert err == ATTN_Q16_OK

        expected = scalar_reference(in_scores, token_count, in_stride, scale_q16)
        for i in range(token_count):
            assert out_scores[i * out_stride] == expected[i]


def test_adversarial_error_contracts() -> None:
    sample = [1, 2, 3]
    out = [0, 0, 0]

    assert (
        attention_q16_apply_score_scale_checked(None, 3, 1, 1, 1 << 16, out, 3, 1)
        == ATTN_Q16_ERR_NULL_PTR
    )
    assert (
        attention_q16_apply_score_scale_checked(sample, 3, 1, 1, 1 << 16, None, 3, 1)
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_apply_score_scale_checked(sample, -1, 1, 1, 1 << 16, out, 3, 1)
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_checked(sample, 3, -1, 1, 1 << 16, out, 3, 1)
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_checked(sample, 3, 2, 0, 1 << 16, out, 3, 1)
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_checked(sample, 3, 2, 1, 1 << 16, out, 3, 0)
        == ATTN_Q16_ERR_BAD_PARAM
    )

    # Source span too small.
    assert (
        attention_q16_apply_score_scale_checked(sample, 1, 2, 1, 1 << 16, out, 3, 1)
        == ATTN_Q16_ERR_BAD_PARAM
    )
    # Destination span too small.
    assert (
        attention_q16_apply_score_scale_checked(sample, 3, 2, 1, 1 << 16, out, 1, 1)
        == ATTN_Q16_ERR_BAD_PARAM
    )

    # Multiply overflow while scaling.
    overflow_in = [I64_MAX]
    overflow_out = [0]
    assert (
        attention_q16_apply_score_scale_checked(
            overflow_in,
            1,
            1,
            1,
            I64_MAX,
            overflow_out,
            1,
            1,
        )
        == ATTN_Q16_ERR_OVERFLOW
    )


def run() -> None:
    test_source_contains_helper_symbol()
    test_known_vectors_match_reference()
    test_randomized_parity()
    test_adversarial_error_contracts()
    print("attention_q16_apply_score_scale_checked=ok")


if __name__ == "__main__":
    run()
