#!/usr/bin/env python3
"""Parity checks for AttentionQ16ComputeQKDotRowChecked."""

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


# Keep this parity harness self-contained so it can validate contracts
# before/without TempleOS runtime integration.
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


def attention_q16_compute_qk_dot_row_checked(
    q_row_q16,
    q_row_capacity: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_score_stride: int,
) -> int:
    if q_row_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_row_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if token_count < 0 or k_row_stride_q16 < 0 or head_dim < 0 or out_score_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if head_dim > q_row_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if token_count > 0 and k_row_stride_q16 < head_dim:
        return ATTN_Q16_ERR_BAD_PARAM
    if token_count > 0 and out_score_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_k_cells = try_mul_i64_checked(token_count, k_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err
    if required_k_cells > k_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if token_count == 0:
        return ATTN_Q16_OK

    err, required_out_cells = try_mul_i64_checked(token_count - 1, out_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    for token_index in range(token_count):
        err, token_base = try_mul_i64_checked(token_index, k_row_stride_q16)
        if err != ATTN_Q16_OK:
            return err
        err, out_base = try_mul_i64_checked(token_index, out_score_stride)
        if err != ATTN_Q16_OK:
            return err

        score_q32 = 0
        for lane_index in range(head_dim):
            err, product_q32 = try_mul_i64_checked(
                q_row_q16[lane_index],
                k_rows_q16[token_base + lane_index],
            )
            if err != ATTN_Q16_OK:
                return err
            err, score_q32 = try_add_i64_checked(score_q32, product_q32)
            if err != ATTN_Q16_OK:
                return err

        out_scores_q32[out_base] = score_q32

    return ATTN_Q16_OK


def scalar_reference_scores(q_row_q16: list[int], k_rows_q16: list[int], token_count: int, k_row_stride_q16: int, head_dim: int) -> list[int]:
    out = []
    for token_index in range(token_count):
        base = token_index * k_row_stride_q16
        score = 0
        for lane_index in range(head_dim):
            score += q_row_q16[lane_index] * k_rows_q16[base + lane_index]
        out.append(score)
    return out


def q16_from_text(text: str, width: int) -> list[int]:
    raw = text.encode("utf-8")
    vals: list[int] = []
    for b in raw:
        # Center bytes around zero then lift to Q16.
        vals.append((b - 128) << 16)
        if len(vals) == width:
            break
    while len(vals) < width:
        vals.append(0)
    return vals


def test_source_contains_helper_symbol() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 AttentionQ16ComputeQKDotRowChecked(" in source


def test_known_multilingual_span_vectors_match_reference() -> None:
    head_dim = 16
    token_count = 4
    k_row_stride_q16 = 20
    out_score_stride = 2

    q_row = q16_from_text("γειά σου κόσμε", head_dim)
    token_texts = [
        "こんにちは世界",
        "مرحبا بالعالم",
        "Привет, мир",
        "hola mundo",
    ]

    k_rows = [0] * (token_count * k_row_stride_q16)
    for token_index, text in enumerate(token_texts):
        row = q16_from_text(text, head_dim)
        base = token_index * k_row_stride_q16
        for lane in range(head_dim):
            k_rows[base + lane] = row[lane]

    out = [777] * ((token_count - 1) * out_score_stride + 1)
    err = attention_q16_compute_qk_dot_row_checked(
        q_row,
        len(q_row),
        k_rows,
        len(k_rows),
        token_count,
        k_row_stride_q16,
        head_dim,
        out,
        len(out),
        out_score_stride,
    )
    assert err == ATTN_Q16_OK

    expected = scalar_reference_scores(q_row, k_rows, token_count, k_row_stride_q16, head_dim)
    for token_index in range(token_count):
        assert out[token_index * out_score_stride] == expected[token_index]

    # Ensure deterministic layout: untouched padding slots remain sentinel.
    for token_index in range(token_count):
        pad_index = token_index * out_score_stride + 1
        if pad_index < len(out):
            assert out[pad_index] == 777


def test_randomized_parity() -> None:
    rng = random.Random(20260419)

    for _ in range(200):
        head_dim = rng.randint(1, 48)
        token_count = rng.randint(1, 40)
        k_row_stride_q16 = head_dim + rng.randint(0, 8)
        out_score_stride = rng.randint(1, 4)

        # Keep values in a safe Q16 range so success-path parity is stable.
        q_row = [rng.randint(-64, 64) << 16 for _ in range(head_dim)]
        k_rows = [0] * (token_count * k_row_stride_q16)
        for token_index in range(token_count):
            base = token_index * k_row_stride_q16
            for lane in range(head_dim):
                k_rows[base + lane] = rng.randint(-64, 64) << 16

        out_len = (token_count - 1) * out_score_stride + 1
        out = [0] * out_len

        err = attention_q16_compute_qk_dot_row_checked(
            q_row,
            len(q_row),
            k_rows,
            len(k_rows),
            token_count,
            k_row_stride_q16,
            head_dim,
            out,
            len(out),
            out_score_stride,
        )
        assert err == ATTN_Q16_OK

        expected = scalar_reference_scores(q_row, k_rows, token_count, k_row_stride_q16, head_dim)
        for token_index, score in enumerate(expected):
            assert out[token_index * out_score_stride] == score


def test_adversarial_error_contracts() -> None:
    q = [1 << 16, 2 << 16]
    k = [3 << 16, 4 << 16, 5 << 16, 6 << 16]
    out = [0, 0]

    err = attention_q16_compute_qk_dot_row_checked(None, 0, k, len(k), 1, 2, 2, out, len(out), 1)
    assert err == ATTN_Q16_ERR_NULL_PTR

    err = attention_q16_compute_qk_dot_row_checked(q, len(q), k, len(k), -1, 2, 2, out, len(out), 1)
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_compute_qk_dot_row_checked(q, 1, k, len(k), 1, 2, 2, out, len(out), 1)
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_compute_qk_dot_row_checked(q, len(q), k, len(k), 2, 1, 2, out, len(out), 1)
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_compute_qk_dot_row_checked(q, len(q), k, len(k), 2, 2, 2, out, 1, 1)
    assert err == ATTN_Q16_ERR_BAD_PARAM

    # Multiply overflow from adversarial lanes.
    q_over = [I64_MAX]
    k_over = [2]
    out_over = [0]
    err = attention_q16_compute_qk_dot_row_checked(
        q_over,
        len(q_over),
        k_over,
        len(k_over),
        1,
        1,
        1,
        out_over,
        len(out_over),
        1,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def run() -> None:
    test_source_contains_helper_symbol()
    test_known_multilingual_span_vectors_match_reference()
    test_randomized_parity()
    test_adversarial_error_contracts()
    print("attention_q16_compute_qk_dot_row_checked=ok")


if __name__ == "__main__":
    run()
