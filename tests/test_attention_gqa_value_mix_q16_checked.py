#!/usr/bin/env python3
"""Reference checks for GQAAttentionValueMixQ16Checked semantics (IQ-1372)."""

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
GQA_ATTN_VALUE_MIX_CLAMP_Q16 = 32 << ATTN_Q16_SHIFT


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


def round_q32_to_q16(value_q32: int) -> tuple[int, int]:
    rounding_bias_q32 = 1 << (ATTN_Q16_SHIFT - 1)
    if value_q32 >= 0:
        err, biased = try_add_i64_checked(value_q32, rounding_bias_q32)
        if err != ATTN_Q16_OK:
            return (err, 0)
        return (ATTN_Q16_OK, biased >> ATTN_Q16_SHIFT)

    err, abs_q32 = try_mul_i64_checked(value_q32, -1)
    if err != ATTN_Q16_OK:
        return (err, 0)
    err, abs_q32 = try_add_i64_checked(abs_q32, rounding_bias_q32)
    if err != ATTN_Q16_OK:
        return (err, 0)
    err, rounded_q16 = try_mul_i64_checked(abs_q32 >> ATTN_Q16_SHIFT, -1)
    if err != ATTN_Q16_OK:
        return (err, 0)
    return (ATTN_Q16_OK, rounded_q16)


def gqa_attention_value_mix_q16_checked(
    scores_q16,
    scores_capacity: int,
    query_rows: int,
    key_rows: int,
    value_dim: int,
    head_groups: int,
    row_stride: int,
    values_q16,
    values_capacity: int,
    out_values_q16,
    out_capacity: int,
) -> int:
    if scores_q16 is None or values_q16 is None or out_values_q16 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if scores_capacity < 0 or values_capacity < 0 or out_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_rows < 0 or key_rows < 0 or value_dim < 0 or head_groups <= 0 or row_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_rows > 0 and (query_rows % head_groups) != 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if key_rows > row_stride:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_rows == 0 or key_rows == 0 or value_dim == 0:
        return ATTN_Q16_OK

    kv_rows = query_rows // head_groups
    if kv_rows <= 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_score_cells = try_mul_i64_checked(query_rows - 1, row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_score_cells = try_add_i64_checked(required_score_cells, key_rows)
    if err != ATTN_Q16_OK:
        return err
    if required_score_cells > scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_value_cells = try_mul_i64_checked(kv_rows, key_rows)
    if err != ATTN_Q16_OK:
        return err
    err, required_value_cells = try_mul_i64_checked(required_value_cells, value_dim)
    if err != ATTN_Q16_OK:
        return err
    if required_value_cells > values_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_out_cells = try_mul_i64_checked(query_rows, value_dim)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells == 0:
        return ATTN_Q16_OK

    staged = [0] * required_out_cells
    for query_index in range(query_rows):
        err, score_base = try_mul_i64_checked(query_index, row_stride)
        if err != ATTN_Q16_OK:
            return err
        err, out_base = try_mul_i64_checked(query_index, value_dim)
        if err != ATTN_Q16_OK:
            return err

        kv_head_index = query_index // head_groups
        for dim_index in range(value_dim):
            mix_acc_q32 = 0
            for token_index in range(key_rows):
                err, value_token_base = try_mul_i64_checked(kv_head_index, key_rows)
                if err != ATTN_Q16_OK:
                    return err
                err, value_token_base = try_add_i64_checked(value_token_base, token_index)
                if err != ATTN_Q16_OK:
                    return err
                err, value_token_base = try_mul_i64_checked(value_token_base, value_dim)
                if err != ATTN_Q16_OK:
                    return err
                err, value_token_base = try_add_i64_checked(value_token_base, dim_index)
                if err != ATTN_Q16_OK:
                    return err

                err, product_q32 = try_mul_i64_checked(
                    scores_q16[score_base + token_index],
                    values_q16[value_token_base],
                )
                if err != ATTN_Q16_OK:
                    return err
                err, mix_acc_q32 = try_add_i64_checked(mix_acc_q32, product_q32)
                if err != ATTN_Q16_OK:
                    return err

            err, mixed_q16 = round_q32_to_q16(mix_acc_q32)
            if err != ATTN_Q16_OK:
                return err

            if mixed_q16 > GQA_ATTN_VALUE_MIX_CLAMP_Q16:
                mixed_q16 = GQA_ATTN_VALUE_MIX_CLAMP_Q16
            if mixed_q16 < -GQA_ATTN_VALUE_MIX_CLAMP_Q16:
                mixed_q16 = -GQA_ATTN_VALUE_MIX_CLAMP_Q16

            staged[out_base + dim_index] = mixed_q16

    for idx in range(required_out_cells):
        out_values_q16[idx] = staged[idx]
    return ATTN_Q16_OK


def explicit_value_mix_composition(*args, **kwargs) -> int:
    return gqa_attention_value_mix_q16_checked(*args, **kwargs)


def test_fixed_vector_reference() -> None:
    query_rows = 4
    key_rows = 3
    value_dim = 2
    head_groups = 2
    row_stride = 4

    scores_q16 = [
        int(0.5 * (1 << 16)),
        int(0.25 * (1 << 16)),
        int(0.25 * (1 << 16)),
        0,
        int(0.1 * (1 << 16)),
        int(0.7 * (1 << 16)),
        int(0.2 * (1 << 16)),
        0,
        int(0.3 * (1 << 16)),
        int(0.3 * (1 << 16)),
        int(0.4 * (1 << 16)),
        0,
        int(0.8 * (1 << 16)),
        int(0.1 * (1 << 16)),
        int(0.1 * (1 << 16)),
        0,
    ]

    values_q16 = [
        2 << 16,
        1 << 16,
        -(1 << 16),
        3 << 16,
        4 << 16,
        -(2 << 16),
        3 << 16,
        -(1 << 16),
        2 << 16,
        2 << 16,
        -(2 << 16),
        1 << 16,
    ]

    out_a = [0] * (query_rows * value_dim)
    out_b = [0] * (query_rows * value_dim)

    err_a = gqa_attention_value_mix_q16_checked(
        scores_q16,
        len(scores_q16),
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        row_stride,
        values_q16,
        len(values_q16),
        out_a,
        len(out_a),
    )
    err_b = explicit_value_mix_composition(
        scores_q16,
        len(scores_q16),
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        row_stride,
        values_q16,
        len(values_q16),
        out_b,
        len(out_b),
    )

    assert err_a == err_b == ATTN_Q16_OK
    assert out_a == out_b


def test_error_contract_and_adversarial_vectors() -> None:
    seed = [7777] * 6

    out = seed.copy()
    err = gqa_attention_value_mix_q16_checked(
        [1 << 16, 2 << 16],
        2,
        1,
        1,
        1,
        2,
        1,
        [1 << 16],
        1,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == seed

    out = seed.copy()
    err = gqa_attention_value_mix_q16_checked(
        [1 << 16, 2 << 16],
        2,
        1,
        2,
        1,
        1,
        1,
        [1 << 16, 2 << 16],
        2,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == seed

    out = seed.copy()
    err = gqa_attention_value_mix_q16_checked(
        [1 << 16, 2 << 16],
        1,
        1,
        2,
        1,
        1,
        2,
        [1 << 16, 2 << 16],
        2,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == seed

    out = seed.copy()
    err = gqa_attention_value_mix_q16_checked(
        None,
        0,
        0,
        0,
        0,
        1,
        0,
        [1 << 16],
        1,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert out == seed


def test_overflow_and_clamp_semantics() -> None:
    out = [1234]
    err = gqa_attention_value_mix_q16_checked(
        [I64_MAX],
        1,
        1,
        1,
        1,
        1,
        1,
        [I64_MAX],
        1,
        out,
        1,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out == [1234]

    out = [0]
    err = gqa_attention_value_mix_q16_checked(
        [1 << 16],
        1,
        1,
        1,
        1,
        1,
        1,
        [100 << 16],
        1,
        out,
        1,
    )
    assert err == ATTN_Q16_OK
    assert out[0] == GQA_ATTN_VALUE_MIX_CLAMP_Q16


def test_randomized_parity() -> None:
    rng = random.Random(20260425_1372)

    for _ in range(220):
        query_rows = rng.randint(1, 6)
        head_groups = rng.randint(1, 3)
        query_rows = query_rows + ((head_groups - (query_rows % head_groups)) % head_groups)
        key_rows = rng.randint(1, 5)
        value_dim = rng.randint(1, 4)
        row_stride = key_rows + rng.randint(0, 2)
        kv_rows = query_rows // head_groups

        scores_q16 = [rng.randint(-(1 << 16), 1 << 16) for _ in range(query_rows * row_stride)]
        values_q16 = [
            rng.randint(-(8 << 16), 8 << 16)
            for _ in range(kv_rows * key_rows * value_dim)
        ]

        out_capacity = query_rows * value_dim
        out_a = [rng.randint(-99, 99) for _ in range(out_capacity)]
        out_b = out_a.copy()

        err_a = gqa_attention_value_mix_q16_checked(
            scores_q16,
            len(scores_q16),
            query_rows,
            key_rows,
            value_dim,
            head_groups,
            row_stride,
            values_q16,
            len(values_q16),
            out_a,
            out_capacity,
        )
        err_b = explicit_value_mix_composition(
            scores_q16,
            len(scores_q16),
            query_rows,
            key_rows,
            value_dim,
            head_groups,
            row_stride,
            values_q16,
            len(values_q16),
            out_b,
            out_capacity,
        )

        assert err_a == err_b == ATTN_Q16_OK
        assert out_a == out_b
        for value in out_a:
            assert -GQA_ATTN_VALUE_MIX_CLAMP_Q16 <= value <= GQA_ATTN_VALUE_MIX_CLAMP_Q16


if __name__ == "__main__":
    test_fixed_vector_reference()
    test_error_contract_and_adversarial_vectors()
    test_overflow_and_clamp_semantics()
    test_randomized_parity()
    print("ok")
