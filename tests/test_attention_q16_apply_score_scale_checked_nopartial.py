#!/usr/bin/env python3
"""Parity checks for AttentionQ16ApplyScoreScaleCheckedNoPartial."""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_ERR_OVERFLOW,
    ATTN_Q16_OK,
    I64_MAX,
    attention_q16_apply_score_scale_checked,
    try_add_i64_checked,
    try_mul_i64_checked,
)


def attention_q16_apply_score_scale_checked_nopartial(
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

    err, _total_in_cells = try_mul_i64_checked(token_count, in_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, _total_out_cells = try_mul_i64_checked(token_count, out_score_stride)
    if err != ATTN_Q16_OK:
        return err

    err, last_in_base = try_mul_i64_checked(token_count - 1, in_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_in_cells = try_add_i64_checked(last_in_base, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_in_cells > in_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, last_out_base = try_mul_i64_checked(token_count - 1, out_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(last_out_base, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    # All-or-nothing staging into dense temporary scores.
    staged_scores = [0] * token_count
    err = attention_q16_apply_score_scale_checked(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        in_score_stride,
        score_scale_q16,
        staged_scores,
        token_count,
        1,
    )
    if err != ATTN_Q16_OK:
        return err

    for token_index in range(token_count):
        err, out_base = try_mul_i64_checked(token_index, out_score_stride)
        if err != ATTN_Q16_OK:
            return err
        out_scores_q32[out_base] = staged_scores[token_index]

    return ATTN_Q16_OK


def explicit_staged_composition(
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

    err, last_in_base = try_mul_i64_checked(token_count - 1, in_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_in_cells = try_add_i64_checked(last_in_base, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_in_cells > in_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, last_out_base = try_mul_i64_checked(token_count - 1, out_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(last_out_base, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_scores = [0] * token_count
    err = attention_q16_apply_score_scale_checked(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        in_score_stride,
        score_scale_q16,
        staged_scores,
        token_count,
        1,
    )
    if err != ATTN_Q16_OK:
        return err

    for token_index in range(token_count):
        err, out_base = try_mul_i64_checked(token_index, out_score_stride)
        if err != ATTN_Q16_OK:
            return err
        out_scores_q32[out_base] = staged_scores[token_index]

    return ATTN_Q16_OK


def test_source_contains_helper_symbol_and_staged_core_call() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 AttentionQ16ApplyScoreScaleCheckedNoPartial(" in source
    body = source.split("I32 AttentionQ16ApplyScoreScaleCheckedNoPartial(", 1)[1]
    assert "status = AttentionQ16ApplyScoreScaleChecked(" in body
    assert "staged_scores_q32 = MAlloc(stage_bytes);" in body


def test_known_vector_no_partial_parity() -> None:
    token_count = 4
    in_stride = 3
    out_stride = 4
    scale_q16 = 23170

    in_scores = [0] * (1 + (token_count - 1) * in_stride)
    values = [123456789, -222222222, 333333333, -444444444]
    for i, value in enumerate(values):
        in_scores[i * in_stride] = value

    out_new = [999] * (1 + (token_count - 1) * out_stride)
    out_ref = out_new.copy()

    err_new = attention_q16_apply_score_scale_checked_nopartial(
        in_scores,
        len(in_scores),
        token_count,
        in_stride,
        scale_q16,
        out_new,
        len(out_new),
        out_stride,
    )
    err_ref = explicit_staged_composition(
        in_scores,
        len(in_scores),
        token_count,
        in_stride,
        scale_q16,
        out_ref,
        len(out_ref),
        out_stride,
    )

    assert err_new == err_ref == ATTN_Q16_OK
    assert out_new == out_ref


def test_failure_paths_preserve_output_no_partial() -> None:
    base_out = [0x5AA5] * 8

    out_new = base_out.copy()
    out_ref = base_out.copy()
    err_new = attention_q16_apply_score_scale_checked_nopartial(
        [I64_MAX, 2, 3],
        3,
        1,
        1,
        I64_MAX,
        out_new,
        len(out_new),
        1,
    )
    err_ref = explicit_staged_composition(
        [I64_MAX, 2, 3],
        3,
        1,
        1,
        I64_MAX,
        out_ref,
        len(out_ref),
        1,
    )
    assert err_new == err_ref == ATTN_Q16_ERR_OVERFLOW
    assert out_new == base_out
    assert out_ref == base_out

    out_new = base_out.copy()
    out_ref = base_out.copy()
    err_new = attention_q16_apply_score_scale_checked_nopartial(
        [1, 2, 3],
        1,
        2,
        1,
        1 << 16,
        out_new,
        len(out_new),
        1,
    )
    err_ref = explicit_staged_composition(
        [1, 2, 3],
        1,
        2,
        1,
        1 << 16,
        out_ref,
        len(out_ref),
        1,
    )
    assert err_new == err_ref == ATTN_Q16_ERR_BAD_PARAM
    assert out_new == base_out
    assert out_ref == base_out


def test_randomized_parity_and_no_partial_contract() -> None:
    rng = random.Random(20260419_547)

    for _ in range(500):
        token_count = rng.randint(0, 32)
        in_stride = rng.randint(1, 6)
        out_stride = rng.randint(1, 6)
        scale_q16 = rng.randint(-(1 << 16), (1 << 16))

        in_capacity = 0 if token_count == 0 else 1 + (token_count - 1) * in_stride
        out_capacity = 0 if token_count == 0 else 1 + (token_count - 1) * out_stride

        in_scores = [0] * max(in_capacity, 1)
        for i in range(token_count):
            in_scores[i * in_stride] = rng.randint(-(1 << 30), (1 << 30))

        baseline = [rng.randint(-999, 999) for _ in range(max(out_capacity, 1))]
        out_new = copy.deepcopy(baseline)
        out_ref = copy.deepcopy(baseline)

        err_new = attention_q16_apply_score_scale_checked_nopartial(
            in_scores,
            in_capacity,
            token_count,
            in_stride,
            scale_q16,
            out_new,
            out_capacity,
            out_stride,
        )
        err_ref = explicit_staged_composition(
            in_scores,
            in_capacity,
            token_count,
            in_stride,
            scale_q16,
            out_ref,
            out_capacity,
            out_stride,
        )

        assert err_new == err_ref
        if err_new == ATTN_Q16_OK:
            assert out_new == out_ref
        else:
            assert out_new == baseline
            assert out_ref == baseline


def run() -> None:
    test_source_contains_helper_symbol_and_staged_core_call()
    test_known_vector_no_partial_parity()
    test_failure_paths_preserve_output_no_partial()
    test_randomized_parity_and_no_partial_contract()
    print("attention_q16_apply_score_scale_checked_nopartial=ok")


if __name__ == "__main__":
    run()
