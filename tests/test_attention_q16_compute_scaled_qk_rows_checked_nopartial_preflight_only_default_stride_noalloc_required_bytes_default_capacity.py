#!/usr/bin/env python3
"""Parity harness for ...PreflightOnlyDefaultStrideNoAllocRequiredBytesDefaultCapacity."""

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
from test_attention_q16_compute_scaled_qk_rows_checked import try_mul_i64_checked
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes,
)

I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes_default_capacity(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, default_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        default_stage_cell_capacity,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells,
    )


def explicit_checked_default_capacity_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, default_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        default_stage_cell_capacity,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells,
    )


def test_source_contains_default_capacity_required_bytes_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnlyDefaultStrideNoAllocRequiredBytesDefaultCapacity("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "status = AttentionTryMulI64Checked(query_row_count," in body
    assert "token_count," in body
    assert (
        "return AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnlyDefaultStrideNoAllocRequiredBytes(" in body
    )


def test_known_vectors_and_zero_case() -> None:
    query_row_count = 4
    token_count = 6
    head_dim = 8

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    out_scores = [0] * out_capacity

    got_stage_cells = [111]
    got_stage_bytes = [222]
    got_out = [333]

    exp_stage_cells = [444]
    exp_stage_bytes = [555]
    exp_out = [666]

    err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes_default_capacity(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        got_stage_cells,
        got_stage_bytes,
        got_out,
    )
    err_exp = explicit_checked_default_capacity_composition(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        exp_stage_cells,
        exp_stage_bytes,
        exp_out,
    )

    assert err_got == err_exp == ATTN_Q16_OK
    assert got_stage_cells == exp_stage_cells == [query_row_count * token_count]
    assert got_stage_bytes == exp_stage_bytes == [query_row_count * token_count * 8]
    assert got_out == exp_out == [query_row_count * token_count]

    got_stage_cells = [9]
    got_stage_bytes = [8]
    got_out = [7]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes_default_capacity(
        [0],
        1,
        0,
        [0],
        1,
        7,
        3,
        [0],
        1,
        got_stage_cells,
        got_stage_bytes,
        got_out,
    )
    assert err == ATTN_Q16_OK
    assert got_stage_cells == [0]
    assert got_stage_bytes == [0]
    assert got_out == [0]


def test_error_paths_preserve_outputs() -> None:
    out_stage_cells = [41]
    out_stage_bytes = [42]
    out_out = [43]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes_default_capacity(
        None,
        1,
        1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        out_stage_cells,
        out_stage_bytes,
        out_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert out_stage_cells == [41]
    assert out_stage_bytes == [42]
    assert out_out == [43]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes_default_capacity(
        [0],
        -1,
        1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        out_stage_cells,
        out_stage_bytes,
        out_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_stage_cells == [41]
    assert out_stage_bytes == [42]
    assert out_out == [43]

    huge = 1 << 62
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes_default_capacity(
        [0],
        I64_MAX,
        huge,
        [0],
        I64_MAX,
        8,
        8,
        [0],
        I64_MAX,
        out_stage_cells,
        out_stage_bytes,
        out_out,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out_stage_cells == [41]
    assert out_stage_bytes == [42]
    assert out_out == [43]


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(650)

    for _ in range(400):
        query_row_count = rng.randint(0, 10)
        token_count = rng.randint(0, 12)
        head_dim = rng.randint(0, 9)

        required_q = query_row_count * head_dim
        required_k = token_count * head_dim
        required_out = query_row_count * token_count

        q_capacity = required_q + rng.randint(0, 4)
        k_capacity = required_k + rng.randint(0, 4)
        out_capacity = required_out + rng.randint(0, 4)

        if rng.random() < 0.15 and required_q > 0:
            q_capacity = rng.randint(0, required_q - 1)
        if rng.random() < 0.15 and required_k > 0:
            k_capacity = rng.randint(0, required_k - 1)
        if rng.random() < 0.15 and required_out > 0:
            out_capacity = rng.randint(0, required_out - 1)

        q_rows = [0] * max(q_capacity, 1)
        k_rows = [0] * max(k_capacity, 1)
        out_scores = [0] * max(out_capacity, 1)

        got_stage_cells = [0xA1]
        got_stage_bytes = [0xA2]
        got_out = [0xA3]

        exp_stage_cells = [0xB1]
        exp_stage_bytes = [0xB2]
        exp_out = [0xB3]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes_default_capacity(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            got_stage_cells,
            got_stage_bytes,
            got_out,
        )
        err_exp = explicit_checked_default_capacity_composition(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
            assert got_out == exp_out
        else:
            assert got_stage_cells == [0xA1]
            assert got_stage_bytes == [0xA2]
            assert got_out == [0xA3]


if __name__ == "__main__":
    test_source_contains_default_capacity_required_bytes_wrapper()
    test_known_vectors_and_zero_case()
    test_error_paths_preserve_outputs()
    test_randomized_parity_against_explicit_composition()
    print("ok")
