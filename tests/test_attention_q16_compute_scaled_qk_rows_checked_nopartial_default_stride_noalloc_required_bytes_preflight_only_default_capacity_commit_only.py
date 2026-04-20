#!/usr/bin/env python3
"""Parity harness for ...RequiredBytesPreflightOnlyDefaultCapacityCommitOnly."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
) -> int:
    if (
        out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    canonical_q = [0]
    canonical_k = [0]
    canonical_out = [0]
    canonical_stage_cells = [0]
    canonical_stage_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        canonical_q,
        canonical_k,
        canonical_out,
        canonical_stage_cells,
        canonical_stage_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    err, default_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    parity_q = [0]
    parity_k = [0]
    parity_out = [0]
    parity_stage_cells = [0]
    parity_stage_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        parity_q,
        parity_k,
        parity_out,
        parity_stage_cells,
        parity_stage_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    if parity_stage_cells[0] != default_stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if canonical_q[0] != parity_q[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if canonical_k[0] != parity_k[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if canonical_out[0] != parity_out[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if canonical_stage_cells[0] != parity_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if canonical_stage_bytes[0] != parity_stage_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_q_cells[0] = canonical_q[0]
    out_required_k_cells[0] = canonical_k[0]
    out_required_out_cells[0] = canonical_out[0]
    out_required_stage_cells[0] = canonical_stage_cells[0]
    out_required_stage_bytes[0] = canonical_stage_bytes[0]
    return ATTN_Q16_OK


def explicit_checked_composition(*args, **kwargs) -> int:
    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only(
        *args, **kwargs
    )


def test_source_contains_required_bytes_preflight_only_default_capacity_commit_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesPreflightOnlyDefaultCapacityCommitOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesPreflightOnlyDefaultCapacity(" in body
    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesPreflightOnly(" in body
    assert "if (parity_required_stage_cells != default_stage_cell_capacity)" in body


def test_known_vector_and_no_partial_publish_on_error() -> None:
    q_rows = [0] * 96
    k_rows = [0] * 96
    out_scores = [0] * 96

    got_q = [111]
    got_k = [112]
    got_out = [113]
    got_stage_cells = [114]
    got_stage_bytes = [115]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only(
        q_rows,
        len(q_rows),
        3,
        k_rows,
        len(k_rows),
        4,
        6,
        out_scores,
        len(out_scores),
        got_q,
        got_k,
        got_out,
        got_stage_cells,
        got_stage_bytes,
    )
    assert err == ATTN_Q16_OK
    assert got_q[0] == 18
    assert got_k[0] == 24
    assert got_out[0] == 12
    assert got_stage_cells[0] == 12
    assert got_stage_bytes[0] == 96

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only(
        q_rows,
        len(q_rows),
        (1 << 63) - 1,
        k_rows,
        len(k_rows),
        2,
        1,
        out_scores,
        len(out_scores),
        got_q,
        got_k,
        got_out,
        got_stage_cells,
        got_stage_bytes,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert got_q == [18]
    assert got_k == [24]
    assert got_out == [12]
    assert got_stage_cells == [12]
    assert got_stage_bytes == [96]


def test_null_and_bad_param_contracts() -> None:
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only(
            None,
            0,
            0,
            [],
            0,
            0,
            0,
            [],
            0,
            [0],
            [0],
            [0],
            [0],
            [0],
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only(
            [0],
            -1,
            0,
            [0],
            1,
            0,
            0,
            [0],
            1,
            [0],
            [0],
            [0],
            [0],
            [0],
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )


def test_randomized_parity() -> None:
    rng = random.Random(20260420_726)

    for _ in range(2400):
        query_row_count = rng.randint(0, 32)
        token_count = rng.randint(0, 32)
        head_dim = rng.randint(0, 32)

        q_capacity = rng.randint(0, 4096)
        k_capacity = rng.randint(0, 4096)
        out_capacity = rng.randint(0, 4096)

        q_rows = [0] * max(1, q_capacity)
        k_rows = [0] * max(1, k_capacity)
        out_scores = [0] * max(1, out_capacity)

        got_q = [9001]
        got_k = [9002]
        got_out = [9003]
        got_stage_cells = [9004]
        got_stage_bytes = [9005]

        exp_q = [8001]
        exp_k = [8002]
        exp_out = [8003]
        exp_stage_cells = [8004]
        exp_stage_bytes = [8005]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            got_q,
            got_k,
            got_out,
            got_stage_cells,
            got_stage_bytes,
        )
        err_exp = explicit_checked_composition(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            exp_q,
            exp_k,
            exp_out,
            exp_stage_cells,
            exp_stage_bytes,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_q[0] == exp_q[0]
            assert got_k[0] == exp_k[0]
            assert got_out[0] == exp_out[0]
            assert got_stage_cells[0] == exp_stage_cells[0]
            assert got_stage_bytes[0] == exp_stage_bytes[0]
        else:
            assert got_q == [9001]
            assert got_k == [9002]
            assert got_out == [9003]
            assert got_stage_cells == [9004]
            assert got_stage_bytes == [9005]


if __name__ == "__main__":
    test_source_contains_required_bytes_preflight_only_default_capacity_commit_only_helper()
    test_known_vector_and_no_partial_publish_on_error()
    test_null_and_bad_param_contracts()
    test_randomized_parity()
    print("ok")
