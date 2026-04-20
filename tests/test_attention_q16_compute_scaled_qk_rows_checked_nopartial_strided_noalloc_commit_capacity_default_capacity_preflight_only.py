#!/usr/bin/env python3
"""Parity harness for ...StridedNoAllocCommitCapacityDefaultCapacityPreflightOnly."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity,
)

I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity_preflight_only(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    query_row_stride_q16: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_row_stride: int,
    staged_scores_capacity: int,
    out_commit_stage_cell_capacity: list[int] | None,
    out_commit_stage_byte_capacity: list[int] | None,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_commit_stage_cell_capacity is None
        or out_commit_stage_byte_capacity is None
        or
        out_commit_required_stage_cells is None
        or out_commit_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if (
        q_rows_capacity < 0
        or k_rows_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if (
        query_row_count < 0
        or query_row_stride_q16 < 0
        or token_count < 0
        or k_row_stride_q16 < 0
        or head_dim < 0
        or out_row_stride < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    err, commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, commit_stage_byte_capacity = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    status = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
    )
    if status != ATTN_Q16_OK:
        return status

    out_commit_stage_cell_capacity[0] = commit_stage_cell_capacity
    out_commit_stage_byte_capacity[0] = commit_stage_byte_capacity
    return ATTN_Q16_OK


def explicit_checked_preflight_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    query_row_stride_q16: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_row_stride: int,
    staged_scores_capacity: int,
    out_commit_stage_cell_capacity: list[int] | None,
    out_commit_stage_byte_capacity: list[int] | None,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_commit_stage_cell_capacity is None
        or out_commit_stage_byte_capacity is None
        or
        out_commit_required_stage_cells is None
        or out_commit_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if (
        q_rows_capacity < 0
        or k_rows_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if (
        query_row_count < 0
        or query_row_stride_q16 < 0
        or token_count < 0
        or k_row_stride_q16 < 0
        or head_dim < 0
        or out_row_stride < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    err, commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, commit_stage_byte_capacity = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    status = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
    )
    if status != ATTN_Q16_OK:
        return status

    out_commit_stage_cell_capacity[0] = commit_stage_cell_capacity
    out_commit_stage_byte_capacity[0] = commit_stage_byte_capacity
    return ATTN_Q16_OK


def test_source_contains_strided_default_capacity_preflight_only_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityDefaultCapacityPreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionTryMulI64Checked(query_row_count," in body
    assert "AttentionTryMulI64Checked(staged_scores_capacity," in body
    assert "*out_commit_stage_cell_capacity = commit_stage_cell_capacity;" in body
    assert "*out_commit_stage_byte_capacity = commit_stage_byte_capacity;" in body
    assert (
        "status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacity("
        in body
    )


def test_known_vectors_and_zero_case() -> None:
    query_row_count = 3
    token_count = 5
    head_dim = 7
    query_row_stride_q16 = 9
    k_row_stride_q16 = 11
    out_row_stride = 6

    q_capacity = query_row_count * query_row_stride_q16
    k_capacity = token_count * k_row_stride_q16
    out_capacity = query_row_count * out_row_stride

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    out_scores = [0] * out_capacity

    got_stage_cells = [111]
    got_stage_bytes = [222]
    got_out_cells = [333]
    got_commit_stage_cells = [777]
    got_commit_stage_bytes = [888]
    exp_stage_cells = [444]
    exp_stage_bytes = [555]
    exp_out_cells = [666]
    exp_commit_stage_cells = [999]
    exp_commit_stage_bytes = [1111]

    err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity_preflight_only(
        q_rows,
        q_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows,
        k_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores,
        out_capacity,
        out_row_stride,
        query_row_count * token_count,
        got_commit_stage_cells,
        got_commit_stage_bytes,
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
    )
    err_exp = explicit_checked_preflight_composition(
        q_rows,
        q_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows,
        k_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores,
        out_capacity,
        out_row_stride,
        query_row_count * token_count,
        exp_commit_stage_cells,
        exp_commit_stage_bytes,
        exp_stage_cells,
        exp_stage_bytes,
        exp_out_cells,
    )

    assert err_got == err_exp == ATTN_Q16_OK
    assert got_commit_stage_cells == exp_commit_stage_cells == [query_row_count * token_count]
    assert got_commit_stage_bytes == exp_commit_stage_bytes == [query_row_count * token_count * 8]
    assert got_stage_cells == exp_stage_cells == [query_row_count * token_count]
    assert got_stage_bytes == exp_stage_bytes == [query_row_count * token_count * 8]

    required_out_cells = (query_row_count - 1) * out_row_stride + token_count
    assert got_out_cells == exp_out_cells == [required_out_cells]

    # zero query rows => zero required stage/out cells
    z_stage_cells = [9]
    z_stage_bytes = [9]
    z_out_cells = [9]
    z_commit_stage_cells = [9]
    z_commit_stage_bytes = [9]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity_preflight_only(
        [0],
        1,
        0,
        3,
        [0],
        1,
        4,
        4,
        3,
        [0],
        1,
        5,
        2,
        z_commit_stage_cells,
        z_commit_stage_bytes,
        z_stage_cells,
        z_stage_bytes,
        z_out_cells,
    )
    assert err == ATTN_Q16_OK
    assert z_commit_stage_cells == [0]
    assert z_commit_stage_bytes == [16]
    assert z_stage_cells == [0]
    assert z_stage_bytes == [0]
    assert z_out_cells == [0]


def test_randomized_parity_and_overflow_behavior() -> None:
    random.seed(639)

    for _ in range(250):
        query_row_count = random.randint(0, 10)
        token_count = random.randint(0, 10)
        head_dim = random.randint(0, 12)

        query_row_stride_q16 = head_dim + random.randint(0, 5)
        k_row_stride_q16 = head_dim + random.randint(0, 5)
        out_row_stride = token_count + random.randint(0, 5)

        q_capacity = query_row_count * query_row_stride_q16
        k_capacity = token_count * k_row_stride_q16
        out_capacity = query_row_count * out_row_stride

        staged_capacity = max(0, query_row_count * token_count + random.randint(-3, 4))

        q_rows = [0] * max(1, q_capacity)
        k_rows = [0] * max(1, k_capacity)
        out_scores = [0] * max(1, out_capacity)

        got_stage_cells = [101]
        got_stage_bytes = [202]
        got_out_cells = [303]
        got_commit_stage_cells = [707]
        got_commit_stage_bytes = [808]
        exp_stage_cells = [404]
        exp_stage_bytes = [505]
        exp_out_cells = [606]
        exp_commit_stage_cells = [909]
        exp_commit_stage_bytes = [1001]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity_preflight_only(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_scores,
            out_capacity,
            out_row_stride,
            staged_capacity,
            got_commit_stage_cells,
            got_commit_stage_bytes,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
        )
        err_exp = explicit_checked_preflight_composition(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_scores,
            out_capacity,
            out_row_stride,
            staged_capacity,
            exp_commit_stage_cells,
            exp_commit_stage_bytes,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_commit_stage_cells == exp_commit_stage_cells
            assert got_commit_stage_bytes == exp_commit_stage_bytes
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
            assert got_out_cells == exp_out_cells
        else:
            assert got_commit_stage_cells == [707]
            assert got_commit_stage_bytes == [808]
            assert got_stage_cells == [101]
            assert got_stage_bytes == [202]
            assert got_out_cells == [303]
            assert exp_commit_stage_cells == [909]
            assert exp_commit_stage_bytes == [1001]
            assert exp_stage_cells == [404]
            assert exp_stage_bytes == [505]
            assert exp_out_cells == [606]

    out1 = [1]
    out2 = [1]
    out3 = [1]
    out4 = [1]
    out5 = [1]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity_preflight_only(
        [0],
        1,
        I64_MAX,
        1,
        [0],
        1,
        2,
        2,
        1,
        [0],
        1,
        2,
        1,
        out1,
        out2,
        out3,
        out4,
        out5,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


if __name__ == "__main__":
    test_source_contains_strided_default_capacity_preflight_only_wrapper()
    test_known_vectors_and_zero_case()
    test_randomized_parity_and_overflow_behavior()
    print("ok")
