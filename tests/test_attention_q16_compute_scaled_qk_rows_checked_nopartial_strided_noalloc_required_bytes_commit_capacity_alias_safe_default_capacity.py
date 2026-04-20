#!/usr/bin/env python3
"""Parity harness for ...RequiredBytesCommitCapacityAliasSafeDefaultCapacity."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe,
)

I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity(
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
    staged_scores_q32,
    staged_scores_capacity: int,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    *,
    q_base_addr: int = 0x100000,
    k_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    if (
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

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
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
        staged_scores_q32,
        staged_scores_capacity,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
        q_base_addr=q_base_addr,
        k_base_addr=k_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )


def explicit_checked_composition(
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
    staged_scores_q32,
    staged_scores_capacity: int,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    *,
    q_base_addr: int = 0x100000,
    k_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    err, commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, commit_stage_byte_capacity = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
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
        staged_scores_q32,
        staged_scores_capacity,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
        q_base_addr=q_base_addr,
        k_base_addr=k_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )


def test_source_contains_required_bytes_commit_capacity_alias_safe_default_capacity_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacityAliasSafeDefaultCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionTryMulI64Checked(query_row_count," in body
    assert "AttentionTryMulI64Checked(staged_scores_capacity," in body
    assert (
        "return AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacityAliasSafe("
        in body
    )


def test_known_vectors_and_alias_overlap_path() -> None:
    q_rows = [0] * 56
    k_rows = [0] * 64
    out_scores = [0] * 36
    staged_scores = [0] * 30

    got_stage_cells = [111]
    got_stage_bytes = [222]
    got_out_cells = [333]
    exp_stage_cells = [444]
    exp_stage_bytes = [555]
    exp_out_cells = [666]

    err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity(
        q_rows,
        len(q_rows),
        3,
        8,
        k_rows,
        len(k_rows),
        5,
        7,
        6,
        out_scores,
        len(out_scores),
        8,
        staged_scores,
        len(staged_scores),
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
    )

    err_exp = explicit_checked_composition(
        q_rows,
        len(q_rows),
        3,
        8,
        k_rows,
        len(k_rows),
        5,
        7,
        6,
        out_scores,
        len(out_scores),
        8,
        staged_scores,
        len(staged_scores),
        exp_stage_cells,
        exp_stage_bytes,
        exp_out_cells,
    )

    assert err_got == err_exp == ATTN_Q16_OK
    assert got_stage_cells == exp_stage_cells == [15]
    assert got_stage_bytes == exp_stage_bytes == [120]
    assert got_out_cells == exp_out_cells == [21]

    err_alias = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity(
        q_rows,
        len(q_rows),
        3,
        8,
        k_rows,
        len(k_rows),
        5,
        7,
        6,
        out_scores,
        len(out_scores),
        8,
        staged_scores,
        len(staged_scores),
        [0],
        [0],
        [0],
        stage_base_addr=0x100010,
    )
    assert err_alias == ATTN_Q16_ERR_BAD_PARAM


def test_diagnostics_no_partial_on_failure() -> None:
    q_rows = [0] * 32
    k_rows = [0] * 32
    out_scores = [0] * 32
    staged_scores = [0] * 32

    out_stage_cells = [7001]
    out_stage_bytes = [7002]
    out_out_cells = [7003]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity(
        q_rows,
        len(q_rows),
        2,
        8,
        k_rows,
        len(k_rows),
        4,
        8,
        6,
        out_scores,
        len(out_scores),
        6,
        staged_scores,
        7,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_stage_cells == [7001]
    assert out_stage_bytes == [7002]
    assert out_out_cells == [7003]


def test_error_paths_and_overflow() -> None:
    out_stage_cells = [1]
    out_stage_bytes = [2]
    out_out_cells = [3]

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity(
            [0],
            1,
            1,
            1,
            [0],
            1,
            1,
            1,
            1,
            [0],
            1,
            1,
            [0],
            1,
            None,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity(
            [0],
            -1,
            1,
            1,
            [0],
            1,
            1,
            1,
            1,
            [0],
            1,
            1,
            [0],
            1,
            out_stage_cells,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity(
        [0],
        I64_MAX,
        I64_MAX,
        1,
        [0],
        I64_MAX,
        2,
        1,
        1,
        [0],
        I64_MAX,
        1,
        [0],
        1,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity(
        [0],
        1,
        1,
        1,
        [0],
        1,
        1,
        1,
        1,
        [0],
        1,
        1,
        [0],
        I64_MAX,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_checked_composition() -> None:
    rng = random.Random(20260420_654)

    for _ in range(1200):
        query_row_count = rng.randint(0, 8)
        token_count = rng.randint(0, 10)
        head_dim = rng.randint(0, 12)

        query_row_stride_q16 = rng.randint(max(1, head_dim), max(1, head_dim + 10))
        k_row_stride_q16 = rng.randint(max(1, head_dim), max(1, head_dim + 10))
        out_row_stride = rng.randint(max(1, token_count), max(1, token_count + 10))

        q_capacity = rng.randint(0, 120)
        k_capacity = rng.randint(0, 120)
        out_capacity = rng.randint(0, 120)
        staged_capacity = rng.randint(0, 120)

        q_rows = [rng.randint(-100, 100) for _ in range(max(1, q_capacity))]
        k_rows = [rng.randint(-100, 100) for _ in range(max(1, k_capacity))]
        out_scores = [rng.randint(-100, 100) for _ in range(max(1, out_capacity))]
        staged_scores = [rng.randint(-100, 100) for _ in range(max(1, staged_capacity))]

        got_stage_cells = [101]
        got_stage_bytes = [102]
        got_out_cells = [103]
        exp_stage_cells = [201]
        exp_stage_bytes = [202]
        exp_out_cells = [203]

        q_base_addr = 0x100000 + rng.randint(0, 0x400)
        k_base_addr = 0x200000 + rng.randint(0, 0x400)
        out_base_addr = 0x300000 + rng.randint(0, 0x400)
        stage_base_addr = 0x400000 + rng.randint(0, 0x400)
        if rng.random() < 0.25:
            stage_base_addr = q_base_addr + rng.randint(0, 128)

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity(
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
            staged_scores,
            staged_capacity,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
            q_base_addr=q_base_addr,
            k_base_addr=k_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )

        err_exp = explicit_checked_composition(
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
            staged_scores,
            staged_capacity,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
            q_base_addr=q_base_addr,
            k_base_addr=k_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
            assert got_out_cells == exp_out_cells


if __name__ == "__main__":
    test_source_contains_required_bytes_commit_capacity_alias_safe_default_capacity_wrapper()
    test_known_vectors_and_alias_overlap_path()
    test_diagnostics_no_partial_on_failure()
    test_error_paths_and_overflow()
    test_randomized_parity_against_explicit_checked_composition()
    print("ok")
