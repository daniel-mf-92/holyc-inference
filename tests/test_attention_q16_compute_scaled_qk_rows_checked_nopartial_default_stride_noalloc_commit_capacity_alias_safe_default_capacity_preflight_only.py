#!/usr/bin/env python3
"""Parity harness for ...DefaultStrideNoAllocCommitCapacityAliasSafeDefaultCapacityPreflightOnly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    *,
    q_base_addr: int = 0x100000,
    k_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    if (
        out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None or staged_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if (
        q_rows_capacity < 0
        or k_rows_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    default_query_row_stride_q16 = head_dim
    default_k_row_stride_q16 = head_dim
    default_out_row_stride = token_count

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        default_query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        default_out_row_stride,
        staged_scores_q32,
        staged_scores_capacity,
        out_required_q_cells,
        out_required_k_cells,
        out_required_out_cells,
        out_required_stage_cells,
        out_required_stage_bytes,
        q_base_addr=q_base_addr,
        k_base_addr=k_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )


def explicit_checked_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    *,
    q_base_addr: int = 0x100000,
    k_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    default_query_row_stride_q16 = head_dim
    default_k_row_stride_q16 = head_dim
    default_out_row_stride = token_count

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        default_query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        default_out_row_stride,
        staged_scores_q32,
        staged_scores_capacity,
        out_required_q_cells,
        out_required_k_cells,
        out_required_out_cells,
        out_required_stage_cells,
        out_required_stage_bytes,
        q_base_addr=q_base_addr,
        k_base_addr=k_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )


def test_source_contains_default_stride_alias_safe_default_capacity_preflight_only_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityAliasSafeDefaultCapacityPreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "default_query_row_stride_q16 = head_dim;" in body
    assert "default_k_row_stride_q16 = head_dim;" in body
    assert "default_out_row_stride = token_count;" in body
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafeDefaultCapacityPreflightOnly("
        in body
    )


def test_known_vector_and_alias_rejection() -> None:
    q_rows = [0] * 64
    k_rows = [0] * 96
    out_scores = [0] * 64
    staged_scores = [0] * 64

    got_q_cells = [11]
    got_k_cells = [12]
    got_out_cells = [13]
    got_stage_cells = [14]
    got_stage_bytes = [15]

    exp_q_cells = [21]
    exp_k_cells = [22]
    exp_out_cells = [23]
    exp_stage_cells = [24]
    exp_stage_bytes = [25]

    err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only(
        q_rows,
        len(q_rows),
        4,
        k_rows,
        len(k_rows),
        5,
        6,
        out_scores,
        len(out_scores),
        staged_scores,
        len(staged_scores),
        got_q_cells,
        got_k_cells,
        got_out_cells,
        got_stage_cells,
        got_stage_bytes,
    )

    err_exp = explicit_checked_composition(
        q_rows,
        len(q_rows),
        4,
        k_rows,
        len(k_rows),
        5,
        6,
        out_scores,
        len(out_scores),
        staged_scores,
        len(staged_scores),
        exp_q_cells,
        exp_k_cells,
        exp_out_cells,
        exp_stage_cells,
        exp_stage_bytes,
    )

    assert err_got == err_exp == ATTN_Q16_OK
    assert got_q_cells == exp_q_cells == [24]
    assert got_k_cells == exp_k_cells == [30]
    assert got_out_cells == exp_out_cells == [20]
    assert got_stage_cells == exp_stage_cells == [20]
    assert got_stage_bytes == exp_stage_bytes == [160]

    err_alias = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only(
        q_rows,
        len(q_rows),
        4,
        k_rows,
        len(k_rows),
        5,
        6,
        out_scores,
        len(out_scores),
        staged_scores,
        len(staged_scores),
        [0],
        [0],
        [0],
        [0],
        [0],
        stage_base_addr=0x100000,
    )
    assert err_alias == ATTN_Q16_ERR_BAD_PARAM


def test_null_and_bad_param_contracts() -> None:
    q_rows = [0] * 32
    k_rows = [0] * 32
    out_scores = [0] * 32
    staged_scores = [0] * 32

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only(
            q_rows,
            len(q_rows),
            2,
            k_rows,
            len(k_rows),
            2,
            4,
            out_scores,
            len(out_scores),
            staged_scores,
            len(staged_scores),
            None,
            [0],
            [0],
            [0],
            [0],
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only(
            q_rows,
            -1,
            2,
            k_rows,
            len(k_rows),
            2,
            4,
            out_scores,
            len(out_scores),
            staged_scores,
            len(staged_scores),
            [0],
            [0],
            [0],
            [0],
            [0],
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )


def test_randomized_parity() -> None:
    rng = random.Random(669)

    for _ in range(300):
        query_row_count = rng.randint(0, 8)
        token_count = rng.randint(0, 8)
        head_dim = rng.randint(0, 8)

        q_capacity = rng.randint(0, 200)
        k_capacity = rng.randint(0, 220)
        out_capacity = rng.randint(0, 200)
        stage_capacity = rng.randint(0, 220)

        q_rows = [0] * max(1, q_capacity)
        k_rows = [0] * max(1, k_capacity)
        out_scores = [0] * max(1, out_capacity)
        staged_scores = [0] * max(1, stage_capacity)

        q_addr = 0x100000 + rng.randint(0, 4096)
        k_addr = 0x200000 + rng.randint(0, 4096)
        out_addr = 0x300000 + rng.randint(0, 4096)
        stage_addr = 0x400000 + rng.randint(0, 4096)

        if rng.random() < 0.25:
            stage_addr = q_addr
        elif rng.random() < 0.25:
            stage_addr = k_addr
        elif rng.random() < 0.25:
            stage_addr = out_addr

        got_q_cells = [111]
        got_k_cells = [112]
        got_out_cells = [113]
        got_stage_cells = [114]
        got_stage_bytes = [115]

        exp_q_cells = [211]
        exp_k_cells = [212]
        exp_out_cells = [213]
        exp_stage_cells = [214]
        exp_stage_bytes = [215]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            staged_scores,
            stage_capacity,
            got_q_cells,
            got_k_cells,
            got_out_cells,
            got_stage_cells,
            got_stage_bytes,
            q_base_addr=q_addr,
            k_base_addr=k_addr,
            out_base_addr=out_addr,
            stage_base_addr=stage_addr,
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
            staged_scores,
            stage_capacity,
            exp_q_cells,
            exp_k_cells,
            exp_out_cells,
            exp_stage_cells,
            exp_stage_bytes,
            q_base_addr=q_addr,
            k_base_addr=k_addr,
            out_base_addr=out_addr,
            stage_base_addr=stage_addr,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_q_cells == exp_q_cells
            assert got_k_cells == exp_k_cells
            assert got_out_cells == exp_out_cells
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
        else:
            assert got_q_cells == [111]
            assert got_k_cells == [112]
            assert got_out_cells == [113]
            assert got_stage_cells == [114]
            assert got_stage_bytes == [115]


if __name__ == "__main__":
    test_source_contains_default_stride_alias_safe_default_capacity_preflight_only_wrapper()
    test_known_vector_and_alias_rejection()
    test_null_and_bad_param_contracts()
    test_randomized_parity()
    print("ok")
