#!/usr/bin/env python3
"""Parity harness for ...DefaultStrideNoAllocRequiredBytesCommitCapacityAliasSafeDefaultCapacity."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity(
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
    out_commit_stage_cell_capacity: list[int] | None,
    out_commit_stage_byte_capacity: list[int] | None,
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
        out_commit_stage_cell_capacity is None
        or out_commit_stage_byte_capacity is None
        or out_required_q_cells is None
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

    tmp_commit_cells = [0]
    tmp_commit_bytes = [0]
    tmp_q_cells = [0]
    tmp_k_cells = [0]
    tmp_out_cells = [0]
    tmp_stage_cells = [0]
    tmp_stage_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        staged_scores_q32,
        staged_scores_capacity,
        tmp_commit_cells,
        tmp_commit_bytes,
        tmp_q_cells,
        tmp_k_cells,
        tmp_out_cells,
        tmp_stage_cells,
        tmp_stage_bytes,
        q_base_addr=q_base_addr,
        k_base_addr=k_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )
    if err != ATTN_Q16_OK:
        return err

    out_commit_stage_cell_capacity[0] = tmp_commit_cells[0]
    out_commit_stage_byte_capacity[0] = tmp_commit_bytes[0]
    out_required_q_cells[0] = tmp_q_cells[0]
    out_required_k_cells[0] = tmp_k_cells[0]
    out_required_out_cells[0] = tmp_out_cells[0]
    out_required_stage_cells[0] = tmp_stage_cells[0]
    out_required_stage_bytes[0] = tmp_stage_bytes[0]
    return ATTN_Q16_OK


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
    out_commit_stage_cell_capacity: list[int] | None,
    out_commit_stage_byte_capacity: list[int] | None,
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
    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        staged_scores_q32,
        staged_scores_capacity,
        out_commit_stage_cell_capacity,
        out_commit_stage_byte_capacity,
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


def test_source_contains_default_stride_required_bytes_commit_capacity_alias_safe_default_capacity_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacityAliasSafeDefaultCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacityAliasSafeDefaultCapacityPreflightOnly(" in body
    assert "*out_commit_stage_cell_capacity = commit_stage_cell_capacity;" in body
    assert "*out_commit_stage_byte_capacity = commit_stage_byte_capacity;" in body
    assert "*out_required_q_cells = required_q_cells;" in body
    assert "*out_required_k_cells = required_k_cells;" in body
    assert "*out_required_out_cells = required_out_cells;" in body
    assert "*out_required_stage_cells = required_stage_cells;" in body
    assert "*out_required_stage_bytes = required_stage_bytes;" in body


def test_known_vector_and_alias_rejection() -> None:
    q_rows = [0] * 48
    k_rows = [0] * 64
    out_scores = [0] * 32
    staged_scores = [0] * 32

    got_commit_cells = [11]
    got_commit_bytes = [22]
    got_q_cells = [33]
    got_k_cells = [44]
    got_out_cells = [55]
    got_stage_cells = [66]
    got_stage_bytes = [77]

    exp_commit_cells = [88]
    exp_commit_bytes = [99]
    exp_q_cells = [111]
    exp_k_cells = [222]
    exp_out_cells = [333]
    exp_stage_cells = [444]
    exp_stage_bytes = [555]

    err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity(
        q_rows,
        len(q_rows),
        3,
        k_rows,
        len(k_rows),
        4,
        6,
        out_scores,
        len(out_scores),
        staged_scores,
        len(staged_scores),
        got_commit_cells,
        got_commit_bytes,
        got_q_cells,
        got_k_cells,
        got_out_cells,
        got_stage_cells,
        got_stage_bytes,
    )

    err_exp = explicit_checked_composition(
        q_rows,
        len(q_rows),
        3,
        k_rows,
        len(k_rows),
        4,
        6,
        out_scores,
        len(out_scores),
        staged_scores,
        len(staged_scores),
        exp_commit_cells,
        exp_commit_bytes,
        exp_q_cells,
        exp_k_cells,
        exp_out_cells,
        exp_stage_cells,
        exp_stage_bytes,
    )

    assert err_got == err_exp == ATTN_Q16_OK
    assert got_commit_cells == exp_commit_cells == [12]
    assert got_commit_bytes == exp_commit_bytes == [256]
    assert got_q_cells == exp_q_cells == [18]
    assert got_k_cells == exp_k_cells == [24]
    assert got_out_cells == exp_out_cells == [12]
    assert got_stage_cells == exp_stage_cells == [12]
    assert got_stage_bytes == exp_stage_bytes == [96]

    err_alias = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity(
        q_rows,
        len(q_rows),
        3,
        k_rows,
        len(k_rows),
        4,
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
        [0],
        [0],
        stage_base_addr=0x100010,
    )
    assert err_alias == ATTN_Q16_ERR_BAD_PARAM


def test_randomized_parity() -> None:
    rng = random.Random(20260420_674)

    for _ in range(2500):
        query_row_count = rng.randint(0, 7)
        token_count = rng.randint(0, 9)
        head_dim = rng.randint(0, 11)

        q_need = 0 if query_row_count == 0 or head_dim == 0 else (query_row_count - 1) * head_dim + head_dim
        k_need = 0 if token_count == 0 or head_dim == 0 else (token_count - 1) * head_dim + head_dim
        out_need = 0 if query_row_count == 0 or token_count == 0 else (query_row_count - 1) * token_count + token_count
        stage_need = query_row_count * token_count

        q_capacity = max(0, q_need + rng.randint(-2, 3))
        k_capacity = max(0, k_need + rng.randint(-2, 3))
        out_capacity = max(0, out_need + rng.randint(-2, 3))
        stage_capacity = max(0, stage_need + rng.randint(-2, 3))

        q_rows = [0] * q_capacity
        k_rows = [0] * k_capacity
        out_scores = [0] * out_capacity
        staged_scores = [0] * stage_capacity

        if rng.randint(0, 9) < 3:
            stage_base = rng.choice([0x100000, 0x200000, 0x300000]) + rng.randint(0, 24)
        else:
            stage_base = 0x400000 + rng.randint(0, 256)

        got_commit_cells = [900]
        got_commit_bytes = [901]
        got_q_cells = [902]
        got_k_cells = [903]
        got_out_cells = [904]
        got_stage_cells = [905]
        got_stage_bytes = [906]

        exp_commit_cells = [900]
        exp_commit_bytes = [901]
        exp_q_cells = [902]
        exp_k_cells = [903]
        exp_out_cells = [904]
        exp_stage_cells = [905]
        exp_stage_bytes = [906]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity(
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
            got_commit_cells,
            got_commit_bytes,
            got_q_cells,
            got_k_cells,
            got_out_cells,
            got_stage_cells,
            got_stage_bytes,
            q_base_addr=0x100000,
            k_base_addr=0x200000,
            out_base_addr=0x300000,
            stage_base_addr=stage_base,
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
            exp_commit_cells,
            exp_commit_bytes,
            exp_q_cells,
            exp_k_cells,
            exp_out_cells,
            exp_stage_cells,
            exp_stage_bytes,
            q_base_addr=0x100000,
            k_base_addr=0x200000,
            out_base_addr=0x300000,
            stage_base_addr=stage_base,
        )

        assert err_got == err_exp
        assert got_commit_cells == exp_commit_cells
        assert got_commit_bytes == exp_commit_bytes
        assert got_q_cells == exp_q_cells
        assert got_k_cells == exp_k_cells
        assert got_out_cells == exp_out_cells
        assert got_stage_cells == exp_stage_cells
        assert got_stage_bytes == exp_stage_bytes


if __name__ == "__main__":
    test_source_contains_default_stride_required_bytes_commit_capacity_alias_safe_default_capacity_wrapper()
    test_known_vector_and_alias_rejection()
    test_randomized_parity()
    print("ok")
