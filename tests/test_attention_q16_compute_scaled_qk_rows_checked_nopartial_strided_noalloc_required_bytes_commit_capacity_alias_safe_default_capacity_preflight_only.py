#!/usr/bin/env python3
"""Parity harness for ...RequiredBytesCommitCapacityAliasSafeDefaultCapacityPreflightOnly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_ERR_OVERFLOW,
    ATTN_Q16_OK,
)
from test_attention_q16_compute_scaled_qk_rows_checked import try_mul_i64_checked
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
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

    err, commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, commit_stage_byte_capacity = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    req_q = [0]
    req_k = [0]
    req_out = [0]
    req_stage_cells = [0]
    req_stage_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only(
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
        staged_scores_q32,
        staged_scores_capacity,
        req_q,
        req_k,
        req_out,
        req_stage_cells,
        req_stage_bytes,
        q_base_addr=q_base_addr,
        k_base_addr=k_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )
    if err != ATTN_Q16_OK:
        return err

    out_commit_stage_cell_capacity[0] = commit_stage_cell_capacity
    out_commit_stage_byte_capacity[0] = commit_stage_byte_capacity
    out_required_q_cells[0] = req_q[0]
    out_required_k_cells[0] = req_k[0]
    out_required_out_cells[0] = req_out[0]
    out_required_stage_cells[0] = req_stage_cells[0]
    out_required_stage_bytes[0] = req_stage_bytes[0]
    return ATTN_Q16_OK


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

    err, commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, commit_stage_byte_capacity = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    req_q = [0]
    req_k = [0]
    req_out = [0]
    req_stage_cells = [0]
    req_stage_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only(
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
        staged_scores_q32,
        staged_scores_capacity,
        req_q,
        req_k,
        req_out,
        req_stage_cells,
        req_stage_bytes,
        q_base_addr=q_base_addr,
        k_base_addr=k_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )
    if err != ATTN_Q16_OK:
        return err

    out_commit_stage_cell_capacity[0] = commit_stage_cell_capacity
    out_commit_stage_byte_capacity[0] = commit_stage_byte_capacity
    out_required_q_cells[0] = req_q[0]
    out_required_k_cells[0] = req_k[0]
    out_required_out_cells[0] = req_out[0]
    out_required_stage_cells[0] = req_stage_cells[0]
    out_required_stage_bytes[0] = req_stage_bytes[0]
    return ATTN_Q16_OK


def test_source_contains_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacityAliasSafeDefaultCapacityPreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionTryMulI64Checked(query_row_count," in body
    assert "AttentionTryMulI64Checked(staged_scores_capacity," in body
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafeDefaultCapacityPreflightOnly(" in body
    )
    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocPreflightOnly(" in body
    assert "if (required_q_cells != canonical_required_q_cells)" in body
    assert "if (required_k_cells != canonical_required_k_cells)" in body
    assert "if (required_out_cells != canonical_required_out_cells)" in body
    assert "if (required_stage_cells != canonical_required_stage_cells)" in body
    assert "if (required_stage_bytes != canonical_required_stage_bytes)" in body
    assert "*out_commit_stage_cell_capacity = commit_stage_cell_capacity;" in body
    assert "*out_commit_stage_byte_capacity = commit_stage_byte_capacity;" in body
    assert "*out_required_q_cells =" in body
    assert "*out_required_k_cells =" in body
    assert "*out_required_out_cells =" in body
    assert "*out_required_stage_cells =" in body
    assert "*out_required_stage_bytes =" in body


def test_known_vectors_and_alias_rejection() -> None:
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

    err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
        q_rows,
        len(q_rows),
        3,
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
    assert got_q_cells == exp_q_cells == [22]
    assert got_k_cells == exp_k_cells == [32]
    assert got_out_cells == exp_out_cells == [16]
    assert got_stage_cells == exp_stage_cells == [12]
    assert got_stage_bytes == exp_stage_bytes == [96]

    err_alias = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
        q_rows,
        len(q_rows),
        3,
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
    assert err_alias != ATTN_Q16_OK


def test_overflow_and_null_contracts() -> None:
    q_rows = [0] * 8
    k_rows = [0] * 8
    out_scores = [0] * 8

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
            q_rows,
            len(q_rows),
            1,
            1,
            k_rows,
            len(k_rows),
            1,
            1,
            1,
            out_scores,
            len(out_scores),
            1,
            [0] * 8,
            8,
            None,
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    i64_max = (1 << 63) - 1
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
            q_rows,
            len(q_rows),
            i64_max,
            1,
            k_rows,
            len(k_rows),
            2,
            1,
            1,
            out_scores,
            len(out_scores),
            1,
            [0] * 8,
            8,
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
        )
        == ATTN_Q16_ERR_OVERFLOW
    )


def test_no_partial_outputs_on_error_paths() -> None:
    q_rows = [0] * 12
    k_rows = [0] * 12
    out_scores = [0] * 12
    staged_scores = [0] * 12

    sentinels = [101, 102, 103, 104, 105, 106, 107]
    outs = [[v] for v in sentinels]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
        q_rows,
        len(q_rows),
        (1 << 63) - 1,
        1,
        k_rows,
        len(k_rows),
        2,
        1,
        1,
        out_scores,
        len(out_scores),
        1,
        staged_scores,
        len(staged_scores),
        outs[0],
        outs[1],
        outs[2],
        outs[3],
        outs[4],
        outs[5],
        outs[6],
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert [x[0] for x in outs] == sentinels

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
        q_rows,
        len(q_rows),
        2,
        4,
        k_rows,
        len(k_rows),
        2,
        4,
        3,
        out_scores,
        len(out_scores),
        2,
        staged_scores,
        len(staged_scores),
        outs[0],
        outs[1],
        outs[2],
        outs[3],
        outs[4],
        outs[5],
        outs[6],
        stage_base_addr=0x100004,
    )
    assert err != ATTN_Q16_OK
    assert [x[0] for x in outs] == sentinels


def test_randomized_parity() -> None:
    rng = random.Random(20260420_655)

    for _ in range(2500):
        query_row_count = rng.randint(0, 7)
        token_count = rng.randint(0, 9)
        head_dim = rng.randint(0, 11)
        query_row_stride_q16 = rng.randint(0, 14)
        k_row_stride_q16 = rng.randint(0, 14)
        out_row_stride = rng.randint(0, 14)

        if query_row_count > 0 and head_dim > 0 and query_row_stride_q16 < head_dim:
            query_row_stride_q16 = head_dim + rng.randint(0, 3)
        if token_count > 0 and head_dim > 0 and k_row_stride_q16 < head_dim:
            k_row_stride_q16 = head_dim + rng.randint(0, 3)
        if query_row_count > 0 and token_count > 0 and out_row_stride < token_count:
            out_row_stride = token_count + rng.randint(0, 4)

        q_need = (
            0
            if query_row_count == 0 or head_dim == 0
            else (query_row_count - 1) * query_row_stride_q16 + head_dim
        )
        k_need = (
            0
            if token_count == 0 or head_dim == 0
            else (token_count - 1) * k_row_stride_q16 + head_dim
        )
        out_need = (
            0
            if query_row_count == 0 or token_count == 0
            else (query_row_count - 1) * out_row_stride + token_count
        )
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

        got_commit_cells = [700]
        got_commit_bytes = [701]
        got_q_cells = [702]
        got_k_cells = [703]
        got_out_cells = [704]
        got_stage_cells = [705]
        got_stage_bytes = [706]

        exp_commit_cells = [700]
        exp_commit_bytes = [701]
        exp_q_cells = [702]
        exp_k_cells = [703]
        exp_out_cells = [704]
        exp_stage_cells = [705]
        exp_stage_bytes = [706]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
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
    test_source_contains_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_helper()
    test_known_vectors_and_alias_rejection()
    test_overflow_and_null_contracts()
    test_no_partial_outputs_on_error_paths()
    test_randomized_parity()
    print("ok")
