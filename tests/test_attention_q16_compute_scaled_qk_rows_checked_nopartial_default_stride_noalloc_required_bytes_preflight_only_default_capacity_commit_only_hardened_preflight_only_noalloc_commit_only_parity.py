#!/usr/bin/env python3
"""Parity harness for ...HardenedPreflightOnlyNoAllocCommitOnlyParity."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc_commit_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc_commit_only,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc_commit_only_parity(
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

    if (
        out_required_q_cells is out_required_k_cells
        or out_required_q_cells is out_required_out_cells
        or out_required_q_cells is out_required_stage_cells
        or out_required_q_cells is out_required_stage_bytes
        or out_required_k_cells is out_required_out_cells
        or out_required_k_cells is out_required_stage_cells
        or out_required_k_cells is out_required_stage_bytes
        or out_required_out_cells is out_required_stage_cells
        or out_required_out_cells is out_required_stage_bytes
        or out_required_stage_cells is out_required_stage_bytes
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot = (
        q_rows_capacity,
        k_rows_capacity,
        out_scores_capacity,
        query_row_count,
        token_count,
        head_dim,
    )

    canonical_q = [0]
    canonical_k = [0]
    canonical_out = [0]
    canonical_stage_cells = [0]
    canonical_stage_bytes = [0]

    parity_q = [0]
    parity_k = [0]
    parity_out = [0]
    parity_stage_cells = [0]
    parity_stage_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc(
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

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc_commit_only(
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

    if snapshot != (
        q_rows_capacity,
        k_rows_capacity,
        out_scores_capacity,
        query_row_count,
        token_count,
        head_dim,
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        canonical_q[0] != parity_q[0]
        or canonical_k[0] != parity_k[0]
        or canonical_out[0] != parity_out[0]
        or canonical_stage_cells[0] != parity_stage_cells[0]
        or canonical_stage_bytes[0] != parity_stage_bytes[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_q_cells[0] = canonical_q[0]
    out_required_k_cells[0] = canonical_k[0]
    out_required_out_cells[0] = canonical_out[0]
    out_required_stage_cells[0] = canonical_stage_cells[0]
    out_required_stage_bytes[0] = canonical_stage_bytes[0]
    return ATTN_Q16_OK


def explicit_checked_composition(*args, **kwargs) -> int:
    (
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_required_q_cells,
        out_required_k_cells,
        out_required_out_cells,
        out_required_stage_cells,
        out_required_stage_bytes,
    ) = args

    if (
        out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        out_required_q_cells is out_required_k_cells
        or out_required_q_cells is out_required_out_cells
        or out_required_q_cells is out_required_stage_cells
        or out_required_q_cells is out_required_stage_bytes
        or out_required_k_cells is out_required_out_cells
        or out_required_k_cells is out_required_stage_cells
        or out_required_k_cells is out_required_stage_bytes
        or out_required_out_cells is out_required_stage_cells
        or out_required_out_cells is out_required_stage_bytes
        or out_required_stage_cells is out_required_stage_bytes
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot = (
        q_rows_capacity,
        k_rows_capacity,
        out_scores_capacity,
        query_row_count,
        token_count,
        head_dim,
    )

    canonical_q = [0]
    canonical_k = [0]
    canonical_out = [0]
    canonical_stage_cells = [0]
    canonical_stage_bytes = [0]

    parity_q = [0]
    parity_k = [0]
    parity_out = [0]
    parity_stage_cells = [0]
    parity_stage_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc_commit_only(
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

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc(
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

    if snapshot != (
        q_rows_capacity,
        k_rows_capacity,
        out_scores_capacity,
        query_row_count,
        token_count,
        head_dim,
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        canonical_q[0] != parity_q[0]
        or canonical_k[0] != parity_k[0]
        or canonical_out[0] != parity_out[0]
        or canonical_stage_cells[0] != parity_stage_cells[0]
        or canonical_stage_bytes[0] != parity_stage_bytes[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_q_cells[0] = parity_q[0]
    out_required_k_cells[0] = parity_k[0]
    out_required_out_cells[0] = parity_out[0]
    out_required_stage_cells[0] = parity_stage_cells[0]
    out_required_stage_bytes[0] = parity_stage_bytes[0]
    return ATTN_Q16_OK


def test_source_contains_hardened_preflight_only_noalloc_commit_only_parity_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesPreflightOnlyDefaultCapacityCommitOnlyHardenedPreflightOnlyNoAllocCommitOnlyParity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesPreflightOnlyDefaultCapacityCommitOnlyHardenedPreflightOnlyNoAlloc(" in body
    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesPreflightOnlyDefaultCapacityCommitOnlyHardenedPreflightOnlyNoAllocCommitOnly(" in body
    assert "canonical_required_q_cells" in body
    assert "parity_required_q_cells" in body


def test_known_vector_and_failure_sentinel_immutability() -> None:
    q_rows = [0] * 128
    k_rows = [0] * 128
    out_scores = [0] * 128

    out_q = [111]
    out_k = [112]
    out_out = [113]
    out_stage_cells = [114]
    out_stage_bytes = [115]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc_commit_only_parity(
        q_rows,
        len(q_rows),
        3,
        k_rows,
        len(k_rows),
        5,
        7,
        out_scores,
        len(out_scores),
        out_q,
        out_k,
        out_out,
        out_stage_cells,
        out_stage_bytes,
    )
    assert err == ATTN_Q16_OK
    assert out_q[0] == 21
    assert out_k[0] == 35
    assert out_out[0] == 15
    assert out_stage_cells[0] == 15
    assert out_stage_bytes[0] == 120

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc_commit_only_parity(
        None,
        len(q_rows),
        3,
        k_rows,
        len(k_rows),
        5,
        7,
        out_scores,
        len(out_scores),
        out_q,
        out_k,
        out_out,
        out_stage_cells,
        out_stage_bytes,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert out_q == [21]
    assert out_k == [35]
    assert out_out == [15]
    assert out_stage_cells == [15]
    assert out_stage_bytes == [120]


def test_null_bad_param_and_alias_contracts() -> None:
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc_commit_only_parity(
        [0],
        1,
        0,
        [0],
        1,
        0,
        0,
        [0],
        1,
        None,
        [0],
        [0],
        [0],
        [0],
    )
    assert err == ATTN_Q16_ERR_NULL_PTR

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc_commit_only_parity(
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
    assert err == ATTN_Q16_ERR_BAD_PARAM

    alias = [7]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc_commit_only_parity(
        [0],
        1,
        0,
        [0],
        1,
        0,
        0,
        [0],
        1,
        alias,
        alias,
        [0],
        [0],
        [0],
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert alias == [7]


def test_overflow_and_capacity_sentinel_vectors() -> None:
    i64_max = (1 << 63) - 1
    vectors = [
        (i64_max, i64_max, 2),
        (i64_max, 2, i64_max),
        (2, i64_max, i64_max),
        (1 << 62, 1 << 62, 8),
    ]

    for query_row_count, token_count, head_dim in vectors:
        out_q = [501]
        out_k = [502]
        out_out = [503]
        out_stage_cells = [504]
        out_stage_bytes = [505]

        err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc_commit_only_parity(
            [0],
            1,
            query_row_count,
            [0],
            1,
            token_count,
            head_dim,
            [0],
            1,
            out_q,
            out_k,
            out_out,
            out_stage_cells,
            out_stage_bytes,
        )
        assert err != ATTN_Q16_OK
        assert out_q == [501]
        assert out_k == [502]
        assert out_out == [503]
        assert out_stage_cells == [504]
        assert out_stage_bytes == [505]


def test_randomized_parity_and_no_partial() -> None:
    rng = random.Random(20260420_731)

    for _ in range(3000):
        query_row_count = rng.randint(0, 40)
        token_count = rng.randint(0, 40)
        head_dim = rng.randint(0, 40)

        q_capacity = rng.randint(0, 4096)
        k_capacity = rng.randint(0, 4096)
        out_capacity = rng.randint(0, 4096)

        q_rows = [0] * max(1, q_capacity)
        k_rows = [0] * max(1, k_capacity)
        out_scores = [0] * max(1, out_capacity)

        got_q = [901]
        got_k = [902]
        got_out = [903]
        got_stage_cells = [904]
        got_stage_bytes = [905]

        exp_q = [801]
        exp_k = [802]
        exp_out = [803]
        exp_stage_cells = [804]
        exp_stage_bytes = [805]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity_commit_only_hardened_preflight_only_noalloc_commit_only_parity(
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
            assert got_q == [901]
            assert got_k == [902]
            assert got_out == [903]
            assert got_stage_cells == [904]
            assert got_stage_bytes == [905]


if __name__ == "__main__":
    test_source_contains_hardened_preflight_only_noalloc_commit_only_parity_wrapper()
    test_known_vector_and_failure_sentinel_immutability()
    test_null_bad_param_and_alias_contracts()
    test_overflow_and_capacity_sentinel_vectors()
    test_randomized_parity_and_no_partial()
    print("ok")
