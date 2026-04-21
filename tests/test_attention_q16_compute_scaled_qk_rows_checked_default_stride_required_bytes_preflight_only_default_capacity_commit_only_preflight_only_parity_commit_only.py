#!/usr/bin/env python3
"""Parity harness for ...CommitOnlyPreflightOnlyParityCommitOnly (IQ-834)."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only,
)
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity,
)
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity,
)


def attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity(
    q_rows_q16,
    query_row_count: int,
    k_rows_q16,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_last_q_base_index: list[int] | None,
    out_last_k_base_index: list[int] | None,
    out_last_out_base_index: list[int] | None,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_q_bytes: list[int] | None,
    out_required_k_bytes: list[int] | None,
    out_required_out_bytes: list[int] | None,
) -> int:
    if (
        out_last_q_base_index is None
        or out_last_k_base_index is None
        or out_last_out_base_index is None
        or out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_q_bytes is None
        or out_required_k_bytes is None
        or out_required_out_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        out_last_q_base_index is out_last_k_base_index
        or out_last_q_base_index is out_last_out_base_index
        or out_last_q_base_index is out_required_q_cells
        or out_last_q_base_index is out_required_k_cells
        or out_last_q_base_index is out_required_out_cells
        or out_last_q_base_index is out_required_q_bytes
        or out_last_q_base_index is out_required_k_bytes
        or out_last_q_base_index is out_required_out_bytes
        or out_last_k_base_index is out_last_out_base_index
        or out_last_k_base_index is out_required_q_cells
        or out_last_k_base_index is out_required_k_cells
        or out_last_k_base_index is out_required_out_cells
        or out_last_k_base_index is out_required_q_bytes
        or out_last_k_base_index is out_required_k_bytes
        or out_last_k_base_index is out_required_out_bytes
        or out_last_out_base_index is out_required_q_cells
        or out_last_out_base_index is out_required_k_cells
        or out_last_out_base_index is out_required_out_cells
        or out_last_out_base_index is out_required_q_bytes
        or out_last_out_base_index is out_required_k_bytes
        or out_last_out_base_index is out_required_out_bytes
        or out_required_q_cells is out_required_k_cells
        or out_required_q_cells is out_required_out_cells
        or out_required_q_cells is out_required_q_bytes
        or out_required_q_cells is out_required_k_bytes
        or out_required_q_cells is out_required_out_bytes
        or out_required_k_cells is out_required_out_cells
        or out_required_k_cells is out_required_q_bytes
        or out_required_k_cells is out_required_k_bytes
        or out_required_k_cells is out_required_out_bytes
        or out_required_out_cells is out_required_q_bytes
        or out_required_out_cells is out_required_k_bytes
        or out_required_out_cells is out_required_out_bytes
        or out_required_q_bytes is out_required_k_bytes
        or out_required_q_bytes is out_required_out_bytes
        or out_required_k_bytes is out_required_out_bytes
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_last_q = [0]
    staged_last_k = [0]
    staged_last_out = [0]
    staged_req_q = [0]
    staged_req_k = [0]
    staged_req_out = [0]
    staged_req_q_bytes = [0]
    staged_req_k_bytes = [0]
    staged_req_out_bytes = [0]

    canonical_last_q = [0]
    canonical_last_k = [0]
    canonical_last_out = [0]
    canonical_req_q = [0]
    canonical_req_k = [0]
    canonical_req_out = [0]
    canonical_req_q_bytes = [0]
    canonical_req_k_bytes = [0]
    canonical_req_out_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only(
        q_rows_q16,
        query_row_count,
        k_rows_q16,
        token_count,
        head_dim,
        out_scores_q32,
        staged_last_q,
        staged_last_k,
        staged_last_out,
        staged_req_q,
        staged_req_k,
        staged_req_out,
        staged_req_q_bytes,
        staged_req_k_bytes,
        staged_req_out_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity(
        q_rows_q16,
        query_row_count,
        k_rows_q16,
        token_count,
        head_dim,
        out_scores_q32,
        canonical_last_q,
        canonical_last_k,
        canonical_last_out,
        canonical_req_q,
        canonical_req_k,
        canonical_req_out,
        canonical_req_q_bytes,
        canonical_req_k_bytes,
        canonical_req_out_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    if staged_last_q[0] != canonical_last_q[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_k[0] != canonical_last_k[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_out[0] != canonical_last_out[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_q[0] != canonical_req_q[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_k[0] != canonical_req_k[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_out[0] != canonical_req_out[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_q_bytes[0] != canonical_req_q_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_k_bytes[0] != canonical_req_k_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_out_bytes[0] != canonical_req_out_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    out_last_q_base_index[0] = staged_last_q[0]
    out_last_k_base_index[0] = staged_last_k[0]
    out_last_out_base_index[0] = staged_last_out[0]
    out_required_q_cells[0] = staged_req_q[0]
    out_required_k_cells[0] = staged_req_k[0]
    out_required_out_cells[0] = staged_req_out[0]
    out_required_q_bytes[0] = staged_req_q_bytes[0]
    out_required_k_bytes[0] = staged_req_k_bytes[0]
    out_required_out_bytes[0] = staged_req_out_bytes[0]
    return ATTN_Q16_OK


def attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity_commit_only(
    q_rows_q16,
    query_row_count: int,
    k_rows_q16,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_last_q_base_index: list[int] | None,
    out_last_k_base_index: list[int] | None,
    out_last_out_base_index: list[int] | None,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_q_bytes: list[int] | None,
    out_required_k_bytes: list[int] | None,
    out_required_out_bytes: list[int] | None,
) -> int:
    if (
        out_last_q_base_index is None
        or out_last_k_base_index is None
        or out_last_out_base_index is None
        or out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_q_bytes is None
        or out_required_k_bytes is None
        or out_required_out_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_last_q = [0]
    staged_last_k = [0]
    staged_last_out = [0]
    staged_req_q = [0]
    staged_req_k = [0]
    staged_req_out = [0]
    staged_req_q_bytes = [0]
    staged_req_k_bytes = [0]
    staged_req_out_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity(
        q_rows_q16,
        query_row_count,
        k_rows_q16,
        token_count,
        head_dim,
        out_scores_q32,
        staged_last_q,
        staged_last_k,
        staged_last_out,
        staged_req_q,
        staged_req_k,
        staged_req_out,
        staged_req_q_bytes,
        staged_req_k_bytes,
        staged_req_out_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    out_last_q_base_index[0] = staged_last_q[0]
    out_last_k_base_index[0] = staged_last_k[0]
    out_last_out_base_index[0] = staged_last_out[0]
    out_required_q_cells[0] = staged_req_q[0]
    out_required_k_cells[0] = staged_req_k[0]
    out_required_out_cells[0] = staged_req_out[0]
    out_required_q_bytes[0] = staged_req_q_bytes[0]
    out_required_k_bytes[0] = staged_req_k_bytes[0]
    out_required_out_bytes[0] = staged_req_out_bytes[0]
    return ATTN_Q16_OK


def explicit_commit_only_parity_composition(
    q_rows_q16,
    query_row_count: int,
    k_rows_q16,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_last_q_base_index: list[int] | None,
    out_last_k_base_index: list[int] | None,
    out_last_out_base_index: list[int] | None,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_q_bytes: list[int] | None,
    out_required_k_bytes: list[int] | None,
    out_required_out_bytes: list[int] | None,
) -> int:
    return attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only(
        q_rows_q16,
        query_row_count,
        k_rows_q16,
        token_count,
        head_dim,
        out_scores_q32,
        out_last_q_base_index,
        out_last_k_base_index,
        out_last_out_base_index,
        out_required_q_cells,
        out_required_k_cells,
        out_required_out_cells,
        out_required_q_bytes,
        out_required_k_bytes,
        out_required_out_bytes,
    )


def test_source_contains_commit_only_parity_commit_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStride"
        "RequiredBytesPreflightOnlyDefaultCapacityCommitOnlyPreflightOnlyParityCommitOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "snapshot_query_row_count = query_row_count;" in body
    assert "snapshot_token_count = token_count;" in body
    assert "snapshot_head_dim = head_dim;" in body
    assert "out_last_q_base_index == out_last_k_base_index" in body
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideRequiredBytesPreflightOnlyDefaultCapacityCommitOnlyPreflightOnlyParity("
        in body
    )
    assert "staged_required_q_cells != recomputed_required_q_cells" in body
    assert "staged_required_out_bytes != recomputed_required_out_bytes" in body
    assert "staged_last_out_base_index != recomputed_last_out_base_index" in body
    assert "*out_last_q_base_index = staged_last_q_base_index;" in body
    assert "*out_required_out_bytes = staged_required_out_bytes;" in body


def test_known_vector_matches_expected_geometry() -> None:
    query_row_count = 7
    token_count = 6
    head_dim = 5

    q_rows = [0] * (query_row_count * head_dim)
    k_rows = [0] * (token_count * head_dim)
    out_scores = [0] * (query_row_count * token_count)

    got_last_q = [101]
    got_last_k = [202]
    got_last_out = [303]
    got_req_q = [404]
    got_req_k = [505]
    got_req_out = [606]
    got_req_q_bytes = [707]
    got_req_k_bytes = [808]
    got_req_out_bytes = [909]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity_commit_only(
        q_rows,
        query_row_count,
        k_rows,
        token_count,
        head_dim,
        out_scores,
        got_last_q,
        got_last_k,
        got_last_out,
        got_req_q,
        got_req_k,
        got_req_out,
        got_req_q_bytes,
        got_req_k_bytes,
        got_req_out_bytes,
    )

    assert err == ATTN_Q16_OK
    assert got_last_q == [(query_row_count - 1) * head_dim]
    assert got_last_k == [(token_count - 1) * head_dim]
    assert got_last_out == [(query_row_count - 1) * token_count]
    assert got_req_q == [query_row_count * head_dim]
    assert got_req_k == [token_count * head_dim]
    assert got_req_out == [query_row_count * token_count]
    assert got_req_q_bytes == [query_row_count * head_dim * 8]
    assert got_req_k_bytes == [token_count * head_dim * 8]
    assert got_req_out_bytes == [query_row_count * token_count * 8]


def test_error_paths_preserve_outputs() -> None:
    last_q = [11]
    last_k = [22]
    last_out = [33]
    req_q = [44]
    req_k = [55]
    req_out = [66]
    req_q_bytes = [77]
    req_k_bytes = [88]
    req_out_bytes = [99]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity_commit_only(
        None,
        1,
        [0],
        1,
        1,
        [0],
        last_q,
        last_k,
        last_out,
        req_q,
        req_k,
        req_out,
        req_q_bytes,
        req_k_bytes,
        req_out_bytes,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert last_q == [11]
    assert last_k == [22]
    assert last_out == [33]
    assert req_q == [44]
    assert req_k == [55]
    assert req_out == [66]
    assert req_q_bytes == [77]
    assert req_k_bytes == [88]
    assert req_out_bytes == [99]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(834)

    for _ in range(1100):
        query_row_count = rng.randint(0, 40)
        token_count = rng.randint(0, 40)
        head_dim = rng.randint(0, 96)

        q_rows = [0] * max(1, rng.randint(1, 8192))
        k_rows = [0] * max(1, rng.randint(1, 8192))
        out_scores = [0] * max(1, rng.randint(1, 8192))

        got_last_q = [9101]
        got_last_k = [9102]
        got_last_out = [9103]
        got_req_q = [9104]
        got_req_k = [9105]
        got_req_out = [9106]
        got_req_q_bytes = [9107]
        got_req_k_bytes = [9108]
        got_req_out_bytes = [9109]

        exp_last_q = [8101]
        exp_last_k = [8102]
        exp_last_out = [8103]
        exp_req_q = [8104]
        exp_req_k = [8105]
        exp_req_out = [8106]
        exp_req_q_bytes = [8107]
        exp_req_k_bytes = [8108]
        exp_req_out_bytes = [8109]

        err_got = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity_commit_only(
            q_rows,
            query_row_count,
            k_rows,
            token_count,
            head_dim,
            out_scores,
            got_last_q,
            got_last_k,
            got_last_out,
            got_req_q,
            got_req_k,
            got_req_out,
            got_req_q_bytes,
            got_req_k_bytes,
            got_req_out_bytes,
        )

        err_exp = explicit_commit_only_parity_composition(
            q_rows,
            query_row_count,
            k_rows,
            token_count,
            head_dim,
            out_scores,
            exp_last_q,
            exp_last_k,
            exp_last_out,
            exp_req_q,
            exp_req_k,
            exp_req_out,
            exp_req_q_bytes,
            exp_req_k_bytes,
            exp_req_out_bytes,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_last_q == exp_last_q
            assert got_last_k == exp_last_k
            assert got_last_out == exp_last_out
            assert got_req_q == exp_req_q
            assert got_req_k == exp_req_k
            assert got_req_out == exp_req_out
            assert got_req_q_bytes == exp_req_q_bytes
            assert got_req_k_bytes == exp_req_k_bytes
            assert got_req_out_bytes == exp_req_out_bytes
        else:
            assert got_last_q == [9101]
            assert got_last_k == [9102]
            assert got_last_out == [9103]
            assert got_req_q == [9104]
            assert got_req_k == [9105]
            assert got_req_out == [9106]
            assert got_req_q_bytes == [9107]
            assert got_req_k_bytes == [9108]
            assert got_req_out_bytes == [9109]


def test_overflow_passthrough_preserves_outputs() -> None:
    huge = 1 << 62

    last_q = [1]
    last_k = [2]
    last_out = [3]
    req_q = [4]
    req_k = [5]
    req_out = [6]
    req_q_bytes = [7]
    req_k_bytes = [8]
    req_out_bytes = [9]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity_commit_only(
        [0],
        huge,
        [0],
        3,
        huge,
        [0],
        last_q,
        last_k,
        last_out,
        req_q,
        req_k,
        req_out,
        req_q_bytes,
        req_k_bytes,
        req_out_bytes,
    )

    assert err == ATTN_Q16_ERR_OVERFLOW
    assert last_q == [1]
    assert last_k == [2]
    assert last_out == [3]
    assert req_q == [4]
    assert req_k == [5]
    assert req_out == [6]
    assert req_q_bytes == [7]
    assert req_k_bytes == [8]
    assert req_out_bytes == [9]


def test_capacity_mismatch_vectors_match_explicit_composition() -> None:
    vectors = [
        (4, 3, 5, 1, 1, 1),
        (8, 7, 9, 16, 8, 24),
        (2, 11, 13, 32, 32, 7),
        (15, 4, 3, 5, 64, 40),
    ]

    for query_row_count, token_count, head_dim, q_cap, k_cap, out_cap in vectors:
        q_rows = [0] * q_cap
        k_rows = [0] * k_cap
        out_scores = [0] * out_cap

        got_last_q = [991]
        got_last_k = [992]
        got_last_out = [993]
        got_req_q = [994]
        got_req_k = [995]
        got_req_out = [996]
        got_req_q_bytes = [997]
        got_req_k_bytes = [998]
        got_req_out_bytes = [999]

        exp_last_q = [891]
        exp_last_k = [892]
        exp_last_out = [893]
        exp_req_q = [894]
        exp_req_k = [895]
        exp_req_out = [896]
        exp_req_q_bytes = [897]
        exp_req_k_bytes = [898]
        exp_req_out_bytes = [899]

        err_got = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity_commit_only(
            q_rows,
            query_row_count,
            k_rows,
            token_count,
            head_dim,
            out_scores,
            got_last_q,
            got_last_k,
            got_last_out,
            got_req_q,
            got_req_k,
            got_req_out,
            got_req_q_bytes,
            got_req_k_bytes,
            got_req_out_bytes,
        )

        err_exp = explicit_commit_only_parity_composition(
            q_rows,
            query_row_count,
            k_rows,
            token_count,
            head_dim,
            out_scores,
            exp_last_q,
            exp_last_k,
            exp_last_out,
            exp_req_q,
            exp_req_k,
            exp_req_out,
            exp_req_q_bytes,
            exp_req_k_bytes,
            exp_req_out_bytes,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_last_q == exp_last_q
            assert got_last_k == exp_last_k
            assert got_last_out == exp_last_out
            assert got_req_q == exp_req_q
            assert got_req_k == exp_req_k
            assert got_req_out == exp_req_out
            assert got_req_q_bytes == exp_req_q_bytes
            assert got_req_k_bytes == exp_req_k_bytes
            assert got_req_out_bytes == exp_req_out_bytes
        else:
            assert got_last_q == [991]
            assert got_last_k == [992]
            assert got_last_out == [993]
            assert got_req_q == [994]
            assert got_req_k == [995]
            assert got_req_out == [996]
            assert got_req_q_bytes == [997]
            assert got_req_k_bytes == [998]
            assert got_req_out_bytes == [999]


if __name__ == "__main__":
    test_source_contains_commit_only_parity_commit_only_helper()
    test_known_vector_matches_expected_geometry()
    test_error_paths_preserve_outputs()
    test_randomized_parity_vs_explicit_composition()
    test_overflow_passthrough_preserves_outputs()
    test_capacity_mismatch_vectors_match_explicit_composition()
    print("ok")
