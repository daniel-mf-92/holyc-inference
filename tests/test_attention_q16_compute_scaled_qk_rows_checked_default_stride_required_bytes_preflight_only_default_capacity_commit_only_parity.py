#!/usr/bin/env python3
"""Parity harness for ...DefaultCapacityCommitOnlyParity (IQ-810)."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity,
)
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only,
)


def attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity(
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

    commit_last_q = [0]
    commit_last_k = [0]
    commit_last_out = [0]
    commit_req_q = [0]
    commit_req_k = [0]
    commit_req_out = [0]
    commit_req_q_bytes = [0]
    commit_req_k_bytes = [0]
    commit_req_out_bytes = [0]

    preflight_last_q = [0]
    preflight_last_k = [0]
    preflight_last_out = [0]
    preflight_req_q = [0]
    preflight_req_k = [0]
    preflight_req_out = [0]
    preflight_req_q_bytes = [0]
    preflight_req_k_bytes = [0]
    preflight_req_out_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only(
        q_rows_q16,
        query_row_count,
        k_rows_q16,
        token_count,
        head_dim,
        out_scores_q32,
        commit_last_q,
        commit_last_k,
        commit_last_out,
        commit_req_q,
        commit_req_k,
        commit_req_out,
        commit_req_q_bytes,
        commit_req_k_bytes,
        commit_req_out_bytes,
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
        preflight_last_q,
        preflight_last_k,
        preflight_last_out,
        preflight_req_q,
        preflight_req_k,
        preflight_req_out,
        preflight_req_q_bytes,
        preflight_req_k_bytes,
        preflight_req_out_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    if commit_last_q[0] != preflight_last_q[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_last_k[0] != preflight_last_k[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_last_out[0] != preflight_last_out[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    if commit_req_q[0] != preflight_req_q[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_req_k[0] != preflight_req_k[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_req_out[0] != preflight_req_out[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    if commit_req_q_bytes[0] != preflight_req_q_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_req_k_bytes[0] != preflight_req_k_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_req_out_bytes[0] != preflight_req_out_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    out_last_q_base_index[0] = commit_last_q[0]
    out_last_k_base_index[0] = commit_last_k[0]
    out_last_out_base_index[0] = commit_last_out[0]
    out_required_q_cells[0] = commit_req_q[0]
    out_required_k_cells[0] = commit_req_k[0]
    out_required_out_cells[0] = commit_req_out[0]
    out_required_q_bytes[0] = commit_req_q_bytes[0]
    out_required_k_bytes[0] = commit_req_k_bytes[0]
    out_required_out_bytes[0] = commit_req_out_bytes[0]
    return ATTN_Q16_OK


def test_source_contains_default_capacity_commit_only_parity_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStride"
        "RequiredBytesPreflightOnlyDefaultCapacityCommitOnlyParity("
    )
    assert signature in source
    body = source.split(signature, 1)[1]
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStride"
        "RequiredBytesPreflightOnlyDefaultCapacityCommitOnly("
    ) in body
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStride"
        "RequiredBytesPreflightOnlyDefaultCapacity("
    ) in body


def test_known_vector() -> None:
    qrc = 5
    tc = 4
    hd = 6
    q = [0] * (qrc * hd)
    k = [0] * (tc * hd)
    out = [0] * (qrc * tc)

    lq = [1]
    lk = [2]
    lo = [3]
    rq = [4]
    rk = [5]
    ro = [6]
    bq = [7]
    bk = [8]
    bo = [9]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity(
        q,
        qrc,
        k,
        tc,
        hd,
        out,
        lq,
        lk,
        lo,
        rq,
        rk,
        ro,
        bq,
        bk,
        bo,
    )
    assert err == ATTN_Q16_OK
    assert lq == [(qrc - 1) * hd]
    assert lk == [(tc - 1) * hd]
    assert lo == [(qrc - 1) * tc]
    assert rq == [qrc * hd]
    assert rk == [tc * hd]
    assert ro == [qrc * tc]
    assert bq == [qrc * hd * 8]
    assert bk == [tc * hd * 8]
    assert bo == [qrc * tc * 8]


def test_error_no_publish() -> None:
    q = [0] * 8
    k = [0] * 8
    out = [0] * 8

    lq = [101]
    lk = [102]
    lo = [103]
    rq = [104]
    rk = [105]
    ro = [106]
    bq = [107]
    bk = [108]
    bo = [109]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity(
        q,
        -1,
        k,
        1,
        1,
        out,
        lq,
        lk,
        lo,
        rq,
        rk,
        ro,
        bq,
        bk,
        bo,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert lq == [101]
    assert lk == [102]
    assert lo == [103]
    assert rq == [104]
    assert rk == [105]
    assert ro == [106]
    assert bq == [107]
    assert bk == [108]
    assert bo == [109]


def test_randomized_parity() -> None:
    rng = random.Random(810)
    for _ in range(1000):
        qrc = rng.randint(0, 48)
        tc = rng.randint(0, 48)
        hd = rng.randint(0, 64)

        q = [0] * max(1, rng.randint(1, 4096))
        k = [0] * max(1, rng.randint(1, 4096))
        out = [0] * max(1, rng.randint(1, 4096))

        got_lq = [9001]
        got_lk = [9002]
        got_lo = [9003]
        got_rq = [9004]
        got_rk = [9005]
        got_ro = [9006]
        got_bq = [9007]
        got_bk = [9008]
        got_bo = [9009]

        exp_lq = [8001]
        exp_lk = [8002]
        exp_lo = [8003]
        exp_rq = [8004]
        exp_rk = [8005]
        exp_ro = [8006]
        exp_bq = [8007]
        exp_bk = [8008]
        exp_bo = [8009]

        got_err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity(
            q,
            qrc,
            k,
            tc,
            hd,
            out,
            got_lq,
            got_lk,
            got_lo,
            got_rq,
            got_rk,
            got_ro,
            got_bq,
            got_bk,
            got_bo,
        )

        exp_err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity(
            q,
            qrc,
            k,
            tc,
            hd,
            out,
            exp_lq,
            exp_lk,
            exp_lo,
            exp_rq,
            exp_rk,
            exp_ro,
            exp_bq,
            exp_bk,
            exp_bo,
        )

        assert got_err == exp_err
        if got_err == ATTN_Q16_OK:
            assert got_lq == exp_lq
            assert got_lk == exp_lk
            assert got_lo == exp_lo
            assert got_rq == exp_rq
            assert got_rk == exp_rk
            assert got_ro == exp_ro
            assert got_bq == exp_bq
            assert got_bk == exp_bk
            assert got_bo == exp_bo
        else:
            assert got_lq == [9001]
            assert got_lk == [9002]
            assert got_lo == [9003]
            assert got_rq == [9004]
            assert got_rk == [9005]
            assert got_ro == [9006]
            assert got_bq == [9007]
            assert got_bk == [9008]
            assert got_bo == [9009]


def test_overflow() -> None:
    huge = 1 << 62
    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity(
        [0],
        huge,
        [0],
        2,
        huge,
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


if __name__ == "__main__":
    test_source_contains_default_capacity_commit_only_parity_helper()
    test_known_vector()
    test_error_no_publish()
    test_randomized_parity()
    test_overflow()
    print("ok")
