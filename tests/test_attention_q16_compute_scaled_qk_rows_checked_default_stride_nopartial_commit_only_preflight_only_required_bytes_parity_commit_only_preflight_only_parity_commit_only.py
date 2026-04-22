#!/usr/bin/env python3
"""Parity harness for ...RequiredBytesParityCommitOnlyPreflightOnlyParityCommitOnly (IQ-1042)."""

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

I64_MAX = (1 << 63) - 1


def _required_tuple(
    query_row_count: int,
    token_count: int,
    staged_scores_capacity: int,
    out_scores_capacity: int,
) -> tuple[int, int, int, int] | int:
    if query_row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count > 0 and token_count > (I64_MAX // query_row_count):
        return ATTN_Q16_ERR_OVERFLOW

    required = query_row_count * token_count
    if required > staged_scores_capacity or required > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if required > (I64_MAX // 8):
        return ATTN_Q16_ERR_OVERFLOW

    required_bytes = required * 8
    last_index = -1 if required == 0 else required - 1
    return required, required_bytes, required, last_index


def attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only(
    query_row_count: int,
    token_count: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_last_out_index: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
        or out_last_out_index is None
    ):
        return ATTN_Q16_ERR_NULL_PTR
    if staged_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    required_tuple = _required_tuple(
        query_row_count,
        token_count,
        staged_scores_capacity,
        out_scores_capacity,
    )
    if isinstance(required_tuple, int):
        return required_tuple

    out_required_stage_cells[0] = required_tuple[0]
    out_required_stage_bytes[0] = required_tuple[1]
    out_required_out_cells[0] = required_tuple[2]
    out_last_out_index[0] = required_tuple[3]
    return ATTN_Q16_OK


def attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity(
    query_row_count: int,
    token_count: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_last_out_index: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
        or out_last_out_index is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        out_required_stage_cells is out_required_stage_bytes
        or out_required_stage_cells is out_required_out_cells
        or out_required_stage_cells is out_last_out_index
        or out_required_stage_bytes is out_required_out_cells
        or out_required_stage_bytes is out_last_out_index
        or out_required_out_cells is out_last_out_index
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if query_row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot_query_row_count = query_row_count
    snapshot_token_count = token_count
    snapshot_staged_scores_capacity = staged_scores_capacity
    snapshot_out_scores_capacity = out_scores_capacity

    parity_required_stage_cells = [0]
    parity_required_stage_bytes = [0]
    parity_required_out_cells = [0]
    parity_last_out_index = [0]

    preflight_required_stage_cells = [0]
    preflight_required_stage_bytes = [0]
    preflight_required_out_cells = [0]
    preflight_last_out_index = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
        preflight_required_stage_cells,
        preflight_required_stage_bytes,
        preflight_required_out_cells,
        preflight_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
        parity_required_stage_cells,
        parity_required_stage_bytes,
        parity_required_out_cells,
        parity_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    if snapshot_query_row_count != query_row_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_token_count != token_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_staged_scores_capacity != staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_out_scores_capacity != out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if preflight_required_stage_cells[0] != parity_required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if preflight_required_stage_bytes[0] != parity_required_stage_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if preflight_required_out_cells[0] != parity_required_out_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if preflight_last_out_index[0] != parity_last_out_index[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = parity_required_stage_cells[0]
    out_required_stage_bytes[0] = parity_required_stage_bytes[0]
    out_required_out_cells[0] = parity_required_out_cells[0]
    out_last_out_index[0] = parity_last_out_index[0]
    return ATTN_Q16_OK


def attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity_commit_only(
    query_row_count: int,
    token_count: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_last_out_index: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
        or out_last_out_index is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        out_required_stage_cells is out_required_stage_bytes
        or out_required_stage_cells is out_required_out_cells
        or out_required_stage_cells is out_last_out_index
        or out_required_stage_bytes is out_required_out_cells
        or out_required_stage_bytes is out_last_out_index
        or out_required_out_cells is out_last_out_index
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if query_row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot_query_row_count = query_row_count
    snapshot_token_count = token_count
    snapshot_staged_scores_capacity = staged_scores_capacity
    snapshot_out_scores_capacity = out_scores_capacity

    parity_required_stage_cells = [0]
    parity_required_stage_bytes = [0]
    parity_required_out_cells = [0]
    parity_last_out_index = [0]

    preflight_required_stage_cells = [0]
    preflight_required_stage_bytes = [0]
    preflight_required_out_cells = [0]
    preflight_last_out_index = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
        parity_required_stage_cells,
        parity_required_stage_bytes,
        parity_required_out_cells,
        parity_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
        preflight_required_stage_cells,
        preflight_required_stage_bytes,
        preflight_required_out_cells,
        preflight_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    if snapshot_query_row_count != query_row_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_token_count != token_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_staged_scores_capacity != staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_out_scores_capacity != out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if parity_required_stage_cells[0] != preflight_required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if parity_required_stage_bytes[0] != preflight_required_stage_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if parity_required_out_cells[0] != preflight_required_out_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if parity_last_out_index[0] != preflight_last_out_index[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = parity_required_stage_cells[0]
    out_required_stage_bytes[0] = parity_required_stage_bytes[0]
    out_required_out_cells[0] = parity_required_out_cells[0]
    out_last_out_index[0] = parity_last_out_index[0]
    return ATTN_Q16_OK


def explicit_commit_only_parity_composition(
    query_row_count: int,
    token_count: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_stage_cells: list[int],
    out_required_stage_bytes: list[int],
    out_required_out_cells: list[int],
    out_last_out_index: list[int],
) -> int:
    parity_required_stage_cells = [0]
    parity_required_stage_bytes = [0]
    parity_required_out_cells = [0]
    parity_last_out_index = [0]

    preflight_required_stage_cells = [0]
    preflight_required_stage_bytes = [0]
    preflight_required_out_cells = [0]
    preflight_last_out_index = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
        parity_required_stage_cells,
        parity_required_stage_bytes,
        parity_required_out_cells,
        parity_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
        preflight_required_stage_cells,
        preflight_required_stage_bytes,
        preflight_required_out_cells,
        preflight_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    if parity_required_stage_cells[0] != preflight_required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if parity_required_stage_bytes[0] != preflight_required_stage_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if parity_required_out_cells[0] != preflight_required_out_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if parity_last_out_index[0] != preflight_last_out_index[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = parity_required_stage_cells[0]
    out_required_stage_bytes[0] = parity_required_stage_bytes[0]
    out_required_out_cells[0] = parity_required_out_cells[0]
    out_last_out_index[0] = parity_last_out_index[0]
    return ATTN_Q16_OK


def test_source_contains_required_bytes_parity_commit_only_preflight_only_parity_commit_only_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnlyPreflightOnlyParityCommitOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnlyPreflightOnlyParity(" in body
    )
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnlyPreflightOnly(" in body
    )
    assert "snapshot_query_row_count = query_row_count;" in body
    assert "snapshot_token_count = token_count;" in body


def test_known_vector_commit_only_parity_outputs() -> None:
    query_row_count = 8
    token_count = 5
    total = query_row_count * token_count

    staged = [0] * total
    out = [0] * total

    got_stage_cells = [11]
    got_stage_bytes = [12]
    got_out_cells = [13]
    got_last_out = [14]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity_commit_only(
        query_row_count,
        token_count,
        staged,
        len(staged),
        out,
        len(out),
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
        got_last_out,
    )

    assert err == ATTN_Q16_OK
    assert got_stage_cells == [total]
    assert got_stage_bytes == [total * 8]
    assert got_out_cells == [total]
    assert got_last_out == [total - 1]


def test_error_paths_preserve_outputs() -> None:
    staged = [0] * 12
    out = [0] * 12

    got_stage_cells = [301]
    got_stage_bytes = [302]
    got_out_cells = [303]
    got_last_out = [304]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity_commit_only(
        -1,
        2,
        staged,
        len(staged),
        out,
        len(out),
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
        got_last_out,
    )

    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert got_stage_cells == [301]
    assert got_stage_bytes == [302]
    assert got_out_cells == [303]
    assert got_last_out == [304]


def test_randomized_commit_only_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260422_1042)

    for _ in range(1600):
        query_row_count = rng.randint(0, 128)
        token_count = rng.randint(0, 128)

        required = query_row_count * token_count
        staged_cap = required + rng.randint(0, 8)
        out_cap = required + rng.randint(0, 8)

        staged = [0] * max(staged_cap, 1)
        out = [0] * max(out_cap, 1)

        got_stage_cells = [401]
        got_stage_bytes = [402]
        got_out_cells = [403]
        got_last_out = [404]

        exp_stage_cells = [501]
        exp_stage_bytes = [502]
        exp_out_cells = [503]
        exp_last_out = [504]

        err_new = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity_commit_only(
            query_row_count,
            token_count,
            staged,
            staged_cap,
            out,
            out_cap,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
            got_last_out,
        )

        err_ref = explicit_commit_only_parity_composition(
            query_row_count,
            token_count,
            staged,
            staged_cap,
            out,
            out_cap,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
            exp_last_out,
        )

        assert err_new == err_ref
        if err_new == ATTN_Q16_OK:
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
            assert got_out_cells == exp_out_cells
            assert got_last_out == exp_last_out
        else:
            assert got_stage_cells == [401]
            assert got_stage_bytes == [402]
            assert got_out_cells == [403]
            assert got_last_out == [404]


if __name__ == "__main__":
    test_source_contains_required_bytes_parity_commit_only_preflight_only_parity_commit_only_wrapper()
    test_known_vector_commit_only_parity_outputs()
    test_error_paths_preserve_outputs()
    test_randomized_commit_only_parity_vs_explicit_composition()
    print(
        "attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity_commit_only=ok"
    )
