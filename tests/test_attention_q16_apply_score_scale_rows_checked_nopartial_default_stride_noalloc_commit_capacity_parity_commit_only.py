#!/usr/bin/env python3
"""Parity harness for ...NoAllocCommitCapacityParityCommitOnly (IQ-867)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_attention_q16_apply_score_scale_checked import (  # noqa: E402
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_ERR_OVERFLOW,
    ATTN_Q16_OK,
    I64_MAX,
    try_mul_i64_checked,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity import (  # noqa: E402
    attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity import (  # noqa: E402
    attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity,
)


def attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_preflight_only(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    parity_out_scores_q32,
    parity_out_scores_capacity: int,
    parity_staged_scores_q32,
    parity_staged_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        out_required_stage_cells is out_required_stage_bytes
        or out_required_stage_cells is out_required_out_cells
        or out_required_stage_bytes is out_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        in_scores_q32 is None
        or out_scores_q32 is None
        or staged_scores_q32 is None
        or parity_out_scores_q32 is None
        or parity_staged_scores_q32 is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        in_scores_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
        or parity_out_scores_capacity < 0
        or parity_staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        out_required_stage_cells is in_scores_q32
        or out_required_stage_cells is out_scores_q32
        or out_required_stage_cells is staged_scores_q32
        or out_required_stage_cells is parity_out_scores_q32
        or out_required_stage_cells is parity_staged_scores_q32
        or out_required_stage_bytes is in_scores_q32
        or out_required_stage_bytes is out_scores_q32
        or out_required_stage_bytes is staged_scores_q32
        or out_required_stage_bytes is parity_out_scores_q32
        or out_required_stage_bytes is parity_staged_scores_q32
        or out_required_out_cells is in_scores_q32
        or out_required_out_cells is out_scores_q32
        or out_required_out_cells is staged_scores_q32
        or out_required_out_cells is parity_out_scores_q32
        or out_required_out_cells is parity_staged_scores_q32
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot_row_count = row_count
    snapshot_token_count = token_count
    snapshot_score_scale_q16 = score_scale_q16
    snapshot_commit_stage_cell_capacity = commit_stage_cell_capacity
    snapshot_commit_stage_byte_capacity = commit_stage_byte_capacity

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
        parity_out_scores_q32,
        parity_out_scores_capacity,
        parity_staged_scores_q32,
        parity_staged_scores_capacity,
    )
    if err != ATTN_Q16_OK:
        return err

    staged_required_stage_cells = [0]
    staged_required_stage_bytes = [0]
    staged_required_out_cells = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        in_scores_q32,
        in_scores_capacity,
        snapshot_row_count,
        snapshot_token_count,
        out_scores_q32,
        out_scores_capacity,
        snapshot_commit_stage_cell_capacity,
        snapshot_commit_stage_byte_capacity,
        staged_required_stage_cells,
        staged_required_stage_bytes,
        staged_required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_cells = try_mul_i64_checked(snapshot_row_count, snapshot_token_count)
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_bytes = try_mul_i64_checked(recomputed_required_stage_cells, 8)
    if err != ATTN_Q16_OK:
        return err

    if snapshot_row_count != row_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_token_count != token_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_score_scale_q16 != score_scale_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_commit_stage_cell_capacity != commit_stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_commit_stage_byte_capacity != commit_stage_byte_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_required_stage_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_bytes[0] != recomputed_required_stage_bytes:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_required_stage_cells[0] > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_cells[0] > parity_staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] > parity_out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = staged_required_stage_bytes[0]
    out_required_out_cells[0] = staged_required_out_cells[0]
    return ATTN_Q16_OK


def attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_commit_only(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    parity_out_scores_q32,
    parity_out_scores_capacity: int,
    parity_staged_scores_q32,
    parity_staged_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        out_required_stage_cells is out_required_stage_bytes
        or out_required_stage_cells is out_required_out_cells
        or out_required_stage_bytes is out_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        in_scores_q32 is None
        or out_scores_q32 is None
        or staged_scores_q32 is None
        or parity_out_scores_q32 is None
        or parity_staged_scores_q32 is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        in_scores_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
        or parity_out_scores_capacity < 0
        or parity_staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        out_required_stage_cells is in_scores_q32
        or out_required_stage_cells is out_scores_q32
        or out_required_stage_cells is staged_scores_q32
        or out_required_stage_cells is parity_out_scores_q32
        or out_required_stage_cells is parity_staged_scores_q32
        or out_required_stage_bytes is in_scores_q32
        or out_required_stage_bytes is out_scores_q32
        or out_required_stage_bytes is staged_scores_q32
        or out_required_stage_bytes is parity_out_scores_q32
        or out_required_stage_bytes is parity_staged_scores_q32
        or out_required_out_cells is in_scores_q32
        or out_required_out_cells is out_scores_q32
        or out_required_out_cells is staged_scores_q32
        or out_required_out_cells is parity_out_scores_q32
        or out_required_out_cells is parity_staged_scores_q32
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot_row_count = row_count
    snapshot_token_count = token_count
    snapshot_score_scale_q16 = score_scale_q16
    snapshot_commit_stage_cell_capacity = commit_stage_cell_capacity
    snapshot_commit_stage_byte_capacity = commit_stage_byte_capacity
    snapshot_out_scores_capacity = out_scores_capacity
    snapshot_staged_scores_capacity = staged_scores_capacity
    snapshot_parity_out_scores_capacity = parity_out_scores_capacity
    snapshot_parity_staged_scores_capacity = parity_staged_scores_capacity

    staged_required_stage_cells = [0]
    staged_required_stage_bytes = [0]
    staged_required_out_cells = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
        parity_out_scores_q32,
        parity_out_scores_capacity,
        parity_staged_scores_q32,
        parity_staged_scores_capacity,
        staged_required_stage_cells,
        staged_required_stage_bytes,
        staged_required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_cells = try_mul_i64_checked(
        snapshot_row_count, snapshot_token_count
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_bytes = try_mul_i64_checked(
        recomputed_required_stage_cells, 8
    )
    if err != ATTN_Q16_OK:
        return err

    if snapshot_row_count != row_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_token_count != token_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_score_scale_q16 != score_scale_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_commit_stage_cell_capacity != commit_stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_commit_stage_byte_capacity != commit_stage_byte_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_out_scores_capacity != out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_staged_scores_capacity != staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_parity_out_scores_capacity != parity_out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_parity_staged_scores_capacity != parity_staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_required_stage_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_bytes[0] != recomputed_required_stage_bytes:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_required_stage_cells[0] > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_cells[0] > parity_staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] > parity_out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = staged_required_stage_bytes[0]
    out_required_out_cells[0] = staged_required_out_cells[0]
    return ATTN_Q16_OK


def explicit_commit_only_composition(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    parity_out_scores_q32,
    parity_out_scores_capacity: int,
    parity_staged_scores_q32,
    parity_staged_scores_capacity: int,
    out_required_stage_cells: list[int],
    out_required_stage_bytes: list[int],
    out_required_out_cells: list[int],
) -> int:
    staged_required_stage_cells = [0]
    staged_required_stage_bytes = [0]
    staged_required_out_cells = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
        parity_out_scores_q32,
        parity_out_scores_capacity,
        parity_staged_scores_q32,
        parity_staged_scores_capacity,
        staged_required_stage_cells,
        staged_required_stage_bytes,
        staged_required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_cells = try_mul_i64_checked(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_bytes = try_mul_i64_checked(
        recomputed_required_stage_cells, 8
    )
    if err != ATTN_Q16_OK:
        return err

    if staged_required_stage_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_bytes[0] != recomputed_required_stage_bytes:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_required_stage_cells[0] > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_cells[0] > parity_staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] > parity_out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = staged_required_stage_bytes[0]
    out_required_out_cells[0] = staged_required_out_cells[0]
    return ATTN_Q16_OK


def test_source_contains_commit_only_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityParityCommitOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityParityPreflightOnly("
        in body
    )
    assert "snapshot_row_count = row_count;" in body
    assert "snapshot_token_count = token_count;" in body
    assert "snapshot_score_scale_q16 = score_scale_q16;" in body
    assert "snapshot_commit_stage_cell_capacity = commit_stage_cell_capacity;" in body
    assert "snapshot_commit_stage_byte_capacity = commit_stage_byte_capacity;" in body


def test_known_vector_outputs() -> None:
    row_count = 4
    token_count = 3
    total = row_count * token_count

    in_scores = [0] * total
    out_scores = [123] * total
    staged_scores = [0] * total
    parity_out_scores = [0] * total
    parity_staged_scores = [0] * total

    got_stage_cells = [9]
    got_stage_bytes = [10]
    got_out_cells = [11]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_commit_only(
        in_scores,
        len(in_scores),
        row_count,
        token_count,
        2 * 65536,
        out_scores,
        len(out_scores),
        total,
        total * 8,
        staged_scores,
        len(staged_scores),
        parity_out_scores,
        len(parity_out_scores),
        parity_staged_scores,
        len(parity_staged_scores),
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
    )

    assert err == ATTN_Q16_OK
    assert got_stage_cells == [total]
    assert got_stage_bytes == [total * 8]
    assert got_out_cells == [total]
    assert out_scores == [123] * total


def test_null_alias_and_bounds() -> None:
    in_scores = [1, 2, 3, 4]
    out_scores = [0, 0, 0, 0]
    staged_scores = [0, 0, 0, 0]
    parity_out_scores = [0, 0, 0, 0]
    parity_staged_scores = [0, 0, 0, 0]

    out_stage_cells = [0]
    out_stage_bytes = [0]
    out_out_cells = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_commit_only(
        None,
        4,
        1,
        4,
        65536,
        out_scores,
        4,
        4,
        32,
        staged_scores,
        4,
        parity_out_scores,
        4,
        parity_staged_scores,
        4,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_commit_only(
        in_scores,
        4,
        1,
        4,
        65536,
        out_scores,
        4,
        4,
        32,
        staged_scores,
        4,
        parity_out_scores,
        4,
        parity_staged_scores,
        4,
        out_stage_cells,
        out_stage_cells,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_commit_only(
        in_scores,
        4,
        1,
        4,
        65536,
        out_scores,
        4,
        3,
        32,
        staged_scores,
        4,
        parity_out_scores,
        4,
        parity_staged_scores,
        4,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM


def test_overflow_paths() -> None:
    buf = [0] * 8
    out_a = [0]
    out_b = [0]
    out_c = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_commit_only(
        buf,
        len(buf),
        I64_MAX,
        2,
        65536,
        buf,
        len(buf),
        I64_MAX,
        I64_MAX,
        [0] * 8,
        8,
        [0] * 8,
        8,
        [0] * 8,
        8,
        out_a,
        out_b,
        out_c,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_commit_only_parity(seed: int = 867, trials: int = 200) -> None:
    rng = random.Random(seed)

    for _ in range(trials):
        row_count = rng.randint(0, 8)
        token_count = rng.randint(0, 8)
        total = row_count * token_count
        pad = rng.randint(0, 6)

        in_capacity = total + pad
        out_capacity = total + pad
        stage_capacity = total + pad
        parity_out_capacity = total + pad
        parity_stage_capacity = total + pad

        in_scores = [rng.randint(-300000, 300000) for _ in range(max(in_capacity, 1))]
        out_scores_a = [0] * max(out_capacity, 1)
        out_scores_b = [0] * max(out_capacity, 1)
        staged_a = [0] * max(stage_capacity, 1)
        staged_b = [0] * max(stage_capacity, 1)
        parity_out_a = [0] * max(parity_out_capacity, 1)
        parity_out_b = [0] * max(parity_out_capacity, 1)
        parity_staged_a = [0] * max(parity_stage_capacity, 1)
        parity_staged_b = [0] * max(parity_stage_capacity, 1)

        if rng.random() < 0.15 and total > 0:
            commit_stage_cell_capacity = total - 1
        else:
            commit_stage_cell_capacity = total
        commit_stage_byte_capacity = commit_stage_cell_capacity * 8

        score_scale_q16 = rng.randint(-(3 << 16), 3 << 16)

        got_a_cells = [0]
        got_a_bytes = [0]
        got_a_out = [0]
        got_b_cells = [0]
        got_b_bytes = [0]
        got_b_out = [0]

        err_a = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_commit_only(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            score_scale_q16,
            out_scores_a,
            out_capacity,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            staged_a,
            stage_capacity,
            parity_out_a,
            parity_out_capacity,
            parity_staged_a,
            parity_stage_capacity,
            got_a_cells,
            got_a_bytes,
            got_a_out,
        )

        err_b = explicit_commit_only_composition(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            score_scale_q16,
            out_scores_b,
            out_capacity,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            staged_b,
            stage_capacity,
            parity_out_b,
            parity_out_capacity,
            parity_staged_b,
            parity_stage_capacity,
            got_b_cells,
            got_b_bytes,
            got_b_out,
        )

        assert err_a == err_b
        if err_a == ATTN_Q16_OK:
            assert got_a_cells == got_b_cells
            assert got_a_bytes == got_b_bytes
            assert got_a_out == got_b_out


def main() -> None:
    test_source_contains_commit_only_wrapper()
    test_known_vector_outputs()
    test_null_alias_and_bounds()
    test_overflow_paths()
    test_randomized_commit_only_parity()
    print("ok")


if __name__ == "__main__":
    main()
