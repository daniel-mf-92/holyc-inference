#!/usr/bin/env python3
"""Parity harness for AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideCommitOnly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
    try_add_i64_checked,
    try_mul_i64_checked,
)


def attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_commit_only(
    row_count: int,
    token_count: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    if staged_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    default_row_stride = token_count

    err, required_stage_cells = try_mul_i64_checked(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, required_out_cells = try_mul_i64_checked(row_count - 1, default_row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, token_count)
    if err != ATTN_Q16_OK:
        return err

    if required_stage_cells > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    for row_index in range(row_count):
        err, row_base = try_mul_i64_checked(row_index, default_row_stride)
        if err != ATTN_Q16_OK:
            return err

        err, stage_row_base = try_mul_i64_checked(row_index, token_count)
        if err != ATTN_Q16_OK:
            return err

        for token_index in range(token_count):
            err, _ = try_add_i64_checked(row_base, token_index)
            if err != ATTN_Q16_OK:
                return err

            err, _ = try_add_i64_checked(stage_row_base, token_index)
            if err != ATTN_Q16_OK:
                return err

    for row_index in range(row_count):
        err, row_base = try_mul_i64_checked(row_index, default_row_stride)
        if err != ATTN_Q16_OK:
            return err

        err, stage_row_base = try_mul_i64_checked(row_index, token_count)
        if err != ATTN_Q16_OK:
            return err

        for token_index in range(token_count):
            err, out_index = try_add_i64_checked(row_base, token_index)
            if err != ATTN_Q16_OK:
                return err

            err, stage_index = try_add_i64_checked(stage_row_base, token_index)
            if err != ATTN_Q16_OK:
                return err

            out_scores_q32[out_index] = staged_scores_q32[stage_index]

    return ATTN_Q16_OK


def explicit_checked_copy_loops(
    row_count: int,
    token_count: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    if staged_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    err, required_stage_cells = try_mul_i64_checked(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, required_out_cells = try_mul_i64_checked(row_count - 1, token_count)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, token_count)
    if err != ATTN_Q16_OK:
        return err

    if required_stage_cells > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    for row_index in range(row_count):
        err, row_base = try_mul_i64_checked(row_index, token_count)
        if err != ATTN_Q16_OK:
            return err

        for token_index in range(token_count):
            err, out_index = try_add_i64_checked(row_base, token_index)
            if err != ATTN_Q16_OK:
                return err

            err, stage_index = try_add_i64_checked(row_base, token_index)
            if err != ATTN_Q16_OK:
                return err

            out_scores_q32[out_index] = staged_scores_q32[stage_index]

    return ATTN_Q16_OK


def test_source_contains_commit_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideCommitOnly("
    )
    assert signature in source

    body = source.split(signature, 1)[1]
    assert "default_row_stride = token_count;" in body
    assert "required_stage_cells" in body
    assert "required_out_cells" in body
    assert "for (row_index = 0; row_index < row_count; row_index++)" in body
    assert "for (token_index = 0; token_index < token_count; token_index++)" in body
    assert "out_scores_q32[out_index] = staged_scores_q32[stage_index];" in body



def test_noalloc_wrapper_uses_commit_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAlloc("
    assert signature in source
    body = source.rsplit(signature, 1)[1].split(
        "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideCommitOnly(", 1
    )[0]
    assert "AttentionQ16ApplyScoreScaleRowsChecked(in_scores_q32," in body
    assert "return AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideCommitOnly(" in body


def test_known_vector_matches_explicit_checked_copy() -> None:
    row_count = 3
    token_count = 4

    staged = [
        101,
        -202,
        303,
        -404,
        505,
        -606,
        707,
        -808,
        909,
        -1001,
        1102,
        -1203,
    ]
    out_new = [0x7A7A] * (row_count * token_count)
    out_ref = out_new.copy()

    err_new = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_commit_only(
        row_count,
        token_count,
        staged,
        len(staged),
        out_new,
        len(out_new),
    )
    err_ref = explicit_checked_copy_loops(
        row_count,
        token_count,
        staged,
        len(staged),
        out_ref,
        len(out_ref),
    )

    assert err_new == err_ref == ATTN_Q16_OK
    assert out_new == out_ref == staged


def test_error_paths_preserve_output() -> None:
    staged = [1, 2, 3, 4, 5, 6]
    out_seed = [5150] * 6

    out_new = out_seed.copy()
    out_ref = out_seed.copy()
    err_new = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_commit_only(
        2,
        3,
        staged,
        len(staged),
        out_new,
        5,
    )
    err_ref = explicit_checked_copy_loops(2, 3, staged, len(staged), out_ref, 5)
    assert err_new == err_ref == ATTN_Q16_ERR_BAD_PARAM
    assert out_new == out_ref == out_seed

    out_new = out_seed.copy()
    out_ref = out_seed.copy()
    err_new = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_commit_only(
        1,
        -1,
        staged,
        len(staged),
        out_new,
        len(out_new),
    )
    err_ref = explicit_checked_copy_loops(1, -1, staged, len(staged), out_ref, len(out_ref))
    assert err_new == err_ref == ATTN_Q16_ERR_BAD_PARAM
    assert out_new == out_ref == out_seed

    out_new = out_seed.copy()
    out_ref = out_seed.copy()
    err_new = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_commit_only(
        1,
        6,
        None,
        len(staged),
        out_new,
        len(out_new),
    )
    err_ref = explicit_checked_copy_loops(1, 6, None, len(staged), out_ref, len(out_ref))
    assert err_new == err_ref == ATTN_Q16_ERR_NULL_PTR
    assert out_new == out_ref == out_seed



def test_randomized_adversarial_parity() -> None:
    rng = random.Random(592)

    for _ in range(300):
        row_count = rng.randint(0, 9)
        token_count = rng.randint(0, 9)

        total_cells = row_count * token_count
        if total_cells == 0:
            total_cells = 1

        staged = [rng.randint(-(1 << 30), (1 << 30)) for _ in range(total_cells)]

        staged_capacity = len(staged)
        out_capacity = row_count * token_count
        if out_capacity == 0:
            out_capacity = 1

        if rng.random() < 0.15 and staged_capacity > 0:
            staged_capacity -= 1
        if rng.random() < 0.15 and out_capacity > 0:
            out_capacity -= 1

        out_new = [0x1234] * max(row_count * token_count, 1)
        out_ref = out_new.copy()

        err_new = (
            attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_commit_only(
                row_count,
                token_count,
                staged,
                staged_capacity,
                out_new,
                out_capacity,
            )
        )
        err_ref = explicit_checked_copy_loops(
            row_count,
            token_count,
            staged,
            staged_capacity,
            out_ref,
            out_capacity,
        )

        assert err_new == err_ref
        assert out_new == out_ref


def main() -> None:
    test_source_contains_commit_only_helper()
    test_noalloc_wrapper_uses_commit_only_helper()
    test_known_vector_matches_explicit_checked_copy()
    test_error_paths_preserve_output()
    test_randomized_adversarial_parity()
    print("ok")


if __name__ == "__main__":
    main()
