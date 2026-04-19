#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_ffn_q16_swiglu_apply_checked import (
    FFN_Q16_ERR_BAD_PARAM,
    FFN_Q16_ERR_NULL_PTR,
    FFN_Q16_ERR_OVERFLOW,
    FFN_Q16_OK,
    i64_add_checked,
    i64_mul_checked,
)


def ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only(
    row_count: int,
    lane_count: int,
    staging_out_q16,
    staging_out_capacity: int,
    stage_cell_capacity: int,
    out_q16,
    out_capacity: int,
) -> int:
    if staging_out_q16 is None or out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if staging_out_capacity < 0 or stage_cell_capacity < 0 or out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        return FFN_Q16_OK

    if staging_out_q16 is out_q16:
        return FFN_Q16_ERR_BAD_PARAM

    err, required_stage_cells = i64_mul_checked(row_count, lane_count)
    if err != FFN_Q16_OK:
        return err

    err, required_stage_bytes = i64_mul_checked(required_stage_cells, 8)
    if err != FFN_Q16_OK:
        return err

    err, required_out_cells = i64_mul_checked(row_count - 1, lane_count)
    if err != FFN_Q16_OK:
        return err
    err, required_out_cells = i64_add_checked(required_out_cells, lane_count)
    if err != FFN_Q16_OK:
        return err

    if required_stage_cells > staging_out_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_stage_cells > stage_cell_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_out_cells > out_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_stage_bytes < 0:
        return FFN_Q16_ERR_BAD_PARAM

    for row_index in range(row_count):
        err, row_base = i64_mul_checked(row_index, lane_count)
        if err != FFN_Q16_OK:
            return err

        err, stage_row_base = i64_mul_checked(row_index, lane_count)
        if err != FFN_Q16_OK:
            return err

        for lane_index in range(lane_count):
            err, out_index = i64_add_checked(row_base, lane_index)
            if err != FFN_Q16_OK:
                return err
            if out_index < 0 or out_index >= required_out_cells:
                return FFN_Q16_ERR_BAD_PARAM
            if out_index >= out_capacity:
                return FFN_Q16_ERR_BAD_PARAM

            err, stage_index = i64_add_checked(stage_row_base, lane_index)
            if err != FFN_Q16_OK:
                return err
            if stage_index < 0 or stage_index >= required_stage_cells:
                return FFN_Q16_ERR_BAD_PARAM
            if stage_index >= staging_out_capacity:
                return FFN_Q16_ERR_BAD_PARAM

    for row_index in range(row_count):
        err, row_base = i64_mul_checked(row_index, lane_count)
        if err != FFN_Q16_OK:
            return err

        err, stage_row_base = i64_mul_checked(row_index, lane_count)
        if err != FFN_Q16_OK:
            return err

        for lane_index in range(lane_count):
            err, out_index = i64_add_checked(row_base, lane_index)
            if err != FFN_Q16_OK:
                return err
            if out_index < 0 or out_index >= required_out_cells:
                return FFN_Q16_ERR_BAD_PARAM
            if out_index >= out_capacity:
                return FFN_Q16_ERR_BAD_PARAM

            err, stage_index = i64_add_checked(stage_row_base, lane_index)
            if err != FFN_Q16_OK:
                return err
            if stage_index < 0 or stage_index >= required_stage_cells:
                return FFN_Q16_ERR_BAD_PARAM
            if stage_index >= staging_out_capacity:
                return FFN_Q16_ERR_BAD_PARAM

            out_q16[out_index] = staging_out_q16[stage_index]

    return FFN_Q16_OK


def explicit_checked_copy_loops(
    row_count: int,
    lane_count: int,
    staging_out_q16,
    staging_out_capacity: int,
    stage_cell_capacity: int,
    out_q16,
    out_capacity: int,
) -> int:
    if staging_out_q16 is None or out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if staging_out_capacity < 0 or stage_cell_capacity < 0 or out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        return FFN_Q16_OK

    if staging_out_q16 is out_q16:
        return FFN_Q16_ERR_BAD_PARAM

    err, required_stage_cells = i64_mul_checked(row_count, lane_count)
    if err != FFN_Q16_OK:
        return err

    err, required_out_cells = i64_mul_checked(row_count - 1, lane_count)
    if err != FFN_Q16_OK:
        return err
    err, required_out_cells = i64_add_checked(required_out_cells, lane_count)
    if err != FFN_Q16_OK:
        return err

    if required_stage_cells > staging_out_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_stage_cells > stage_cell_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_out_cells > out_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    for row_index in range(row_count):
        row_base = row_index * lane_count
        for lane_index in range(lane_count):
            out_index = row_base + lane_index
            out_q16[out_index] = staging_out_q16[out_index]

    return FFN_Q16_OK


def test_source_contains_noalloc_commit_only_helper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(" in body
    assert "if (staging_out_q16 == out_q16)" in body
    assert "if (required_stage_cells > stage_cell_capacity)" in body
    assert "if (out_index < 0 || out_index >= required_out_cells)" in body
    assert "if (stage_index < 0 || stage_index >= required_stage_cells)" in body
    assert "for (row_index = 0; row_index < row_count; row_index++)" in body
    assert "for (lane_index = 0; lane_index < lane_count; lane_index++)" in body
    assert "out_q16[out_index] = staging_out_q16[stage_index];" in body


def test_known_vectors_match_explicit_checked_copy() -> None:
    row_count = 4
    lane_count = 6
    required_cells = row_count * lane_count

    stage = [((i * 13) - 101) << 8 for i in range(required_cells)]
    out_new = [0x4444] * required_cells
    out_ref = out_new.copy()

    err_new = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only(
        row_count,
        lane_count,
        stage,
        len(stage),
        required_cells,
        out_new,
        len(out_new),
    )
    err_ref = explicit_checked_copy_loops(
        row_count,
        lane_count,
        stage,
        len(stage),
        required_cells,
        out_ref,
        len(out_ref),
    )

    assert err_new == err_ref == FFN_Q16_OK
    assert out_new == out_ref == stage


def test_error_paths_preserve_output() -> None:
    stage = [1, 2, 3, 4, 5, 6]
    out_seed = [0x5A5A] * 6

    out_new = out_seed.copy()
    out_ref = out_seed.copy()
    err_new = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only(
        2,
        3,
        stage,
        len(stage),
        5,
        out_new,
        len(out_new),
    )
    err_ref = explicit_checked_copy_loops(
        2,
        3,
        stage,
        len(stage),
        5,
        out_ref,
        len(out_ref),
    )

    assert err_new == err_ref == FFN_Q16_ERR_BAD_PARAM
    assert out_new == out_ref == out_seed

    out_new = out_seed.copy()
    out_ref = out_seed.copy()
    err_new = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only(
        -1,
        1,
        stage,
        len(stage),
        len(stage),
        out_new,
        len(out_new),
    )
    err_ref = explicit_checked_copy_loops(
        -1,
        1,
        stage,
        len(stage),
        len(stage),
        out_ref,
        len(out_ref),
    )

    assert err_new == err_ref == FFN_Q16_ERR_BAD_PARAM
    assert out_new == out_ref == out_seed

    out_new = out_seed.copy()
    out_ref = out_seed.copy()
    err_new = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only(
        2,
        3,
        None,
        len(stage),
        len(stage),
        out_new,
        len(out_new),
    )
    err_ref = explicit_checked_copy_loops(
        2,
        3,
        None,
        len(stage),
        len(stage),
        out_ref,
        len(out_ref),
    )

    assert err_new == err_ref == FFN_Q16_ERR_NULL_PTR
    assert out_new == out_ref == out_seed

    shared = [9, 8, 7, 6, 5, 4]
    err_new = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only(
        2,
        3,
        shared,
        len(shared),
        len(shared),
        shared,
        len(shared),
    )
    err_ref = explicit_checked_copy_loops(
        2,
        3,
        shared,
        len(shared),
        len(shared),
        shared,
        len(shared),
    )
    assert err_new == err_ref == FFN_Q16_ERR_BAD_PARAM


def test_overflow_contract() -> None:
    huge = 1 << 62
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only(
        huge,
        huge,
        [1],
        (1 << 63) - 1,
        (1 << 63) - 1,
        [0],
        (1 << 63) - 1,
    )
    assert err == FFN_Q16_ERR_OVERFLOW


def test_randomized_parity_vs_explicit_checked_copy() -> None:
    rng = random.Random(20260420_598)

    for _ in range(5000):
        row_count = rng.randint(0, 32)
        lane_count = rng.randint(0, 32)
        required = row_count * lane_count

        staged_capacity = max(0, required + rng.randint(-12, 12))
        stage_cell_capacity = max(0, required + rng.randint(-12, 12))
        out_capacity = max(0, required + rng.randint(-12, 12))

        if rng.random() < 0.05:
            staged_capacity = -rng.randint(1, 50)
        if rng.random() < 0.05:
            stage_cell_capacity = -rng.randint(1, 50)
        if rng.random() < 0.05:
            out_capacity = -rng.randint(1, 50)

        if rng.random() < 0.03:
            row_count = (1 << 62) + rng.randint(0, 3)
            lane_count = (1 << 62) + rng.randint(0, 3)

        stage = (
            None
            if rng.random() < 0.03
            else [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(staged_capacity, 1))]
        )
        out_seed = [rng.randint(-(1 << 12), (1 << 12)) for _ in range(max(out_capacity, 1))]
        out_new = out_seed.copy()
        out_ref = out_seed.copy()

        err_new = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only(
            row_count,
            lane_count,
            stage,
            staged_capacity,
            stage_cell_capacity,
            out_new,
            out_capacity,
        )
        err_ref = explicit_checked_copy_loops(
            row_count,
            lane_count,
            stage,
            staged_capacity,
            stage_cell_capacity,
            out_ref,
            out_capacity,
        )

        assert err_new == err_ref
        if err_new == FFN_Q16_OK:
            assert out_new == out_ref
        else:
            assert out_new == out_ref == out_seed


if __name__ == "__main__":
    test_source_contains_noalloc_commit_only_helper()
    test_known_vectors_match_explicit_checked_copy()
    test_error_paths_preserve_output()
    test_overflow_contract()
    test_randomized_parity_vs_explicit_checked_copy()
    print("ok")
