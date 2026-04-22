#!/usr/bin/env python3
"""Reference checks for FPQ16SoftmaxRowsStableCheckedNoPartial semantics (IQ-1162)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_softmax_exp_phase_from_preclamped_checked import (
    EXP_Q16_MAX_INPUT,
    EXP_Q16_MIN_INPUT,
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    FP_Q16_ONE,
    I64_MAX_VALUE,
)
from test_softmax_from_preclamped_checked import fpq16_softmax_from_preclamped_checked_reference

I64_MIN_VALUE = -(1 << 63)


def _required_row_cells(row_count: int, lane_count: int, row_stride: int) -> tuple[int, int]:
    if row_count > 0 and lane_count > 0:
        last_row = row_count - 1
        if last_row > 0 and row_stride > (I64_MAX_VALUE // last_row):
            return FP_Q16_ERR_OVERFLOW, 0
        last_base = last_row * row_stride
        if last_base > (I64_MAX_VALUE - (lane_count - 1)):
            return FP_Q16_ERR_OVERFLOW, 0
        return FP_Q16_OK, last_base + lane_count
    return FP_Q16_OK, 0


def fpq16_softmax_rows_stable_checked_nopartial_reference(
    logits_q16: list[int] | None,
    logits_capacity: int,
    row_count: int,
    lane_count: int,
    logits_row_stride: int,
    probs_q16: list[int] | None,
    probs_capacity: int,
    probs_row_stride: int,
    scratch_q16: list[int] | None,
    scratch_capacity: int,
) -> tuple[int, list[int] | None, list[int] | None]:
    if logits_q16 is None or probs_q16 is None or scratch_q16 is None:
        return FP_Q16_ERR_NULL_PTR, probs_q16, scratch_q16

    probs_out = probs_q16[:]
    scratch_out = scratch_q16[:]

    if row_count < 0 or lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM, probs_out, scratch_out
    if logits_capacity < 0 or probs_capacity < 0 or scratch_capacity < 0:
        return FP_Q16_ERR_BAD_PARAM, probs_out, scratch_out

    if lane_count > 0 and (logits_row_stride < lane_count or probs_row_stride < lane_count):
        return FP_Q16_ERR_BAD_PARAM, probs_out, scratch_out

    st, required_logits = _required_row_cells(row_count, lane_count, logits_row_stride)
    if st != FP_Q16_OK:
        return st, probs_out, scratch_out

    st, required_probs = _required_row_cells(row_count, lane_count, probs_row_stride)
    if st != FP_Q16_OK:
        return st, probs_out, scratch_out

    if logits_capacity < required_logits or probs_capacity < required_probs:
        return FP_Q16_ERR_BAD_PARAM, probs_out, scratch_out

    if row_count > 0 and lane_count > 0:
        if row_count > (I64_MAX_VALUE // lane_count):
            return FP_Q16_ERR_OVERFLOW, probs_out, scratch_out
        stage_cells = row_count * lane_count
    else:
        stage_cells = 0

    if lane_count > (I64_MAX_VALUE // 2):
        return FP_Q16_ERR_OVERFLOW, probs_out, scratch_out

    temp_cells = lane_count * 2
    if stage_cells > (I64_MAX_VALUE - temp_cells):
        return FP_Q16_ERR_OVERFLOW, probs_out, scratch_out

    required_scratch = stage_cells + temp_cells
    if scratch_capacity < required_scratch:
        return FP_Q16_ERR_BAD_PARAM, probs_out, scratch_out

    if row_count == 0 or lane_count == 0:
        return FP_Q16_OK, probs_out, scratch_out

    staged_probs = [0] * stage_cells

    for row in range(row_count):
        row_base = row * logits_row_stride
        row_logits = logits_q16[row_base : row_base + lane_count]
        row_max = row_logits[0]
        for lane in range(1, lane_count):
            if row_logits[lane] > row_max:
                row_max = row_logits[lane]

        shifted = [0] * lane_count
        for lane in range(lane_count):
            value = row_logits[lane]

            if row_max < 0 and value > (I64_MAX_VALUE + row_max):
                return FP_Q16_ERR_OVERFLOW, probs_out, scratch_out
            if row_max > 0 and value < (I64_MIN_VALUE + row_max):
                return FP_Q16_ERR_OVERFLOW, probs_out, scratch_out

            delta = value - row_max
            if delta < EXP_Q16_MIN_INPUT:
                delta = EXP_Q16_MIN_INPUT
            elif delta > EXP_Q16_MAX_INPUT:
                delta = EXP_Q16_MAX_INPUT
            shifted[lane] = delta

        err, _exp_out, row_probs = fpq16_softmax_from_preclamped_checked_reference(
            shifted,
            [0] * lane_count,
            lane_count,
            [0] * lane_count,
        )
        if err != FP_Q16_OK:
            return err, probs_out, scratch_out

        staged_probs[row * lane_count : (row + 1) * lane_count] = row_probs

    for row in range(row_count):
        dst_base = row * probs_row_stride
        src_base = row * lane_count
        for lane in range(lane_count):
            probs_out[dst_base + lane] = staged_probs[src_base + lane]

    scratch_out[:stage_cells] = staged_probs
    return FP_Q16_OK, probs_out, scratch_out


def test_source_contains_iq1162_signature_and_core_contracts() -> None:
    src = Path("src/math/softmax.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16SoftmaxRowsStableCheckedNoPartial(I64 *logits_q16,"
    assert sig in src
    body = src.split(sig, 1)[1].split("U0 FPQ16Softmax(", 1)[0]

    assert "SOFTMAX_EXP_Q16_MAX_INPUT" in src
    assert "SOFTMAX_EXP_Q16_MIN_INPUT" in src
    assert "staged_probs_q16 = scratch_q16;" in body
    assert "status = FPQ16SoftmaxFromPreclampedChecked(shifted_row_q16," in body
    assert "if (scratch_capacity < required_scratch_cells)" in body
    assert "for (row = 0; row < row_count; row++)" in body


def test_bad_inputs_and_capacity_fail_without_output_writes() -> None:
    logits = [1, 2, 3, 4, 5, 6]
    probs = [999] * 8
    scratch = [777] * 32

    st, probs_out, _ = fpq16_softmax_rows_stable_checked_nopartial_reference(
        None,
        len(logits),
        2,
        3,
        3,
        probs,
        len(probs),
        4,
        scratch,
        len(scratch),
    )
    assert st == FP_Q16_ERR_NULL_PTR
    assert probs_out == probs

    st, probs_out, _ = fpq16_softmax_rows_stable_checked_nopartial_reference(
        logits,
        len(logits),
        -1,
        3,
        3,
        probs,
        len(probs),
        4,
        scratch,
        len(scratch),
    )
    assert st == FP_Q16_ERR_BAD_PARAM
    assert probs_out == probs

    st, probs_out, _ = fpq16_softmax_rows_stable_checked_nopartial_reference(
        logits,
        len(logits),
        2,
        3,
        3,
        probs,
        len(probs),
        4,
        scratch,
        2,
    )
    assert st == FP_Q16_ERR_BAD_PARAM
    assert probs_out == probs


def test_zero_rows_or_lanes_are_noops() -> None:
    logits = [1, 2, 3, 4]
    probs = [100, 200, 300, 400]
    scratch = [500] * 16

    st, probs_out, scratch_out = fpq16_softmax_rows_stable_checked_nopartial_reference(
        logits,
        len(logits),
        0,
        4,
        4,
        probs,
        len(probs),
        4,
        scratch,
        len(scratch),
    )
    assert st == FP_Q16_OK
    assert probs_out == probs
    assert scratch_out == scratch

    st, probs_out, scratch_out = fpq16_softmax_rows_stable_checked_nopartial_reference(
        logits,
        len(logits),
        2,
        0,
        0,
        probs,
        len(probs),
        0,
        scratch,
        len(scratch),
    )
    assert st == FP_Q16_OK
    assert probs_out == probs
    assert scratch_out == scratch


def test_rows_softmax_probabilities_sum_to_one() -> None:
    row_count = 3
    lane_count = 4
    logits_row_stride = 5
    probs_row_stride = 6

    logits = [
        2 * FP_Q16_ONE,
        1 * FP_Q16_ONE,
        0,
        -(FP_Q16_ONE // 2),
        12345,
        -(3 * FP_Q16_ONE),
        -(2 * FP_Q16_ONE),
        -(FP_Q16_ONE),
        0,
        67890,
        EXP_Q16_MAX_INPUT + (2 * FP_Q16_ONE),
        0,
        -EXP_Q16_MAX_INPUT,
        5 * FP_Q16_ONE,
        111,
    ]

    probs = [0x123456789] * (row_count * probs_row_stride)
    scratch = [0] * (row_count * lane_count + 2 * lane_count)

    st, probs_out, _ = fpq16_softmax_rows_stable_checked_nopartial_reference(
        logits,
        len(logits),
        row_count,
        lane_count,
        logits_row_stride,
        probs,
        len(probs),
        probs_row_stride,
        scratch,
        len(scratch),
    )
    assert st == FP_Q16_OK

    for row in range(row_count):
        row_probs = probs_out[row * probs_row_stride : row * probs_row_stride + lane_count]
        assert all(value >= 0 for value in row_probs)
        assert sum(row_probs) == FP_Q16_ONE


def test_randomized_rows_match_rowwise_reference() -> None:
    rng = random.Random(1162)

    for _ in range(300):
        row_count = rng.randint(1, 8)
        lane_count = rng.randint(1, 16)
        logits_row_stride = lane_count + rng.randint(0, 3)
        probs_row_stride = lane_count + rng.randint(0, 4)

        logits_capacity = row_count * logits_row_stride
        probs_capacity = row_count * probs_row_stride

        logits = [rng.randint(-900000, 900000) for _ in range(logits_capacity)]
        probs = [rng.randint(0, 1000) for _ in range(probs_capacity)]

        stage_cells = row_count * lane_count
        scratch_capacity = stage_cells + (2 * lane_count)
        scratch = [0] * scratch_capacity

        st, probs_out, _ = fpq16_softmax_rows_stable_checked_nopartial_reference(
            logits,
            logits_capacity,
            row_count,
            lane_count,
            logits_row_stride,
            probs,
            probs_capacity,
            probs_row_stride,
            scratch,
            scratch_capacity,
        )
        assert st == FP_Q16_OK

        expected_probs = probs[:]
        for row in range(row_count):
            row_logits = logits[row * logits_row_stride : row * logits_row_stride + lane_count]
            row_max = max(row_logits)
            shifted = [max(EXP_Q16_MIN_INPUT, min(EXP_Q16_MAX_INPUT, value - row_max)) for value in row_logits]
            err, _, row_probs = fpq16_softmax_from_preclamped_checked_reference(
                shifted,
                [0] * lane_count,
                lane_count,
                [0] * lane_count,
            )
            assert err == FP_Q16_OK
            for lane in range(lane_count):
                expected_probs[row * probs_row_stride + lane] = row_probs[lane]

        assert probs_out == expected_probs


def run() -> None:
    test_source_contains_iq1162_signature_and_core_contracts()
    test_bad_inputs_and_capacity_fail_without_output_writes()
    test_zero_rows_or_lanes_are_noops()
    test_rows_softmax_probabilities_sum_to_one()
    test_randomized_rows_match_rowwise_reference()
    print("softmax_rows_stable_checked_nopartial_reference_checks=ok")


if __name__ == "__main__":
    run()
