#!/usr/bin/env python3
"""Parity harness for IQ-1194 FPQ16RMSNormRowsCheckedNoPartialCommitOnly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_rmsnorm_q16_rows_checked import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
)
from test_rmsnorm_q16_rows_checked_nopartial import (
    fpq16_rmsnorm_rows_checked_nopartial_wrapper_reference,
)


def fpq16_rmsnorm_rows_compute_required_cells_checked_reference(
    row_count: int,
    lane_count: int,
    input_row_stride: int,
    gamma_row_stride: int,
    output_row_stride: int,
) -> tuple[int, int, int, int]:
    if row_count < 0 or lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM, 0, 0, 0
    if input_row_stride < 0 or gamma_row_stride < 0 or output_row_stride < 0:
        return FP_Q16_ERR_BAD_PARAM, 0, 0, 0

    if row_count == 0 or lane_count == 0:
        return FP_Q16_OK, 0, 0, 0

    if (
        input_row_stride < lane_count
        or gamma_row_stride < lane_count
        or output_row_stride < lane_count
    ):
        return FP_Q16_ERR_BAD_PARAM, 0, 0, 0

    last_row = row_count - 1

    required_input_cells = last_row * input_row_stride + lane_count
    required_gamma_cells = last_row * gamma_row_stride + lane_count
    required_output_cells = last_row * output_row_stride + lane_count

    if (
        required_input_cells < 0
        or required_gamma_cells < 0
        or required_output_cells < 0
    ):
        return FP_Q16_ERR_OVERFLOW, 0, 0, 0

    return (
        FP_Q16_OK,
        required_input_cells,
        required_gamma_cells,
        required_output_cells,
    )


def fpq16_rmsnorm_rows_checked_nopartial_commit_only_reference(
    input_q16: list[int] | None,
    input_capacity: int,
    input_row_stride: int,
    gamma_q16: list[int] | None,
    gamma_capacity: int,
    gamma_row_stride: int,
    output_q16: list[int] | None,
    output_capacity: int,
    output_row_stride: int,
    row_count: int,
    lane_count: int,
    eps_q16: int,
) -> tuple[int, list[int], int | None, int | None, int | None]:
    if (
        input_q16 is None
        or gamma_q16 is None
        or output_q16 is None
    ):
        return FP_Q16_ERR_NULL_PTR, [], None, None, None

    snapshot = (
        row_count,
        lane_count,
        input_capacity,
        gamma_capacity,
        output_capacity,
        input_row_stride,
        gamma_row_stride,
        output_row_stride,
        id(input_q16),
        id(gamma_q16),
        id(output_q16),
    )

    status_req, req_input, req_gamma, req_output = (
        fpq16_rmsnorm_rows_compute_required_cells_checked_reference(
            row_count,
            lane_count,
            input_row_stride,
            gamma_row_stride,
            output_row_stride,
        )
    )
    if status_req != FP_Q16_OK:
        return status_req, [], None, None, None

    if req_input > input_capacity or req_gamma > gamma_capacity or req_output > output_capacity:
        return FP_Q16_ERR_BAD_PARAM, [], None, None, None

    status_core, out = fpq16_rmsnorm_rows_checked_nopartial_wrapper_reference(
        input_q16,
        input_capacity,
        input_row_stride,
        gamma_q16,
        gamma_capacity,
        gamma_row_stride,
        output_q16,
        output_capacity,
        output_row_stride,
        row_count,
        lane_count,
        eps_q16,
    )
    if status_core != FP_Q16_OK:
        return status_core, [], None, None, None

    current = (
        row_count,
        lane_count,
        input_capacity,
        gamma_capacity,
        output_capacity,
        input_row_stride,
        gamma_row_stride,
        output_row_stride,
        id(input_q16),
        id(gamma_q16),
        id(output_q16),
    )
    if current != snapshot:
        return FP_Q16_ERR_BAD_PARAM, [], None, None, None

    status_req2, req_input2, req_gamma2, req_output2 = (
        fpq16_rmsnorm_rows_compute_required_cells_checked_reference(
            row_count,
            lane_count,
            input_row_stride,
            gamma_row_stride,
            output_row_stride,
        )
    )
    if status_req2 != FP_Q16_OK:
        return status_req2, [], None, None, None

    if (req_input2, req_gamma2, req_output2) != (req_input, req_gamma, req_output):
        return FP_Q16_ERR_BAD_PARAM, [], None, None, None

    return FP_Q16_OK, out, req_input, req_gamma, req_output


def test_source_contains_iq1194_wrapper_signature_and_calls() -> None:
    source = Path("src/math/rmsnorm.HC").read_text(encoding="utf-8")

    sig = "I32 FPQ16RMSNormRowsCheckedNoPartialCommitOnly("
    assert sig in source

    helper_sig = "I32 FPQ16RMSNormRowsComputeRequiredCellsChecked("
    assert helper_sig in source

    body = source.split(sig, 1)[1]
    assert "status = FPQ16RMSNormRowsComputeRequiredCellsChecked(snapshot_row_count," in body
    assert "status = FPQ16RMSNormRowsCheckedNoPartial(input_q16," in body
    assert "status = FPQ16RMSNormRowsComputeRequiredCellsChecked(row_count," in body
    assert "*required_input_cells_out = snapshot_required_input_cells;" in body
    assert "*required_gamma_cells_out = snapshot_required_gamma_cells;" in body
    assert "*required_output_cells_out = snapshot_required_output_cells;" in body


def test_null_diagnostics_ptr_rejected() -> None:
    input_q16 = [0, 0, 0, 0]
    gamma_q16 = [1 << 16] * 4
    output_q16 = [123, 456, 789, 111]

    status, out, req_in, req_gamma, req_out = (
        fpq16_rmsnorm_rows_checked_nopartial_commit_only_reference(
            input_q16,
            len(input_q16),
            2,
            gamma_q16,
            len(gamma_q16),
            2,
            output_q16,
            len(output_q16),
            2,
            2,
            2,
            1,
        )
    )
    assert status == FP_Q16_OK
    assert req_in == 4
    assert req_gamma == 4
    assert req_out == 4
    assert len(out) == len(output_q16)


def test_required_cells_capacity_guard() -> None:
    input_q16 = [1, 2, 3, 4, 5, 6]
    gamma_q16 = [1 << 16] * 6
    output_q16 = [9, 9, 9, 9, 9, 9]

    status, out, req_in, req_gamma, req_out = (
        fpq16_rmsnorm_rows_checked_nopartial_commit_only_reference(
            input_q16,
            3,
            3,
            gamma_q16,
            len(gamma_q16),
            3,
            output_q16,
            len(output_q16),
            3,
            2,
            2,
            1,
        )
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out == []
    assert req_in is None
    assert req_gamma is None
    assert req_out is None


def test_output_preserved_on_error() -> None:
    rng = random.Random(119401)

    row_count = 3
    lane_count = 4
    input_stride = 6
    gamma_stride = 6
    output_stride = 7

    input_capacity = row_count * input_stride
    gamma_capacity = row_count * gamma_stride

    input_q16 = [rng.randint(-4 << 16, 4 << 16) for _ in range(input_capacity)]
    gamma_q16 = [rng.randint(1, 3 << 16) for _ in range(gamma_capacity)]
    output_q16 = [12345, -22222]
    before = output_q16[:]

    status, out, req_in, req_gamma, req_out = (
        fpq16_rmsnorm_rows_checked_nopartial_commit_only_reference(
            input_q16,
            input_capacity,
            input_stride,
            gamma_q16,
            gamma_capacity,
            gamma_stride,
            output_q16,
            len(output_q16),
            output_stride,
            row_count,
            lane_count,
            1,
        )
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out == []
    assert req_in is None and req_gamma is None and req_out is None
    assert output_q16 == before


def test_random_parity_commit_only_outputs_and_required_cells() -> None:
    rng = random.Random(119402)

    for _ in range(240):
        row_count = rng.randint(1, 6)
        lane_count = rng.randint(1, 8)
        input_stride = lane_count + rng.randint(0, 4)
        gamma_stride = lane_count + rng.randint(0, 4)
        output_stride = lane_count + rng.randint(0, 4)

        input_capacity = row_count * input_stride
        gamma_capacity = row_count * gamma_stride
        output_capacity = row_count * output_stride

        input_q16 = [rng.randint(-6 << 16, 6 << 16) for _ in range(input_capacity)]
        gamma_q16 = [rng.randint(1, 4 << 16) for _ in range(gamma_capacity)]
        output_q16 = [rng.randint(-(1 << 19), 1 << 19) for _ in range(output_capacity)]

        status_wrap, out_wrap, req_in, req_gamma, req_out = (
            fpq16_rmsnorm_rows_checked_nopartial_commit_only_reference(
                input_q16,
                input_capacity,
                input_stride,
                gamma_q16,
                gamma_capacity,
                gamma_stride,
                output_q16,
                output_capacity,
                output_stride,
                row_count,
                lane_count,
                1,
            )
        )

        status_base, out_base = fpq16_rmsnorm_rows_checked_nopartial_wrapper_reference(
            input_q16,
            input_capacity,
            input_stride,
            gamma_q16,
            gamma_capacity,
            gamma_stride,
            output_q16,
            output_capacity,
            output_stride,
            row_count,
            lane_count,
            1,
        )

        assert status_wrap == status_base
        if status_wrap == FP_Q16_OK:
            assert out_wrap == out_base

            status_req, exp_in, exp_gamma, exp_out = (
                fpq16_rmsnorm_rows_compute_required_cells_checked_reference(
                    row_count,
                    lane_count,
                    input_stride,
                    gamma_stride,
                    output_stride,
                )
            )
            assert status_req == FP_Q16_OK
            assert req_in == exp_in
            assert req_gamma == exp_gamma
            assert req_out == exp_out


if __name__ == "__main__":
    test_source_contains_iq1194_wrapper_signature_and_calls()
    test_null_diagnostics_ptr_rejected()
    test_required_cells_capacity_guard()
    test_output_preserved_on_error()
    test_random_parity_commit_only_outputs_and_required_cells()
    print("rmsnorm_q16_rows_checked_nopartial_commit_only=ok")
