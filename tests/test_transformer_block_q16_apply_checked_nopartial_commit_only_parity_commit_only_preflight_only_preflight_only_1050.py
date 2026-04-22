#!/usr/bin/env python3
"""Parity/adversarial harness for IQ-1050 diagnostics-only block preflight helper."""

from __future__ import annotations

from pathlib import Path

BLOCK_Q16_OK = 0
BLOCK_Q16_ERR_NULL_PTR = 1
BLOCK_Q16_ERR_BAD_PARAM = 2
BLOCK_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)
I64_SIZE = 8


def try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if lhs == 0 or rhs == 0:
        return BLOCK_Q16_OK, 0

    if lhs > 0:
        if rhs > 0:
            if lhs > I64_MAX // rhs:
                return BLOCK_Q16_ERR_OVERFLOW, 0
        else:
            if rhs < I64_MIN // lhs:
                return BLOCK_Q16_ERR_OVERFLOW, 0
    else:
        if rhs > 0:
            if lhs < I64_MIN // rhs:
                return BLOCK_Q16_ERR_OVERFLOW, 0
        else:
            if lhs != 0 and rhs < I64_MAX // lhs:
                return BLOCK_Q16_ERR_OVERFLOW, 0

    return BLOCK_Q16_OK, lhs * rhs


def iq1050_model(
    row_count: int,
    lane_count: int,
    input_capacity: int,
    rms_gamma_capacity: int,
    ffn_gate_capacity: int,
    ffn_up_capacity: int,
    workspace_q16_capacity: int,
    workspace_q32_capacity: int,
    out_capacity: int,
) -> tuple[int, tuple[int, int, int, int, int] | None]:
    vals = [
        row_count,
        lane_count,
        input_capacity,
        rms_gamma_capacity,
        ffn_gate_capacity,
        ffn_up_capacity,
        workspace_q16_capacity,
        workspace_q32_capacity,
        out_capacity,
    ]
    if any(v < 0 for v in vals):
        return BLOCK_Q16_ERR_BAD_PARAM, None

    err, dense_cells = try_mul_i64_checked(row_count, lane_count)
    if err != BLOCK_Q16_OK:
        return err, None

    err, required_q16_cells = try_mul_i64_checked(dense_cells, 3)
    if err != BLOCK_Q16_OK:
        return err, None

    err, required_q32_cells = try_mul_i64_checked(dense_cells, 3)
    if err != BLOCK_Q16_OK:
        return err, None

    err, required_q16_bytes = try_mul_i64_checked(required_q16_cells, I64_SIZE)
    if err != BLOCK_Q16_OK:
        return err, None

    err, required_q32_bytes = try_mul_i64_checked(required_q32_cells, I64_SIZE)
    if err != BLOCK_Q16_OK:
        return err, None

    if dense_cells != 0:
        if dense_cells > input_capacity:
            return BLOCK_Q16_ERR_BAD_PARAM, None
        if dense_cells > ffn_gate_capacity:
            return BLOCK_Q16_ERR_BAD_PARAM, None
        if dense_cells > ffn_up_capacity:
            return BLOCK_Q16_ERR_BAD_PARAM, None
        if dense_cells > out_capacity:
            return BLOCK_Q16_ERR_BAD_PARAM, None
        if lane_count > rms_gamma_capacity:
            return BLOCK_Q16_ERR_BAD_PARAM, None
        if required_q16_cells > workspace_q16_capacity:
            return BLOCK_Q16_ERR_BAD_PARAM, None
        if required_q32_cells > workspace_q32_capacity:
            return BLOCK_Q16_ERR_BAD_PARAM, None

    return BLOCK_Q16_OK, (
        dense_cells,
        required_q16_cells,
        required_q32_cells,
        required_q16_bytes,
        required_q32_bytes,
    )


def test_source_contains_iq1050_signature_and_tuple_contract() -> None:
    source = Path("src/model/block.HC").read_text(encoding="utf-8")
    signature = "I32 TransformerBlockQ16ApplyCheckedNoPartialCommitOnlyParityCommitOnlyPreflightOnlyPreflightOnly_1050("
    assert signature in source

    body = source.rsplit(signature, 1)[1]
    assert "staged_dense_cells" in body
    assert "staged_required_q16_cells" in body
    assert "staged_required_q32_cells" in body
    assert "staged_required_q16_bytes" in body
    assert "staged_required_q32_bytes" in body
    assert "recomputed_dense_cells" in body
    assert "snapshot_row_count" in body


def test_targeted_success_tuple() -> None:
    status, tpl = iq1050_model(
        row_count=8,
        lane_count=64,
        input_capacity=1024,
        rms_gamma_capacity=64,
        ffn_gate_capacity=1024,
        ffn_up_capacity=1024,
        workspace_q16_capacity=2048,
        workspace_q32_capacity=2048,
        out_capacity=1024,
    )
    assert status == BLOCK_Q16_OK
    assert tpl == (512, 1536, 1536, 12288, 12288)


def test_adversarial_capacity_rejection() -> None:
    status, tpl = iq1050_model(
        row_count=4,
        lane_count=32,
        input_capacity=128,
        rms_gamma_capacity=31,
        ffn_gate_capacity=128,
        ffn_up_capacity=128,
        workspace_q16_capacity=384,
        workspace_q32_capacity=384,
        out_capacity=128,
    )
    assert status == BLOCK_Q16_ERR_BAD_PARAM
    assert tpl is None


def test_overflow_detection_large_geometry() -> None:
    status, tpl = iq1050_model(
        row_count=1 << 62,
        lane_count=4,
        input_capacity=I64_MAX,
        rms_gamma_capacity=I64_MAX,
        ffn_gate_capacity=I64_MAX,
        ffn_up_capacity=I64_MAX,
        workspace_q16_capacity=I64_MAX,
        workspace_q32_capacity=I64_MAX,
        out_capacity=I64_MAX,
    )
    assert status == BLOCK_Q16_ERR_OVERFLOW
    assert tpl is None
