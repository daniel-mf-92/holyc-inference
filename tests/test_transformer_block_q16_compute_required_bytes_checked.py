#!/usr/bin/env python3
"""Parity harness for IQ-743 TransformerBlockQ16ComputeRequiredBytesChecked."""

from __future__ import annotations

import random
from pathlib import Path

BLOCK_Q16_OK = 0
BLOCK_Q16_ERR_NULL_PTR = 1
BLOCK_Q16_ERR_BAD_PARAM = 2
BLOCK_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


def try_add_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if rhs > 0 and lhs > I64_MAX - rhs:
        return BLOCK_Q16_ERR_OVERFLOW, 0
    if rhs < 0 and lhs < I64_MIN - rhs:
        return BLOCK_Q16_ERR_OVERFLOW, 0
    return BLOCK_Q16_OK, lhs + rhs


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


def transformer_block_q16_compute_required_bytes_checked(
    row_count: int,
    lane_count: int,
    out_attn_stage_cells,
    out_ffn_stage_cells,
    out_residual_cells,
    out_total_stage_bytes,
) -> int:
    if (
        out_attn_stage_cells is None
        or out_ffn_stage_cells is None
        or out_residual_cells is None
        or out_total_stage_bytes is None
    ):
        return BLOCK_Q16_ERR_NULL_PTR

    if row_count < 0 or lane_count < 0:
        return BLOCK_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        out_attn_stage_cells[0] = 0
        out_ffn_stage_cells[0] = 0
        out_residual_cells[0] = 0
        out_total_stage_bytes[0] = 0
        return BLOCK_Q16_OK

    err, dense_cells = try_mul_i64_checked(row_count, lane_count)
    if err != BLOCK_Q16_OK:
        return err

    # Keep diagnostics staged until all checked math succeeds.
    attn_stage_cells = dense_cells
    ffn_stage_cells = dense_cells
    residual_cells = dense_cells

    err, total_stage_cells = try_add_i64_checked(dense_cells, dense_cells)
    if err != BLOCK_Q16_OK:
        return err
    err, total_stage_cells = try_add_i64_checked(total_stage_cells, dense_cells)
    if err != BLOCK_Q16_OK:
        return err

    err, total_stage_bytes = try_mul_i64_checked(total_stage_cells, 8)
    if err != BLOCK_Q16_OK:
        return err

    out_attn_stage_cells[0] = attn_stage_cells
    out_ffn_stage_cells[0] = ffn_stage_cells
    out_residual_cells[0] = residual_cells
    out_total_stage_bytes[0] = total_stage_bytes
    return BLOCK_Q16_OK


def explicit_checked_composition(row_count: int, lane_count: int) -> tuple[int, tuple[int, int, int, int]]:
    if row_count < 0 or lane_count < 0:
        return BLOCK_Q16_ERR_BAD_PARAM, (0, 0, 0, 0)

    if row_count == 0 or lane_count == 0:
        return BLOCK_Q16_OK, (0, 0, 0, 0)

    err, dense_cells = try_mul_i64_checked(row_count, lane_count)
    if err != BLOCK_Q16_OK:
        return err, (0, 0, 0, 0)

    err, total_stage_cells = try_add_i64_checked(dense_cells, dense_cells)
    if err != BLOCK_Q16_OK:
        return err, (0, 0, 0, 0)
    err, total_stage_cells = try_add_i64_checked(total_stage_cells, dense_cells)
    if err != BLOCK_Q16_OK:
        return err, (0, 0, 0, 0)

    err, total_stage_bytes = try_mul_i64_checked(total_stage_cells, 8)
    if err != BLOCK_Q16_OK:
        return err, (0, 0, 0, 0)

    return BLOCK_Q16_OK, (dense_cells, dense_cells, dense_cells, total_stage_bytes)


def test_source_contains_iq743_signature_and_checked_math() -> None:
    source = Path("src/model/block.HC").read_text(encoding="utf-8")
    signature = "I32 TransformerBlockQ16ComputeRequiredBytesChecked("
    assert signature in source

    body = source.split(signature, 1)[1]
    assert "if (!out_attn_stage_cells || !out_ffn_stage_cells ||" in body
    assert "if (row_count < 0 || lane_count < 0)" in body
    assert "status = BlockTryMulI64Checked(row_count," in body
    assert "status = BlockTryMulI64Checked(total_stage_cells," in body
    assert "*out_total_stage_bytes = total_stage_bytes;" in body


def test_targeted_vectors_and_no_partial_diagnostics() -> None:
    cases = [
        (0, 0, BLOCK_Q16_OK),
        (0, 9, BLOCK_Q16_OK),
        (7, 0, BLOCK_Q16_OK),
        (3, 5, BLOCK_Q16_OK),
        (I64_MAX, 2, BLOCK_Q16_ERR_OVERFLOW),
        (-(1 << 10), 4, BLOCK_Q16_ERR_BAD_PARAM),
        (4, -(1 << 10), BLOCK_Q16_ERR_BAD_PARAM),
    ]

    for row_count, lane_count, expected_err in cases:
        attn = [0x1111]
        ffn = [0x2222]
        residual = [0x3333]
        total_bytes = [0x4444]

        err = transformer_block_q16_compute_required_bytes_checked(
            row_count,
            lane_count,
            attn,
            ffn,
            residual,
            total_bytes,
        )
        assert err == expected_err

        if err == BLOCK_Q16_OK:
            _, expected = explicit_checked_composition(row_count, lane_count)
            assert (attn[0], ffn[0], residual[0], total_bytes[0]) == expected
        elif row_count != 0 and lane_count != 0:
            assert attn[0] == 0x1111
            assert ffn[0] == 0x2222
            assert residual[0] == 0x3333
            assert total_bytes[0] == 0x4444


def test_null_pointer_contract() -> None:
    err = transformer_block_q16_compute_required_bytes_checked(1, 1, None, [0], [0], [0])
    assert err == BLOCK_Q16_ERR_NULL_PTR

    err = transformer_block_q16_compute_required_bytes_checked(1, 1, [0], None, [0], [0])
    assert err == BLOCK_Q16_ERR_NULL_PTR

    err = transformer_block_q16_compute_required_bytes_checked(1, 1, [0], [0], None, [0])
    assert err == BLOCK_Q16_ERR_NULL_PTR

    err = transformer_block_q16_compute_required_bytes_checked(1, 1, [0], [0], [0], None)
    assert err == BLOCK_Q16_ERR_NULL_PTR


def test_randomized_parity_vs_explicit_checked_composition() -> None:
    rng = random.Random(20260420_743)

    for _ in range(6000):
        mode = rng.choice(["ok", "zero", "bad", "overflow_dense", "overflow_bytes"])

        if mode == "ok":
            row_count = rng.randint(1, 1 << 20)
            lane_count = rng.randint(1, 1 << 20)
        elif mode == "zero":
            row_count = rng.choice([0, rng.randint(1, 1 << 20)])
            lane_count = 0 if row_count != 0 else rng.randint(0, 1 << 20)
        elif mode == "bad":
            row_count = -rng.randint(1, 1 << 20) if rng.randint(0, 1) else rng.randint(0, 1 << 20)
            lane_count = -rng.randint(1, 1 << 20) if rng.randint(0, 1) else rng.randint(0, 1 << 20)
            if row_count >= 0 and lane_count >= 0:
                row_count = -1
        elif mode == "overflow_dense":
            row_count = I64_MAX - rng.randint(0, 4096)
            lane_count = rng.randint(2, 4096)
        else:
            dense = (I64_MAX // 8) - rng.randint(0, 32)
            row_count = dense
            lane_count = 3

        attn = [0x1111]
        ffn = [0x2222]
        residual = [0x3333]
        total_bytes = [0x4444]

        err_a = transformer_block_q16_compute_required_bytes_checked(
            row_count,
            lane_count,
            attn,
            ffn,
            residual,
            total_bytes,
        )

        err_b, expected = explicit_checked_composition(row_count, lane_count)

        assert err_a == err_b
        if err_a == BLOCK_Q16_OK:
            assert (attn[0], ffn[0], residual[0], total_bytes[0]) == expected
        elif row_count != 0 and lane_count != 0:
            assert attn[0] == 0x1111
            assert ffn[0] == 0x2222
            assert residual[0] == 0x3333
            assert total_bytes[0] == 0x4444


if __name__ == "__main__":
    test_source_contains_iq743_signature_and_checked_math()
    test_targeted_vectors_and_no_partial_diagnostics()
    test_null_pointer_contract()
    test_randomized_parity_vs_explicit_checked_composition()
    print("ok")
