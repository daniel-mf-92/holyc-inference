#!/usr/bin/env python3
"""Parity harness for IQ-742 out-row-base helper in block residual rows path."""

from __future__ import annotations

import random
from pathlib import Path

BLOCK_Q16_OK = 0
BLOCK_Q16_ERR_NULL_PTR = 1
BLOCK_Q16_ERR_BAD_PARAM = 2
BLOCK_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


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


def transformer_block_q16_residual_add_rows_compute_out_row_base_checked(
    row_index: int,
    out_row_stride_q16: int,
    out_row_base_ptr,
) -> tuple[int, int]:
    if out_row_base_ptr is None:
        return BLOCK_Q16_ERR_NULL_PTR, 0

    if row_index < 0 or out_row_stride_q16 < 0:
        return BLOCK_Q16_ERR_BAD_PARAM, 0

    err, out_row_base = try_mul_i64_checked(row_index, out_row_stride_q16)
    if err != BLOCK_Q16_OK:
        return err, 0

    return BLOCK_Q16_OK, out_row_base


def explicit_out_row_base_checked(row_index: int, out_row_stride_q16: int) -> tuple[int, int]:
    if row_index < 0 or out_row_stride_q16 < 0:
        return BLOCK_Q16_ERR_BAD_PARAM, 0

    err, out_row_base = try_mul_i64_checked(row_index, out_row_stride_q16)
    if err != BLOCK_Q16_OK:
        return err, 0

    return BLOCK_Q16_OK, out_row_base


def test_source_contains_iq742_helper_and_callsite() -> None:
    source = Path("src/model/block.HC").read_text(encoding="utf-8")
    assert "I32 TransformerBlockQ16ResidualAddRowsComputeOutRowBaseChecked(" in source

    helper_body = source.split("I32 TransformerBlockQ16ResidualAddRowsComputeOutRowBaseChecked(", 1)[1]
    assert "if (row_index < 0 || out_row_stride_q16 < 0)" in helper_body
    assert "status = BlockTryMulI64Checked(row_index," in helper_body

    rows_body = source.split("I32 TransformerBlockQ16ResidualAddRowsCheckedNoPartial(", 1)[1]
    assert "TransformerBlockQ16ResidualAddRowsComputeOutRowBaseChecked(row_index," in rows_body


def test_targeted_cases_match_explicit_checked_composition() -> None:
    cases = [
        (0, 0, BLOCK_Q16_OK),
        (0, 15, BLOCK_Q16_OK),
        (9, 13, BLOCK_Q16_OK),
        (I64_MAX, 1, BLOCK_Q16_OK),
        (I64_MAX // 3 + 1, 3, BLOCK_Q16_ERR_OVERFLOW),
        (-1, 1, BLOCK_Q16_ERR_BAD_PARAM),
        (1, -1, BLOCK_Q16_ERR_BAD_PARAM),
    ]

    for row_index, out_row_stride_q16, expected_err in cases:
        err_a, row_base_a = transformer_block_q16_residual_add_rows_compute_out_row_base_checked(
            row_index,
            out_row_stride_q16,
            1,
        )
        err_b, row_base_b = explicit_out_row_base_checked(
            row_index,
            out_row_stride_q16,
        )

        assert err_a == err_b == expected_err
        if expected_err == BLOCK_Q16_OK:
            assert row_base_a == row_base_b


def test_null_pointer_contract() -> None:
    err, out_row_base = transformer_block_q16_residual_add_rows_compute_out_row_base_checked(
        1,
        4,
        None,
    )
    assert err == BLOCK_Q16_ERR_NULL_PTR
    assert out_row_base == 0


def test_randomized_parity_vs_explicit_checked_composition() -> None:
    rng = random.Random(20260420_742)

    for _ in range(3000):
        mode = rng.choice(["ok", "bad", "overflow"]) 
        if mode == "ok":
            row_index = rng.randint(0, 1 << 24)
            out_row_stride_q16 = rng.randint(0, 1 << 24)
        elif mode == "bad":
            row_index = rng.choice([-(rng.randint(1, 1 << 24)), rng.randint(0, 1 << 24)])
            out_row_stride_q16 = rng.choice([-(rng.randint(1, 1 << 24)), rng.randint(0, 1 << 24)])
            if row_index >= 0 and out_row_stride_q16 >= 0:
                row_index = -1
        else:
            row_index = I64_MAX - rng.randint(0, 4096)
            out_row_stride_q16 = rng.randint(2, 4096)

        err_a, row_base_a = transformer_block_q16_residual_add_rows_compute_out_row_base_checked(
            row_index,
            out_row_stride_q16,
            1,
        )
        err_b, row_base_b = explicit_out_row_base_checked(
            row_index,
            out_row_stride_q16,
        )

        assert err_a == err_b
        if err_a == BLOCK_Q16_OK:
            assert row_base_a == row_base_b


if __name__ == "__main__":
    test_source_contains_iq742_helper_and_callsite()
    test_targeted_cases_match_explicit_checked_composition()
    test_null_pointer_contract()
    test_randomized_parity_vs_explicit_checked_composition()
    print("ok")
