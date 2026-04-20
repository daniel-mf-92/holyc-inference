#!/usr/bin/env python3
"""Parity harness for IQ-744 TransformerBlockQ16ApplyCheckedNoPartialPreflightOnly."""

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


def transformer_block_q16_apply_checked_nopartial_preflight_only(
    row_count: int,
    lane_count: int,
    out_required_qkv_cells,
    out_required_attn_scores,
    out_required_ffn_cells,
    out_required_out_cells,
    out_required_stage_bytes,
) -> int:
    if (
        out_required_qkv_cells is None
        or out_required_attn_scores is None
        or out_required_ffn_cells is None
        or out_required_out_cells is None
        or out_required_stage_bytes is None
    ):
        return BLOCK_Q16_ERR_NULL_PTR

    if row_count < 0 or lane_count < 0:
        return BLOCK_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        out_required_qkv_cells[0] = 0
        out_required_attn_scores[0] = 0
        out_required_ffn_cells[0] = 0
        out_required_out_cells[0] = 0
        out_required_stage_bytes[0] = 0
        return BLOCK_Q16_OK

    err, dense_cells = try_mul_i64_checked(row_count, lane_count)
    if err != BLOCK_Q16_OK:
        return err

    err, required_qkv_cells = try_mul_i64_checked(dense_cells, 3)
    if err != BLOCK_Q16_OK:
        return err

    err, required_attn_scores = try_mul_i64_checked(row_count, row_count)
    if err != BLOCK_Q16_OK:
        return err

    required_ffn_cells = dense_cells
    required_out_cells = dense_cells

    err, required_stage_cells = try_add_i64_checked(required_qkv_cells, required_attn_scores)
    if err != BLOCK_Q16_OK:
        return err
    err, required_stage_cells = try_add_i64_checked(required_stage_cells, required_ffn_cells)
    if err != BLOCK_Q16_OK:
        return err
    err, required_stage_cells = try_add_i64_checked(required_stage_cells, required_out_cells)
    if err != BLOCK_Q16_OK:
        return err

    err, required_stage_bytes = try_mul_i64_checked(required_stage_cells, 8)
    if err != BLOCK_Q16_OK:
        return err

    out_required_qkv_cells[0] = required_qkv_cells
    out_required_attn_scores[0] = required_attn_scores
    out_required_ffn_cells[0] = required_ffn_cells
    out_required_out_cells[0] = required_out_cells
    out_required_stage_bytes[0] = required_stage_bytes
    return BLOCK_Q16_OK


def explicit_checked_composition(row_count: int, lane_count: int) -> tuple[int, tuple[int, int, int, int, int]]:
    if row_count < 0 or lane_count < 0:
        return BLOCK_Q16_ERR_BAD_PARAM, (0, 0, 0, 0, 0)

    if row_count == 0 or lane_count == 0:
        return BLOCK_Q16_OK, (0, 0, 0, 0, 0)

    err, dense_cells = try_mul_i64_checked(row_count, lane_count)
    if err != BLOCK_Q16_OK:
        return err, (0, 0, 0, 0, 0)

    err, required_qkv_cells = try_mul_i64_checked(dense_cells, 3)
    if err != BLOCK_Q16_OK:
        return err, (0, 0, 0, 0, 0)

    err, required_attn_scores = try_mul_i64_checked(row_count, row_count)
    if err != BLOCK_Q16_OK:
        return err, (0, 0, 0, 0, 0)

    required_ffn_cells = dense_cells
    required_out_cells = dense_cells

    err, required_stage_cells = try_add_i64_checked(required_qkv_cells, required_attn_scores)
    if err != BLOCK_Q16_OK:
        return err, (0, 0, 0, 0, 0)
    err, required_stage_cells = try_add_i64_checked(required_stage_cells, required_ffn_cells)
    if err != BLOCK_Q16_OK:
        return err, (0, 0, 0, 0, 0)
    err, required_stage_cells = try_add_i64_checked(required_stage_cells, required_out_cells)
    if err != BLOCK_Q16_OK:
        return err, (0, 0, 0, 0, 0)

    err, required_stage_bytes = try_mul_i64_checked(required_stage_cells, 8)
    if err != BLOCK_Q16_OK:
        return err, (0, 0, 0, 0, 0)

    return BLOCK_Q16_OK, (
        required_qkv_cells,
        required_attn_scores,
        required_ffn_cells,
        required_out_cells,
        required_stage_bytes,
    )


def test_source_contains_iq744_signature_and_diagnostics_fields() -> None:
    source = Path("src/model/block.HC").read_text(encoding="utf-8")
    signature = "I32 TransformerBlockQ16ApplyCheckedNoPartialPreflightOnly("
    assert signature in source

    body = source.split(signature, 1)[1]
    assert "required_qkv_cells" in body
    assert "required_attn_scores" in body
    assert "required_ffn_cells" in body
    assert "required_out_cells" in body
    assert "required_stage_bytes" in body


def test_targeted_vectors_and_no_partial_output_publish() -> None:
    cases = [
        (0, 0, BLOCK_Q16_OK),
        (0, 7, BLOCK_Q16_OK),
        (9, 0, BLOCK_Q16_OK),
        (5, 8, BLOCK_Q16_OK),
        (-1, 8, BLOCK_Q16_ERR_BAD_PARAM),
        (8, -1, BLOCK_Q16_ERR_BAD_PARAM),
        (I64_MAX, 2, BLOCK_Q16_ERR_OVERFLOW),
    ]

    for row_count, lane_count, expected_err in cases:
        qkv = [0x1111]
        attn = [0x2222]
        ffn = [0x3333]
        out = [0x4444]
        stage_bytes = [0x5555]

        err = transformer_block_q16_apply_checked_nopartial_preflight_only(
            row_count,
            lane_count,
            qkv,
            attn,
            ffn,
            out,
            stage_bytes,
        )
        assert err == expected_err

        if err == BLOCK_Q16_OK:
            _, expected = explicit_checked_composition(row_count, lane_count)
            assert (qkv[0], attn[0], ffn[0], out[0], stage_bytes[0]) == expected
        elif row_count != 0 and lane_count != 0:
            assert qkv[0] == 0x1111
            assert attn[0] == 0x2222
            assert ffn[0] == 0x3333
            assert out[0] == 0x4444
            assert stage_bytes[0] == 0x5555


def test_null_pointer_contract() -> None:
    err = transformer_block_q16_apply_checked_nopartial_preflight_only(1, 1, None, [0], [0], [0], [0])
    assert err == BLOCK_Q16_ERR_NULL_PTR

    err = transformer_block_q16_apply_checked_nopartial_preflight_only(1, 1, [0], None, [0], [0], [0])
    assert err == BLOCK_Q16_ERR_NULL_PTR

    err = transformer_block_q16_apply_checked_nopartial_preflight_only(1, 1, [0], [0], None, [0], [0])
    assert err == BLOCK_Q16_ERR_NULL_PTR

    err = transformer_block_q16_apply_checked_nopartial_preflight_only(1, 1, [0], [0], [0], None, [0])
    assert err == BLOCK_Q16_ERR_NULL_PTR

    err = transformer_block_q16_apply_checked_nopartial_preflight_only(1, 1, [0], [0], [0], [0], None)
    assert err == BLOCK_Q16_ERR_NULL_PTR


def test_randomized_parity_vs_explicit_checked_composition() -> None:
    rng = random.Random(20260420_744)

    for _ in range(7000):
        mode = rng.choice(["ok", "zero", "bad", "overflow_dense", "overflow_qkv", "overflow_attn", "overflow_bytes"])

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
        elif mode == "overflow_qkv":
            dense = (I64_MAX // 3) - rng.randint(0, 8)
            row_count = dense
            lane_count = 4
        elif mode == "overflow_attn":
            row_count = (1 << 32) + rng.randint(0, 1 << 20)
            lane_count = 1
        else:
            dense = (I64_MAX // 8) - rng.randint(0, 16)
            row_count = dense
            lane_count = 1

        qkv = [0x1111]
        attn = [0x2222]
        ffn = [0x3333]
        out = [0x4444]
        stage_bytes = [0x5555]

        err_a = transformer_block_q16_apply_checked_nopartial_preflight_only(
            row_count,
            lane_count,
            qkv,
            attn,
            ffn,
            out,
            stage_bytes,
        )

        err_b, expected = explicit_checked_composition(row_count, lane_count)

        assert err_a == err_b
        if err_a == BLOCK_Q16_OK:
            assert (qkv[0], attn[0], ffn[0], out[0], stage_bytes[0]) == expected
        elif row_count != 0 and lane_count != 0:
            assert qkv[0] == 0x1111
            assert attn[0] == 0x2222
            assert ffn[0] == 0x3333
            assert out[0] == 0x4444
            assert stage_bytes[0] == 0x5555


if __name__ == "__main__":
    test_source_contains_iq744_signature_and_diagnostics_fields()
    test_targeted_vectors_and_no_partial_output_publish()
    test_null_pointer_contract()
    test_randomized_parity_vs_explicit_checked_composition()
    print("ok")
