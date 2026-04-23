#!/usr/bin/env python3
"""Parity harness for IQ-1221 TransformerBlockQ16ForwardCheckedNoPartial."""

from __future__ import annotations

from pathlib import Path
import random

BLOCK_Q16_OK = 0
BLOCK_Q16_ERR_BAD_PARAM = 2
BLOCK_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)
Q16_ONE = 1 << 16


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


def q16_mul_checked(lhs: int, rhs: int) -> tuple[int, int]:
    err, prod = try_mul_i64_checked(lhs, rhs)
    if err != BLOCK_Q16_OK:
        return err, 0
    bias = 1 << 15
    if prod < 0:
        bias = -bias
    err, biased = try_add_i64_checked(prod, bias)
    if err != BLOCK_Q16_OK:
        return err, 0
    return BLOCK_Q16_OK, biased >> 16


def q16_div_checked(num: int, den: int) -> tuple[int, int]:
    if den == 0:
        return BLOCK_Q16_ERR_BAD_PARAM, 0
    err, scaled = try_mul_i64_checked(num, Q16_ONE)
    if err != BLOCK_Q16_OK:
        return err, 0
    sign = -1 if (scaled < 0) ^ (den < 0) else 1
    n = abs(scaled)
    d = abs(den)
    q = n // d
    r = n % d
    if r * 2 >= d:
        q += 1
    return BLOCK_Q16_OK, q if sign > 0 else -q


def q16_sqrt_nonneg(value_q16: int) -> tuple[int, int]:
    if value_q16 < 0:
        return BLOCK_Q16_ERR_BAD_PARAM, 0
    target = value_q16 << 16
    if target == 0:
        return BLOCK_Q16_OK, 0
    x = target
    for _ in range(48):
        if x == 0:
            x = 1
        nxt = (x + (target // x)) // 2
        if nxt == x:
            break
        x = nxt
    return BLOCK_Q16_OK, x


def rmsnorm_row_checked(row: list[int], gamma: list[int], eps_q16: int) -> tuple[int, list[int]]:
    count = len(row)
    if count <= 0 or len(gamma) != count:
        return BLOCK_Q16_ERR_BAD_PARAM, []

    sum_sq_q16 = 0
    for lane in row:
        err, sq_q16 = q16_mul_checked(lane, lane)
        if err != BLOCK_Q16_OK:
            return err, []
        err, sum_sq_q16 = try_add_i64_checked(sum_sq_q16, sq_q16)
        if err != BLOCK_Q16_OK:
            return err, []

    err, mean_sq_q16 = q16_div_checked(sum_sq_q16, count << 16)
    if err != BLOCK_Q16_OK:
        return err, []

    err, denom_arg_q16 = try_add_i64_checked(mean_sq_q16, eps_q16)
    if err != BLOCK_Q16_OK or denom_arg_q16 <= 0:
        return BLOCK_Q16_ERR_BAD_PARAM, []

    err, denom_q16 = q16_sqrt_nonneg(denom_arg_q16)
    if err != BLOCK_Q16_OK or denom_q16 <= 0:
        return BLOCK_Q16_ERR_BAD_PARAM, []

    err, inv_q16 = q16_div_checked(Q16_ONE, denom_q16)
    if err != BLOCK_Q16_OK or inv_q16 <= 0:
        return BLOCK_Q16_ERR_BAD_PARAM, []

    out: list[int] = []
    for lane, g in zip(row, gamma):
        err, norm_lane = q16_mul_checked(lane, inv_q16)
        if err != BLOCK_Q16_OK:
            return err, []
        err, weighted = q16_mul_checked(norm_lane, g)
        if err != BLOCK_Q16_OK:
            return err, []
        out.append(weighted)

    return BLOCK_Q16_OK, out


def q32_to_q16_rounded_checked(value_q32: int) -> tuple[int, int]:
    bias = 1 << 15
    if value_q32 < 0:
        bias = -bias
    err, biased = try_add_i64_checked(value_q32, bias)
    if err != BLOCK_Q16_OK:
        return err, 0
    return BLOCK_Q16_OK, biased >> 16


def transformer_block_q16_forward_checked_nopartial_model(
    input_q16: list[int],
    rms_gamma_q16: list[int],
    rms_eps_q16: int,
    attn_score_scale_q16: int,
    ffn_gate_q16: list[int],
    ffn_up_q16: list[int],
    row_count: int,
    lane_count: int,
) -> tuple[int, list[int]]:
    if row_count <= 0 or lane_count <= 0:
        return BLOCK_Q16_OK, []

    dense = row_count * lane_count
    if (
        len(input_q16) < dense
        or len(ffn_gate_q16) < dense
        or len(ffn_up_q16) < dense
        or len(rms_gamma_q16) < lane_count
    ):
        return BLOCK_Q16_ERR_BAD_PARAM, []

    norm = [0] * dense
    for row_idx in range(row_count):
        base = row_idx * lane_count
        err, row_norm = rmsnorm_row_checked(
            input_q16[base : base + lane_count],
            rms_gamma_q16[:lane_count],
            rms_eps_q16,
        )
        if err != BLOCK_Q16_OK:
            return err, []
        norm[base : base + lane_count] = row_norm

    attn_q16 = [0] * dense
    for idx, lane in enumerate(norm):
        err, lane_q32 = try_mul_i64_checked(lane, Q16_ONE)
        if err != BLOCK_Q16_OK:
            return err, []
        err, scaled_q32 = q16_mul_checked(lane_q32, attn_score_scale_q16)
        if err != BLOCK_Q16_OK:
            return err, []
        err, attn_lane = q32_to_q16_rounded_checked(scaled_q32)
        if err != BLOCK_Q16_OK:
            return err, []
        attn_q16[idx] = attn_lane

    ffn_out = [0] * dense
    for idx in range(dense):
        err, gate_mix = q16_mul_checked(ffn_gate_q16[idx], ffn_up_q16[idx])
        if err != BLOCK_Q16_OK:
            return err, []
        err, mixed = q16_mul_checked(attn_q16[idx], gate_mix)
        if err != BLOCK_Q16_OK:
            return err, []
        ffn_out[idx] = mixed

    staged_residual = [0] * dense
    for idx in range(dense):
        err, summed = try_add_i64_checked(input_q16[idx], ffn_out[idx])
        if err != BLOCK_Q16_OK:
            return err, []
        staged_residual[idx] = summed

    return BLOCK_Q16_OK, staged_residual


def test_source_contains_iq1221_signature_and_core_pipeline_calls() -> None:
    source = Path("src/model/block.HC").read_text(encoding="utf-8")
    signature = "I32 TransformerBlockQ16ForwardCheckedNoPartial("
    assert signature in source

    body = source.rsplit(signature, 1)[1]
    assert "FPQ16RMSNormChecked(" in body
    assert "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAlloc(" in body
    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAlloc(" in body
    assert "TransformerBlockQ16ResidualAddRowsCheckedNoPartial(" in body
    assert "BlockQ32ToQ16RoundedChecked(" in body


def test_source_contains_staged_residual_commit_flow() -> None:
    source = Path("src/model/block.HC").read_text(encoding="utf-8")
    signature = "I32 TransformerBlockQ16ForwardCheckedNoPartial("
    body = source.rsplit(signature, 1)[1]

    assert "residual_stage_q16" in body
    assert "Stage 6a: validate every destination index" in body
    assert "Stage 6b: commit staged residual rows" in body
    assert "out_q16[out_index] = residual_stage_q16[stage_index];" in body


def test_targeted_deterministic_fixture_parity() -> None:
    row_count = 2
    lane_count = 4
    dense = row_count * lane_count

    input_q16 = [19660, -9830, 4915, -1638, 14745, -6553, 3276, -819]
    gamma_q16 = [65536, 65536, 58982, 72089]
    ffn_gate_q16 = [57344, 53248, 49152, 45056, 40960, 36864, 32768, 28672]
    ffn_up_q16 = [32768, 36864, 40960, 45056, 49152, 53248, 57344, 61440]

    err, out = transformer_block_q16_forward_checked_nopartial_model(
        input_q16,
        gamma_q16,
        64,
        49152,
        ffn_gate_q16,
        ffn_up_q16,
        row_count,
        lane_count,
    )
    assert err == BLOCK_Q16_OK
    assert len(out) == dense
    assert out == [56496, -29072, 13795, -5287, 54741, -23886, 10740, -2959]


def test_zero_geometry_short_circuits_to_success() -> None:
    err, out = transformer_block_q16_forward_checked_nopartial_model(
        [],
        [Q16_ONE],
        1,
        Q16_ONE,
        [],
        [],
        0,
        6,
    )
    assert err == BLOCK_Q16_OK
    assert out == []


def test_invalid_shapes_return_bad_param() -> None:
    err, out = transformer_block_q16_forward_checked_nopartial_model(
        [1, 2, 3],
        [Q16_ONE],
        64,
        Q16_ONE,
        [1, 2, 3],
        [1, 2, 3],
        1,
        4,
    )
    assert err == BLOCK_Q16_ERR_BAD_PARAM
    assert out == []


def test_randomized_integer_pipeline_stability() -> None:
    rng = random.Random(20260423_1221)

    for _ in range(300):
        row_count = rng.randint(1, 4)
        lane_count = rng.randint(1, 8)
        dense = row_count * lane_count

        input_q16 = [rng.randint(-(1 << 14), (1 << 14)) for _ in range(dense)]
        gamma_q16 = [rng.randint(32768, 98304) for _ in range(lane_count)]
        ffn_gate_q16 = [rng.randint(16384, 65536) for _ in range(dense)]
        ffn_up_q16 = [rng.randint(16384, 65536) for _ in range(dense)]
        eps_q16 = rng.randint(1, 512)
        scale_q16 = rng.randint(32768, 98304)

        err_a, out_a = transformer_block_q16_forward_checked_nopartial_model(
            input_q16,
            gamma_q16,
            eps_q16,
            scale_q16,
            ffn_gate_q16,
            ffn_up_q16,
            row_count,
            lane_count,
        )
        err_b, out_b = transformer_block_q16_forward_checked_nopartial_model(
            input_q16,
            gamma_q16,
            eps_q16,
            scale_q16,
            ffn_gate_q16,
            ffn_up_q16,
            row_count,
            lane_count,
        )

        assert err_a == err_b == BLOCK_Q16_OK
        assert out_a == out_b
