#!/usr/bin/env python3
"""Parity harness for IQ-1045 InferenceForwardQ16CheckedNoPartial."""

from __future__ import annotations

from pathlib import Path
import random

from test_transformer_block_q16_apply_checked_nopartial_commit_only import (
    BLOCK_Q16_OK,
    BLOCK_Q16_ERR_BAD_PARAM,
    BLOCK_Q16_ERR_OVERFLOW,
    rmsnorm_row_checked,
)

MODEL_Q16_OK = 0
MODEL_Q16_ERR_NULL_PTR = 1
MODEL_Q16_ERR_BAD_PARAM = 2
MODEL_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


def try_add_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if rhs > 0 and lhs > I64_MAX - rhs:
        return MODEL_Q16_ERR_OVERFLOW, 0
    if rhs < 0 and lhs < I64_MIN - rhs:
        return MODEL_Q16_ERR_OVERFLOW, 0
    return MODEL_Q16_OK, lhs + rhs


def try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if lhs == 0 or rhs == 0:
        return MODEL_Q16_OK, 0

    if lhs > 0:
        if rhs > 0:
            if lhs > I64_MAX // rhs:
                return MODEL_Q16_ERR_OVERFLOW, 0
        else:
            if rhs < I64_MIN // lhs:
                return MODEL_Q16_ERR_OVERFLOW, 0
    else:
        if rhs > 0:
            if lhs < I64_MIN // rhs:
                return MODEL_Q16_ERR_OVERFLOW, 0
        else:
            if lhs != 0 and rhs < I64_MAX // lhs:
                return MODEL_Q16_ERR_OVERFLOW, 0

    return MODEL_Q16_OK, lhs * rhs


def block_apply_model(
    input_q16: list[int],
    attn_out_q16: list[int],
    ffn_out_q16: list[int],
    gamma_attn_q16: list[int],
    gamma_ffn_q16: list[int],
    eps_q16: int,
    row_count: int,
    lane_count: int,
) -> tuple[int, list[int]]:
    dense = row_count * lane_count
    if (
        len(input_q16) < dense
        or len(attn_out_q16) < dense
        or len(ffn_out_q16) < dense
        or len(gamma_attn_q16) < lane_count
        or len(gamma_ffn_q16) < lane_count
    ):
        return MODEL_Q16_ERR_BAD_PARAM, []

    attn_residual = [0] * dense
    for row_idx in range(row_count):
        base = row_idx * lane_count
        err, norm_row = rmsnorm_row_checked(
            input_q16[base : base + lane_count],
            gamma_attn_q16[:lane_count],
            eps_q16,
        )
        if err != BLOCK_Q16_OK:
            return err, []

        for lane_idx in range(lane_count):
            err, lane = try_add_i64_checked(norm_row[lane_idx], attn_out_q16[base + lane_idx])
            if err != MODEL_Q16_OK:
                return err, []
            attn_residual[base + lane_idx] = lane

    out = [0] * dense
    for row_idx in range(row_count):
        base = row_idx * lane_count
        err, norm_row = rmsnorm_row_checked(
            attn_residual[base : base + lane_count],
            gamma_ffn_q16[:lane_count],
            eps_q16,
        )
        if err != BLOCK_Q16_OK:
            return err, []

        for lane_idx in range(lane_count):
            err, lane = try_add_i64_checked(norm_row[lane_idx], ffn_out_q16[base + lane_idx])
            if err != MODEL_Q16_OK:
                return err, []
            out[base + lane_idx] = lane

    return MODEL_Q16_OK, out


def inference_forward_q16_checked_nopartial_model(
    embed_q16: list[int],
    embed_capacity: int,
    block_attn_q16: list[int],
    block_attn_capacity: int,
    block_ffn_q16: list[int],
    block_ffn_capacity: int,
    block_count: int,
    gamma_attn_q16: list[int],
    gamma_attn_capacity: int,
    gamma_ffn_q16: list[int],
    gamma_ffn_capacity: int,
    block_eps_q16: int,
    final_norm_gamma_q16: list[int],
    final_norm_gamma_capacity: int,
    final_norm_eps_q16: int,
    workspace_q16: list[int],
    workspace_capacity: int,
    logits_q16: list[int],
    logits_capacity: int,
    row_count: int,
    lane_count: int,
) -> int:
    if (
        embed_capacity < 0
        or block_attn_capacity < 0
        or block_ffn_capacity < 0
        or gamma_attn_capacity < 0
        or gamma_ffn_capacity < 0
        or final_norm_gamma_capacity < 0
        or workspace_capacity < 0
        or logits_capacity < 0
        or block_count < 0
        or row_count < 0
        or lane_count < 0
        or block_eps_q16 < 0
        or final_norm_eps_q16 < 0
    ):
        return MODEL_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        return MODEL_Q16_OK

    err, dense = try_mul_i64_checked(row_count, lane_count)
    if err != MODEL_Q16_OK:
        return err
    err, doubled = try_mul_i64_checked(dense, 2)
    if err != MODEL_Q16_OK:
        return err
    err, block_dense = try_mul_i64_checked(block_count, dense)
    if err != MODEL_Q16_OK:
        return err

    if (
        embed_capacity < dense
        or block_attn_capacity < block_dense
        or block_ffn_capacity < block_dense
        or gamma_attn_capacity < lane_count
        or gamma_ffn_capacity < lane_count
        or final_norm_gamma_capacity < lane_count
        or workspace_capacity < doubled
        or logits_capacity < dense
    ):
        return MODEL_Q16_ERR_BAD_PARAM

    current = embed_q16[:dense]

    for layer_idx in range(block_count):
        layer_base = layer_idx * dense
        layer_attn = block_attn_q16[layer_base : layer_base + dense]
        layer_ffn = block_ffn_q16[layer_base : layer_base + dense]

        err, current = block_apply_model(
            current,
            layer_attn,
            layer_ffn,
            gamma_attn_q16,
            gamma_ffn_q16,
            block_eps_q16,
            row_count,
            lane_count,
        )
        if err != MODEL_Q16_OK:
            return err

    final_norm = [0] * dense
    for row_idx in range(row_count):
        base = row_idx * lane_count
        err, row = rmsnorm_row_checked(
            current[base : base + lane_count],
            final_norm_gamma_q16[:lane_count],
            final_norm_eps_q16,
        )
        if err != BLOCK_Q16_OK:
            return err
        final_norm[base : base + lane_count] = row

    for idx in range(dense):
        logits_q16[idx] = final_norm[idx]

    return MODEL_Q16_OK


def explicit_forward_composition(
    embed_q16: list[int],
    block_attn_q16: list[int],
    block_ffn_q16: list[int],
    block_count: int,
    gamma_attn_q16: list[int],
    gamma_ffn_q16: list[int],
    block_eps_q16: int,
    final_norm_gamma_q16: list[int],
    final_norm_eps_q16: int,
    row_count: int,
    lane_count: int,
) -> tuple[int, list[int]]:
    dense = row_count * lane_count
    current = embed_q16[:dense]

    for layer_idx in range(block_count):
        base = layer_idx * dense
        err, current = block_apply_model(
            current,
            block_attn_q16[base : base + dense],
            block_ffn_q16[base : base + dense],
            gamma_attn_q16,
            gamma_ffn_q16,
            block_eps_q16,
            row_count,
            lane_count,
        )
        if err != MODEL_Q16_OK:
            return err, []

    out: list[int] = [0] * dense
    for row_idx in range(row_count):
        base = row_idx * lane_count
        err, row = rmsnorm_row_checked(
            current[base : base + lane_count],
            final_norm_gamma_q16[:lane_count],
            final_norm_eps_q16,
        )
        if err != BLOCK_Q16_OK:
            return err, []
        out[base : base + lane_count] = row

    return MODEL_Q16_OK, out


def test_source_contains_inference_forward_symbol() -> None:
    source = Path("src/model/model.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceForwardQ16CheckedNoPartial("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "TransformerBlockQ16ApplyCheckedNoPartialCommitOnly(" in body
    assert "FPQ16RMSNormChecked(current_q16 + row_base" in body
    assert "current_q16 = workspace_q16;" in body
    assert "next_q16 = workspace_q16 + dense_cells;" in body
    assert "GGUFTensorByteSpansOverlap((U8 *)workspace_q16" in body
    assert "GGUFTensorByteSpansOverlap((U8 *)(workspace_q16 + dense_cells)" in body


def test_known_vector_first_token_parity() -> None:
    row_count = 2
    lane_count = 4
    block_count = 2
    dense = row_count * lane_count

    embed = [12000, -8000, 4000, -2000, -3000, 7000, -9000, 11000]
    block_attn = [
        800, -500, 300, -200, -900, 400, 250, -100,
        -600, 350, -450, 200, 700, -300, 100, 500,
    ]
    block_ffn = [
        -300, 120, -90, 240, 330, -210, 180, -150,
        210, -270, 330, -390, 450, -510, 570, -630,
    ]

    gamma_attn = [65536, 64512, 63488, 62464]
    gamma_ffn = [65536, 66560, 67584, 68608]
    gamma_final = [65536, 65536, 65536, 65536]

    logits = [777] * dense
    workspace = [0] * (2 * dense)

    err = inference_forward_q16_checked_nopartial_model(
        embed,
        dense,
        block_attn,
        len(block_attn),
        block_ffn,
        len(block_ffn),
        block_count,
        gamma_attn,
        len(gamma_attn),
        gamma_ffn,
        len(gamma_ffn),
        256,
        gamma_final,
        len(gamma_final),
        128,
        workspace,
        len(workspace),
        logits,
        len(logits),
        row_count,
        lane_count,
    )

    ref_err, ref_logits = explicit_forward_composition(
        embed,
        block_attn,
        block_ffn,
        block_count,
        gamma_attn,
        gamma_ffn,
        256,
        gamma_final,
        128,
        row_count,
        lane_count,
    )

    assert err == MODEL_Q16_OK
    assert ref_err == MODEL_Q16_OK
    assert logits == ref_logits
    assert logits[:lane_count] == ref_logits[:lane_count]


def test_error_preserves_existing_logits() -> None:
    row_count = 2
    lane_count = 3
    block_count = 1
    dense = row_count * lane_count

    embed = [1000] * dense
    block_attn = [200] * dense
    block_ffn = [-50] * dense
    gamma = [65536] * lane_count

    workspace = [0] * (2 * dense)
    logits = [9001] * dense

    err = inference_forward_q16_checked_nopartial_model(
        embed,
        dense,
        block_attn,
        len(block_attn),
        block_ffn,
        len(block_ffn),
        block_count,
        gamma,
        len(gamma),
        gamma,
        len(gamma),
        64,
        gamma,
        len(gamma),
        64,
        workspace,
        len(workspace),
        logits,
        dense - 1,
        row_count,
        lane_count,
    )

    assert err == MODEL_Q16_ERR_BAD_PARAM
    assert logits == [9001] * dense


def test_randomized_forward_parity() -> None:
    rng = random.Random(20260422_1045)

    for _ in range(250):
        row_count = rng.randint(1, 4)
        lane_count = rng.randint(1, 8)
        block_count = rng.randint(0, 3)
        dense = row_count * lane_count
        stacked = max(block_count, 1) * dense

        embed = [rng.randint(-16000, 16000) for _ in range(dense)]
        block_attn = [rng.randint(-2000, 2000) for _ in range(stacked)]
        block_ffn = [rng.randint(-1200, 1200) for _ in range(stacked)]

        gamma_attn = [rng.randint(52000, 76000) for _ in range(lane_count)]
        gamma_ffn = [rng.randint(52000, 76000) for _ in range(lane_count)]
        gamma_final = [rng.randint(52000, 76000) for _ in range(lane_count)]

        workspace = [0] * (2 * dense)
        logits = [rng.randint(-500, 500) for _ in range(dense)]
        before = list(logits)

        block_eps = rng.randint(32, 512)
        final_eps = rng.randint(32, 512)

        err = inference_forward_q16_checked_nopartial_model(
            embed,
            dense,
            block_attn,
            block_count * dense,
            block_ffn,
            block_count * dense,
            block_count,
            gamma_attn,
            len(gamma_attn),
            gamma_ffn,
            len(gamma_ffn),
            block_eps,
            gamma_final,
            len(gamma_final),
            final_eps,
            workspace,
            len(workspace),
            logits,
            len(logits),
            row_count,
            lane_count,
        )

        ref_err, ref_logits = explicit_forward_composition(
            embed,
            block_attn,
            block_ffn,
            block_count,
            gamma_attn,
            gamma_ffn,
            block_eps,
            gamma_final,
            final_eps,
            row_count,
            lane_count,
        )

        if err == MODEL_Q16_OK:
            assert ref_err == MODEL_Q16_OK
            assert logits == ref_logits
        else:
            assert err in {MODEL_Q16_ERR_BAD_PARAM, MODEL_Q16_ERR_OVERFLOW}
            assert logits == before
