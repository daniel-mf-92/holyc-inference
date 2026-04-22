#!/usr/bin/env python3
"""Parity harness for IQ-1053 forward preflight-only contract helper."""

from __future__ import annotations

from pathlib import Path
import random

MODEL_Q16_OK = 0
MODEL_Q16_ERR_NULL_PTR = 1
MODEL_Q16_ERR_BAD_PARAM = 2
MODEL_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


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


def inference_forward_q16_checked_nopartial_commit_only_preflight_only_model(
    embed_capacity: int,
    block_attn_capacity: int,
    block_ffn_capacity: int,
    block_count: int,
    gamma_attn_capacity: int,
    gamma_ffn_capacity: int,
    final_norm_gamma_capacity: int,
    workspace_capacity: int,
    logits_capacity: int,
    row_count: int,
    lane_count: int,
    out_dense_cells: list[int],
    out_required_workspace_cells: list[int],
    out_required_block_tensor_cells: list[int],
    out_required_logits_cells: list[int],
) -> int:
    if any(
        value < 0
        for value in (
            embed_capacity,
            block_attn_capacity,
            block_ffn_capacity,
            block_count,
            gamma_attn_capacity,
            gamma_ffn_capacity,
            final_norm_gamma_capacity,
            workspace_capacity,
            logits_capacity,
            row_count,
            lane_count,
        )
    ):
        return MODEL_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        out_dense_cells[0] = 0
        out_required_workspace_cells[0] = 0
        out_required_block_tensor_cells[0] = 0
        out_required_logits_cells[0] = 0
        return MODEL_Q16_OK

    err, dense_cells = try_mul_i64_checked(row_count, lane_count)
    if err != MODEL_Q16_OK:
        return err
    err, required_workspace_cells = try_mul_i64_checked(dense_cells, 2)
    if err != MODEL_Q16_OK:
        return err
    err, required_block_tensor_cells = try_mul_i64_checked(block_count, dense_cells)
    if err != MODEL_Q16_OK:
        return err
    required_logits_cells = dense_cells

    if (
        gamma_attn_capacity < lane_count
        or gamma_ffn_capacity < lane_count
        or final_norm_gamma_capacity < lane_count
        or embed_capacity < dense_cells
        or block_attn_capacity < required_block_tensor_cells
        or block_ffn_capacity < required_block_tensor_cells
        or workspace_capacity < required_workspace_cells
        or logits_capacity < required_logits_cells
    ):
        return MODEL_Q16_ERR_BAD_PARAM

    out_dense_cells[0] = dense_cells
    out_required_workspace_cells[0] = required_workspace_cells
    out_required_block_tensor_cells[0] = required_block_tensor_cells
    out_required_logits_cells[0] = required_logits_cells
    return MODEL_Q16_OK


def explicit_preflight_composition(
    block_count: int,
    row_count: int,
    lane_count: int,
) -> tuple[int, int, int]:
    err, dense = try_mul_i64_checked(row_count, lane_count)
    if err != MODEL_Q16_OK:
        return err, 0, 0
    err, workspace = try_mul_i64_checked(dense, 2)
    if err != MODEL_Q16_OK:
        return err, 0, 0
    err, block_tensor = try_mul_i64_checked(block_count, dense)
    if err != MODEL_Q16_OK:
        return err, 0, 0
    return MODEL_Q16_OK, workspace, block_tensor


def test_source_contains_preflight_only_symbol() -> None:
    source = Path("src/model/model.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "snapshot_embed_capacity = embed_capacity;" in body
    assert "ModelTryMulI64Checked(row_count," in body
    assert "ModelTryMulI64Checked(staged_dense_cells," in body
    assert "if (embed_capacity < staged_dense_cells)" in body


def test_known_vector_contract_outputs() -> None:
    out_dense = [111]
    out_workspace = [222]
    out_block = [333]
    out_logits = [444]

    err = inference_forward_q16_checked_nopartial_commit_only_preflight_only_model(
        embed_capacity=128,
        block_attn_capacity=256,
        block_ffn_capacity=256,
        block_count=4,
        gamma_attn_capacity=8,
        gamma_ffn_capacity=8,
        final_norm_gamma_capacity=8,
        workspace_capacity=64,
        logits_capacity=32,
        row_count=4,
        lane_count=8,
        out_dense_cells=out_dense,
        out_required_workspace_cells=out_workspace,
        out_required_block_tensor_cells=out_block,
        out_required_logits_cells=out_logits,
    )

    assert err == MODEL_Q16_OK
    assert out_dense == [32]
    assert out_workspace == [64]
    assert out_block == [128]
    assert out_logits == [32]


def test_error_path_preserves_outputs() -> None:
    out_dense = [901]
    out_workspace = [902]
    out_block = [903]
    out_logits = [904]

    err = inference_forward_q16_checked_nopartial_commit_only_preflight_only_model(
        embed_capacity=31,
        block_attn_capacity=32,
        block_ffn_capacity=32,
        block_count=1,
        gamma_attn_capacity=8,
        gamma_ffn_capacity=8,
        final_norm_gamma_capacity=8,
        workspace_capacity=64,
        logits_capacity=32,
        row_count=4,
        lane_count=8,
        out_dense_cells=out_dense,
        out_required_workspace_cells=out_workspace,
        out_required_block_tensor_cells=out_block,
        out_required_logits_cells=out_logits,
    )

    assert err == MODEL_Q16_ERR_BAD_PARAM
    assert out_dense == [901]
    assert out_workspace == [902]
    assert out_block == [903]
    assert out_logits == [904]


def test_randomized_preflight_parity() -> None:
    rng = random.Random(20260422_1053)

    for _ in range(1500):
        row_count = rng.randint(0, 128)
        lane_count = rng.randint(0, 128)
        block_count = rng.randint(0, 32)

        if row_count == 0 or lane_count == 0:
            dense = 0
            workspace_req = 0
            block_req = 0
            logits_req = 0
        else:
            dense = row_count * lane_count
            workspace_req = dense * 2
            block_req = dense * block_count
            logits_req = dense

        slack = rng.randint(0, 9)
        embed_capacity = dense + slack
        block_attn_capacity = block_req + slack
        block_ffn_capacity = block_req + slack
        workspace_capacity = workspace_req + slack
        logits_capacity = logits_req + slack

        gamma_attn_capacity = lane_count + rng.randint(0, 3)
        gamma_ffn_capacity = lane_count + rng.randint(0, 3)
        final_gamma_capacity = lane_count + rng.randint(0, 3)

        out_dense = [501]
        out_workspace = [502]
        out_block = [503]
        out_logits = [504]

        err = inference_forward_q16_checked_nopartial_commit_only_preflight_only_model(
            embed_capacity,
            block_attn_capacity,
            block_ffn_capacity,
            block_count,
            gamma_attn_capacity,
            gamma_ffn_capacity,
            final_gamma_capacity,
            workspace_capacity,
            logits_capacity,
            row_count,
            lane_count,
            out_dense,
            out_workspace,
            out_block,
            out_logits,
        )

        if row_count == 0 or lane_count == 0:
            assert err == MODEL_Q16_OK
            assert out_dense == [0]
            assert out_workspace == [0]
            assert out_block == [0]
            assert out_logits == [0]
            continue

        exp_err, exp_workspace, exp_block = explicit_preflight_composition(
            block_count,
            row_count,
            lane_count,
        )

        assert exp_err == MODEL_Q16_OK
        assert err == MODEL_Q16_OK
        assert out_dense == [dense]
        assert out_workspace == [exp_workspace]
        assert out_block == [exp_block]
        assert out_logits == [dense]
