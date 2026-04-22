#!/usr/bin/env python3
"""Parity harness for InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-1054)."""

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


def preflight_only_reference(
    *,
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
) -> tuple[int, dict[str, int] | None]:
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
        return MODEL_Q16_ERR_BAD_PARAM, None

    if row_count == 0 or lane_count == 0:
        return MODEL_Q16_OK, {
            "dense_cells": 0,
            "required_workspace_cells": 0,
            "required_block_tensor_cells": 0,
            "required_logits_cells": 0,
        }

    err, dense_cells = try_mul_i64_checked(row_count, lane_count)
    if err != MODEL_Q16_OK:
        return err, None

    err, required_workspace_cells = try_mul_i64_checked(dense_cells, 2)
    if err != MODEL_Q16_OK:
        return err, None

    err, required_block_tensor_cells = try_mul_i64_checked(block_count, dense_cells)
    if err != MODEL_Q16_OK:
        return err, None

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
        return MODEL_Q16_ERR_BAD_PARAM, None

    return MODEL_Q16_OK, {
        "dense_cells": dense_cells,
        "required_workspace_cells": required_workspace_cells,
        "required_block_tensor_cells": required_block_tensor_cells,
        "required_logits_cells": required_logits_cells,
    }


def inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_reference(
    *,
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
    out_dense_cells: list[int] | None,
    out_required_workspace_cells: list[int] | None,
    out_required_block_tensor_cells: list[int] | None,
    out_required_logits_cells: list[int] | None,
    preflight_fn=preflight_only_reference,
) -> int:
    if (
        out_dense_cells is None
        or out_required_workspace_cells is None
        or out_required_block_tensor_cells is None
        or out_required_logits_cells is None
    ):
        return MODEL_Q16_ERR_NULL_PTR

    if (
        out_dense_cells is out_required_workspace_cells
        or out_dense_cells is out_required_block_tensor_cells
        or out_dense_cells is out_required_logits_cells
        or out_required_workspace_cells is out_required_block_tensor_cells
        or out_required_workspace_cells is out_required_logits_cells
        or out_required_block_tensor_cells is out_required_logits_cells
    ):
        return MODEL_Q16_ERR_BAD_PARAM

    snapshot_embed_capacity = embed_capacity
    snapshot_block_attn_capacity = block_attn_capacity
    snapshot_block_ffn_capacity = block_ffn_capacity
    snapshot_block_count = block_count
    snapshot_gamma_attn_capacity = gamma_attn_capacity
    snapshot_gamma_ffn_capacity = gamma_ffn_capacity
    snapshot_final_norm_gamma_capacity = final_norm_gamma_capacity
    snapshot_workspace_capacity = workspace_capacity
    snapshot_logits_capacity = logits_capacity
    snapshot_row_count = row_count
    snapshot_lane_count = lane_count

    snapshot_out_dense_cells = out_dense_cells
    snapshot_out_required_workspace_cells = out_required_workspace_cells
    snapshot_out_required_block_tensor_cells = out_required_block_tensor_cells
    snapshot_out_required_logits_cells = out_required_logits_cells

    status, commit_diag = preflight_fn(
        embed_capacity=embed_capacity,
        block_attn_capacity=block_attn_capacity,
        block_ffn_capacity=block_ffn_capacity,
        block_count=block_count,
        gamma_attn_capacity=gamma_attn_capacity,
        gamma_ffn_capacity=gamma_ffn_capacity,
        final_norm_gamma_capacity=final_norm_gamma_capacity,
        workspace_capacity=workspace_capacity,
        logits_capacity=logits_capacity,
        row_count=row_count,
        lane_count=lane_count,
    )
    if status != MODEL_Q16_OK:
        return status
    assert commit_diag is not None

    if row_count == 0 or lane_count == 0:
        canonical_dense_cells = 0
        canonical_required_workspace_cells = 0
        canonical_required_block_tensor_cells = 0
        canonical_required_logits_cells = 0
    else:
        err, canonical_dense_cells = try_mul_i64_checked(row_count, lane_count)
        if err != MODEL_Q16_OK:
            return err

        err, canonical_required_workspace_cells = try_mul_i64_checked(canonical_dense_cells, 2)
        if err != MODEL_Q16_OK:
            return err

        err, canonical_required_block_tensor_cells = try_mul_i64_checked(block_count, canonical_dense_cells)
        if err != MODEL_Q16_OK:
            return err

        canonical_required_logits_cells = canonical_dense_cells

        if (
            gamma_attn_capacity < lane_count
            or gamma_ffn_capacity < lane_count
            or final_norm_gamma_capacity < lane_count
            or embed_capacity < canonical_dense_cells
            or block_attn_capacity < canonical_required_block_tensor_cells
            or block_ffn_capacity < canonical_required_block_tensor_cells
            or workspace_capacity < canonical_required_workspace_cells
            or logits_capacity < canonical_required_logits_cells
        ):
            return MODEL_Q16_ERR_BAD_PARAM

    if (
        snapshot_embed_capacity != embed_capacity
        or snapshot_block_attn_capacity != block_attn_capacity
        or snapshot_block_ffn_capacity != block_ffn_capacity
        or snapshot_block_count != block_count
        or snapshot_gamma_attn_capacity != gamma_attn_capacity
        or snapshot_gamma_ffn_capacity != gamma_ffn_capacity
        or snapshot_final_norm_gamma_capacity != final_norm_gamma_capacity
        or snapshot_workspace_capacity != workspace_capacity
        or snapshot_logits_capacity != logits_capacity
        or snapshot_row_count != row_count
        or snapshot_lane_count != lane_count
    ):
        return MODEL_Q16_ERR_BAD_PARAM

    if (
        snapshot_out_dense_cells is not out_dense_cells
        or snapshot_out_required_workspace_cells is not out_required_workspace_cells
        or snapshot_out_required_block_tensor_cells is not out_required_block_tensor_cells
        or snapshot_out_required_logits_cells is not out_required_logits_cells
    ):
        return MODEL_Q16_ERR_BAD_PARAM

    if commit_diag["dense_cells"] != canonical_dense_cells:
        return MODEL_Q16_ERR_BAD_PARAM
    if commit_diag["required_workspace_cells"] != canonical_required_workspace_cells:
        return MODEL_Q16_ERR_BAD_PARAM
    if commit_diag["required_block_tensor_cells"] != canonical_required_block_tensor_cells:
        return MODEL_Q16_ERR_BAD_PARAM
    if commit_diag["required_logits_cells"] != canonical_required_logits_cells:
        return MODEL_Q16_ERR_BAD_PARAM

    out_dense_cells[0] = commit_diag["dense_cells"]
    out_required_workspace_cells[0] = commit_diag["required_workspace_cells"]
    out_required_block_tensor_cells[0] = commit_diag["required_block_tensor_cells"]
    out_required_logits_cells[0] = commit_diag["required_logits_cells"]
    return MODEL_Q16_OK


def test_source_contains_iq_1054_symbol() -> None:
    source = Path("src/model/model.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert signature in source
    body = source[source.index(signature):]

    assert "status = InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "snapshot_out_dense_cells = out_dense_cells;" in body
    assert "if (snapshot_out_dense_cells != out_dense_cells ||" in body
    assert "if (commit_required_logits_cells != canonical_required_logits_cells)" in body


def test_known_vector_success() -> None:
    out_dense = [111]
    out_workspace = [222]
    out_block = [333]
    out_logits = [444]

    status = inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_reference(
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

    assert status == MODEL_Q16_OK
    assert out_dense == [32]
    assert out_workspace == [64]
    assert out_block == [128]
    assert out_logits == [32]


def test_no_partial_publish_on_mismatch() -> None:
    out_dense = [901]
    out_workspace = [902]
    out_block = [903]
    out_logits = [904]

    def _bad_preflight(**_kwargs: int) -> tuple[int, dict[str, int] | None]:
        return MODEL_Q16_OK, {
            "dense_cells": 33,
            "required_workspace_cells": 66,
            "required_block_tensor_cells": 132,
            "required_logits_cells": 32,
        }

    status = inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_reference(
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
        preflight_fn=_bad_preflight,
    )

    assert status == MODEL_Q16_ERR_BAD_PARAM
    assert out_dense == [901]
    assert out_workspace == [902]
    assert out_block == [903]
    assert out_logits == [904]


def test_randomized_geometry_capacity_overflow_vectors() -> None:
    rng = random.Random(20260422_1054)

    for _ in range(1200):
        row_count = rng.randint(0, 256)
        lane_count = rng.randint(0, 256)
        block_count = rng.randint(0, 64)

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

        slack = rng.randint(0, 8)

        status = inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_reference(
            embed_capacity=dense + slack,
            block_attn_capacity=block_req + slack,
            block_ffn_capacity=block_req + slack,
            block_count=block_count,
            gamma_attn_capacity=lane_count + rng.randint(0, 3),
            gamma_ffn_capacity=lane_count + rng.randint(0, 3),
            final_norm_gamma_capacity=lane_count + rng.randint(0, 3),
            workspace_capacity=workspace_req + slack,
            logits_capacity=logits_req + slack,
            row_count=row_count,
            lane_count=lane_count,
            out_dense_cells=[-1],
            out_required_workspace_cells=[-1],
            out_required_block_tensor_cells=[-1],
            out_required_logits_cells=[-1],
        )

        assert status == MODEL_Q16_OK

    out_dense = [77]
    out_workspace = [88]
    out_block = [99]
    out_logits = [111]

    status = inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_reference(
        embed_capacity=I64_MAX,
        block_attn_capacity=I64_MAX,
        block_ffn_capacity=I64_MAX,
        block_count=2,
        gamma_attn_capacity=I64_MAX,
        gamma_ffn_capacity=I64_MAX,
        final_norm_gamma_capacity=I64_MAX,
        workspace_capacity=I64_MAX,
        logits_capacity=I64_MAX,
        row_count=1 << 62,
        lane_count=8,
        out_dense_cells=out_dense,
        out_required_workspace_cells=out_workspace,
        out_required_block_tensor_cells=out_block,
        out_required_logits_cells=out_logits,
    )

    assert status == MODEL_Q16_ERR_OVERFLOW
    assert out_dense == [77]
    assert out_workspace == [88]
    assert out_block == [99]
    assert out_logits == [111]


if __name__ == "__main__":
    test_source_contains_iq_1054_symbol()
    test_known_vector_success()
    test_no_partial_publish_on_mismatch()
    test_randomized_geometry_capacity_overflow_vectors()
    print("inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity=ok")
