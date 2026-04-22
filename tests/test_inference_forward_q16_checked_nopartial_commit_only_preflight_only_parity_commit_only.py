#!/usr/bin/env python3
"""Parity harness for InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly (IQ-1055)."""

from __future__ import annotations

from pathlib import Path
import random
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from test_inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity import (
    MODEL_Q16_ERR_BAD_PARAM,
    MODEL_Q16_ERR_NULL_PTR,
    MODEL_Q16_ERR_OVERFLOW,
    MODEL_Q16_OK,
    inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_reference,
)
from test_inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity import (
    preflight_only_reference,
)


def inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
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
    parity_fn=inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_reference,
    canonical_fn=preflight_only_reference,
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

    commit_dense_cells = [0]
    commit_required_workspace_cells = [0]
    commit_required_block_tensor_cells = [0]
    commit_required_logits_cells = [0]

    status = parity_fn(
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
        out_dense_cells=commit_dense_cells,
        out_required_workspace_cells=commit_required_workspace_cells,
        out_required_block_tensor_cells=commit_required_block_tensor_cells,
        out_required_logits_cells=commit_required_logits_cells,
    )
    if status != MODEL_Q16_OK:
        return status

    status, canonical_diag = canonical_fn(
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
    assert canonical_diag is not None

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

    if commit_dense_cells[0] != canonical_diag["dense_cells"]:
        return MODEL_Q16_ERR_BAD_PARAM
    if commit_required_workspace_cells[0] != canonical_diag["required_workspace_cells"]:
        return MODEL_Q16_ERR_BAD_PARAM
    if commit_required_block_tensor_cells[0] != canonical_diag["required_block_tensor_cells"]:
        return MODEL_Q16_ERR_BAD_PARAM
    if commit_required_logits_cells[0] != canonical_diag["required_logits_cells"]:
        return MODEL_Q16_ERR_BAD_PARAM

    out_dense_cells[0] = commit_dense_cells[0]
    out_required_workspace_cells[0] = commit_required_workspace_cells[0]
    out_required_block_tensor_cells[0] = commit_required_block_tensor_cells[0]
    out_required_logits_cells[0] = commit_required_logits_cells[0]
    return MODEL_Q16_OK


def test_source_contains_iq_1055_symbol() -> None:
    source = Path("src/model/model.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly("
    assert signature in source
    body = source[source.index(signature):]

    assert "status = InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "status = InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "snapshot_out_dense_cells = out_dense_cells;" in body
    assert "if (commit_required_logits_cells != canonical_required_logits_cells)" in body


def test_known_vector_success() -> None:
    out_dense = [1001]
    out_workspace = [1002]
    out_block = [1003]
    out_logits = [1004]

    status = inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
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


def test_no_partial_publish_on_canonical_mismatch() -> None:
    out_dense = [901]
    out_workspace = [902]
    out_block = [903]
    out_logits = [904]

    def _bad_canonical(**_kwargs: int) -> tuple[int, dict[str, int] | None]:
        return MODEL_Q16_OK, {
            "dense_cells": 31,
            "required_workspace_cells": 62,
            "required_block_tensor_cells": 124,
            "required_logits_cells": 31,
        }

    status = inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
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
        canonical_fn=_bad_canonical,
    )

    assert status == MODEL_Q16_ERR_BAD_PARAM
    assert out_dense == [901]
    assert out_workspace == [902]
    assert out_block == [903]
    assert out_logits == [904]


def test_randomized_geometry_capacity_overflow_vectors() -> None:
    rng = random.Random(20260422_1055)

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

        status = inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
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

    status = inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
        embed_capacity=(1 << 63) - 1,
        block_attn_capacity=(1 << 63) - 1,
        block_ffn_capacity=(1 << 63) - 1,
        block_count=2,
        gamma_attn_capacity=(1 << 63) - 1,
        gamma_ffn_capacity=(1 << 63) - 1,
        final_norm_gamma_capacity=(1 << 63) - 1,
        workspace_capacity=(1 << 63) - 1,
        logits_capacity=(1 << 63) - 1,
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
    test_source_contains_iq_1055_symbol()
    test_known_vector_success()
    test_no_partial_publish_on_canonical_mismatch()
    test_randomized_geometry_capacity_overflow_vectors()
    print("inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only=ok")
