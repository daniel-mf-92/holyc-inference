#!/usr/bin/env python3
"""Parity harness for InferenceForwardQ16...ParityCommitOnlyPreflightOnlyParity (IQ-1059)."""

from __future__ import annotations

from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity import (
    MODEL_Q16_ERR_BAD_PARAM,
    MODEL_Q16_ERR_NULL_PTR,
    MODEL_Q16_OK,
)
from test_inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only import (
    parity_commit_only_reference,
)
from test_inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only import (
    parity_commit_only_preflight_only_reference,
)


def parity_commit_only_preflight_only_parity_reference(
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
    preflight_only_fn=parity_commit_only_preflight_only_reference,
    commit_only_fn=parity_commit_only_reference,
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

    snapshot_row_count = row_count
    snapshot_lane_count = lane_count
    snapshot_block_count = block_count
    snapshot_workspace_capacity = workspace_capacity
    snapshot_logits_capacity = logits_capacity

    snapshot_out_dense_cells = out_dense_cells
    snapshot_out_required_workspace_cells = out_required_workspace_cells
    snapshot_out_required_block_tensor_cells = out_required_block_tensor_cells
    snapshot_out_required_logits_cells = out_required_logits_cells

    staged_preflight_dense = [0]
    staged_preflight_workspace = [0]
    staged_preflight_block = [0]
    staged_preflight_logits = [0]

    status = preflight_only_fn(
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
        out_dense_cells=staged_preflight_dense,
        out_required_workspace_cells=staged_preflight_workspace,
        out_required_block_tensor_cells=staged_preflight_block,
        out_required_logits_cells=staged_preflight_logits,
    )
    if status != MODEL_Q16_OK:
        return status

    staged_commit_dense = [0]
    staged_commit_workspace = [0]
    staged_commit_block = [0]
    staged_commit_logits = [0]

    status = commit_only_fn(
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
        out_dense_cells=staged_commit_dense,
        out_required_workspace_cells=staged_commit_workspace,
        out_required_block_tensor_cells=staged_commit_block,
        out_required_logits_cells=staged_commit_logits,
    )
    if status != MODEL_Q16_OK:
        return status

    if (
        snapshot_row_count != row_count
        or snapshot_lane_count != lane_count
        or snapshot_block_count != block_count
        or snapshot_workspace_capacity != workspace_capacity
        or snapshot_logits_capacity != logits_capacity
    ):
        return MODEL_Q16_ERR_BAD_PARAM

    if (
        snapshot_out_dense_cells is not out_dense_cells
        or snapshot_out_required_workspace_cells is not out_required_workspace_cells
        or snapshot_out_required_block_tensor_cells is not out_required_block_tensor_cells
        or snapshot_out_required_logits_cells is not out_required_logits_cells
    ):
        return MODEL_Q16_ERR_BAD_PARAM

    if staged_preflight_dense[0] != staged_commit_dense[0]:
        return MODEL_Q16_ERR_BAD_PARAM
    if staged_preflight_workspace[0] != staged_commit_workspace[0]:
        return MODEL_Q16_ERR_BAD_PARAM
    if staged_preflight_block[0] != staged_commit_block[0]:
        return MODEL_Q16_ERR_BAD_PARAM
    if staged_preflight_logits[0] != staged_commit_logits[0]:
        return MODEL_Q16_ERR_BAD_PARAM

    out_dense_cells[0] = staged_preflight_dense[0]
    out_required_workspace_cells[0] = staged_preflight_workspace[0]
    out_required_block_tensor_cells[0] = staged_preflight_block[0]
    out_required_logits_cells[0] = staged_preflight_logits[0]
    return MODEL_Q16_OK


def test_source_contains_iq_1059_symbol() -> None:
    source = Path("src/model/model.HC").read_text(encoding="utf-8")
    signature = (
        "I32 InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParity("
    )
    assert signature in source
    body = source[source.index(signature):]
    assert (
        "status =\n"
        "        InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
        in body
    )
    assert "status = InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "if (staged_preflight_required_logits_cells != staged_commit_required_logits_cells)" in body


def test_known_vector_success() -> None:
    out_dense = [31]
    out_workspace = [32]
    out_block = [33]
    out_logits = [34]

    status = parity_commit_only_preflight_only_parity_reference(
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
    out_dense = [301]
    out_workspace = [302]
    out_block = [303]
    out_logits = [304]

    def _bad_commit_only(**_kwargs: int) -> int:
        return MODEL_Q16_ERR_BAD_PARAM

    status = parity_commit_only_preflight_only_parity_reference(
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
        commit_only_fn=_bad_commit_only,
    )

    assert status == MODEL_Q16_ERR_BAD_PARAM
    assert out_dense == [301]
    assert out_workspace == [302]
    assert out_block == [303]
    assert out_logits == [304]


def test_randomized_geometry_capacity_overflow_vectors() -> None:
    rng = random.Random(20260422_1059)

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

        status = parity_commit_only_preflight_only_parity_reference(
            embed_capacity=dense,
            block_attn_capacity=block_req,
            block_ffn_capacity=block_req,
            block_count=block_count,
            gamma_attn_capacity=lane_count,
            gamma_ffn_capacity=lane_count,
            final_norm_gamma_capacity=lane_count,
            workspace_capacity=workspace_req,
            logits_capacity=logits_req,
            row_count=row_count,
            lane_count=lane_count,
            out_dense_cells=[-1],
            out_required_workspace_cells=[-1],
            out_required_block_tensor_cells=[-1],
            out_required_logits_cells=[-1],
        )
        assert status == MODEL_Q16_OK


if __name__ == "__main__":
    test_source_contains_iq_1059_symbol()
    test_known_vector_success()
    test_no_partial_publish_on_mismatch()
    test_randomized_geometry_capacity_overflow_vectors()
    print("inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity=ok")
