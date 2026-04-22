#!/usr/bin/env python3
"""Commit-only harness for InferenceForwardQ16...Parity...ParityCommitOnly (IQ-1065)."""

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
from test_inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity import (
    parity_commit_only_preflight_only_parity_reference,
)
from test_inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity import (
    parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference,
)


def parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
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
    staged_fn=parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference,
    canonical_fn=parity_commit_only_preflight_only_parity_reference,
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

    staged_dense = [0]
    staged_workspace = [0]
    staged_block = [0]
    staged_logits = [0]

    status = staged_fn(
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
        out_dense_cells=staged_dense,
        out_required_workspace_cells=staged_workspace,
        out_required_block_tensor_cells=staged_block,
        out_required_logits_cells=staged_logits,
    )
    if status != MODEL_Q16_OK:
        return status

    canonical_dense = [0]
    canonical_workspace = [0]
    canonical_block = [0]
    canonical_logits = [0]

    status = canonical_fn(
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
        out_dense_cells=canonical_dense,
        out_required_workspace_cells=canonical_workspace,
        out_required_block_tensor_cells=canonical_block,
        out_required_logits_cells=canonical_logits,
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

    if (
        out_dense_cells is out_required_workspace_cells
        or out_dense_cells is out_required_block_tensor_cells
        or out_dense_cells is out_required_logits_cells
        or out_required_workspace_cells is out_required_block_tensor_cells
        or out_required_workspace_cells is out_required_logits_cells
        or out_required_block_tensor_cells is out_required_logits_cells
    ):
        return MODEL_Q16_ERR_BAD_PARAM

    if staged_dense[0] != canonical_dense[0]:
        return MODEL_Q16_ERR_BAD_PARAM
    if staged_workspace[0] != canonical_workspace[0]:
        return MODEL_Q16_ERR_BAD_PARAM
    if staged_block[0] != canonical_block[0]:
        return MODEL_Q16_ERR_BAD_PARAM
    if staged_logits[0] != canonical_logits[0]:
        return MODEL_Q16_ERR_BAD_PARAM

    out_dense_cells[0] = staged_dense[0]
    out_required_workspace_cells[0] = staged_workspace[0]
    out_required_block_tensor_cells[0] = staged_block[0]
    out_required_logits_cells[0] = staged_logits[0]
    return MODEL_Q16_OK


def test_source_contains_iq_1065_symbol() -> None:
    source = Path("src/model/model.HC").read_text(encoding="utf-8")
    signature = (
        "I32 InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly("
    )
    assert signature in source
    body = source[source.index(signature):]
    assert (
        "status =\n"
        "        InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
        in body
    )
    assert (
        "status =\n"
        "        InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
        in body
    )
    assert "if (staged_required_logits_cells != canonical_required_logits_cells)" in body


def test_known_vector_success() -> None:
    out_dense = [71]
    out_workspace = [72]
    out_block = [73]
    out_logits = [74]

    status = (
        parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            embed_capacity=96,
            block_attn_capacity=192,
            block_ffn_capacity=192,
            block_count=4,
            gamma_attn_capacity=8,
            gamma_ffn_capacity=8,
            final_norm_gamma_capacity=8,
            workspace_capacity=48,
            logits_capacity=24,
            row_count=3,
            lane_count=8,
            out_dense_cells=out_dense,
            out_required_workspace_cells=out_workspace,
            out_required_block_tensor_cells=out_block,
            out_required_logits_cells=out_logits,
        )
    )

    assert status == MODEL_Q16_OK
    assert out_dense == [24]
    assert out_workspace == [48]
    assert out_block == [96]
    assert out_logits == [24]


def test_no_partial_publish_on_staged_parity_failure() -> None:
    out_dense = [801]
    out_workspace = [802]
    out_block = [803]
    out_logits = [804]

    def _bad_staged(**kwargs: int) -> int:
        out_dense_cells = kwargs["out_dense_cells"]
        out_required_workspace_cells = kwargs["out_required_workspace_cells"]
        out_required_block_tensor_cells = kwargs["out_required_block_tensor_cells"]
        out_required_logits_cells = kwargs["out_required_logits_cells"]
        out_dense_cells[0] = 24
        out_required_workspace_cells[0] = 48
        out_required_block_tensor_cells[0] = 96
        out_required_logits_cells[0] = 99
        return MODEL_Q16_OK

    status = (
        parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            embed_capacity=96,
            block_attn_capacity=192,
            block_ffn_capacity=192,
            block_count=4,
            gamma_attn_capacity=8,
            gamma_ffn_capacity=8,
            final_norm_gamma_capacity=8,
            workspace_capacity=48,
            logits_capacity=24,
            row_count=3,
            lane_count=8,
            out_dense_cells=out_dense,
            out_required_workspace_cells=out_workspace,
            out_required_block_tensor_cells=out_block,
            out_required_logits_cells=out_logits,
            staged_fn=_bad_staged,
        )
    )

    assert status == MODEL_Q16_ERR_BAD_PARAM
    assert out_dense == [801]
    assert out_workspace == [802]
    assert out_block == [803]
    assert out_logits == [804]


def test_no_partial_publish_on_canonical_mismatch() -> None:
    out_dense = [901]
    out_workspace = [902]
    out_block = [903]
    out_logits = [904]

    def _bad_canonical(**kwargs: int) -> int:
        out_dense_cells = kwargs["out_dense_cells"]
        out_required_workspace_cells = kwargs["out_required_workspace_cells"]
        out_required_block_tensor_cells = kwargs["out_required_block_tensor_cells"]
        out_required_logits_cells = kwargs["out_required_logits_cells"]
        out_dense_cells[0] = 24
        out_required_workspace_cells[0] = 48
        out_required_block_tensor_cells[0] = 96
        out_required_logits_cells[0] = 31
        return MODEL_Q16_OK

    status = (
        parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            embed_capacity=96,
            block_attn_capacity=192,
            block_ffn_capacity=192,
            block_count=4,
            gamma_attn_capacity=8,
            gamma_ffn_capacity=8,
            final_norm_gamma_capacity=8,
            workspace_capacity=48,
            logits_capacity=24,
            row_count=3,
            lane_count=8,
            out_dense_cells=out_dense,
            out_required_workspace_cells=out_workspace,
            out_required_block_tensor_cells=out_block,
            out_required_logits_cells=out_logits,
            canonical_fn=_bad_canonical,
        )
    )

    assert status == MODEL_Q16_ERR_BAD_PARAM
    assert out_dense == [901]
    assert out_workspace == [902]
    assert out_block == [903]
    assert out_logits == [904]


def test_randomized_geometry_capacity_overflow_vectors() -> None:
    rng = random.Random(20260422_1065)

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

        status = (
            parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
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
        )
        assert status == MODEL_Q16_OK


if __name__ == "__main__":
    test_source_contains_iq_1065_symbol()
    test_known_vector_success()
    test_no_partial_publish_on_staged_parity_failure()
    test_no_partial_publish_on_canonical_mismatch()
    test_randomized_geometry_capacity_overflow_vectors()
    print("ok")
