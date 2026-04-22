#!/usr/bin/env python3
"""Parity harness for InferenceForwardQ16...Parity...ParityCommitOnlyPreflightOnlyParity (IQ-1068)."""

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
from test_inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only import (
    parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference,
)
from test_inference_forward_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only import (
    parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_reference,
)


def parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
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
    preflight_only_fn=parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_reference,
    commit_only_fn=parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference,
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

    if (
        out_dense_cells is out_required_workspace_cells
        or out_dense_cells is out_required_block_tensor_cells
        or out_dense_cells is out_required_logits_cells
        or out_required_workspace_cells is out_required_block_tensor_cells
        or out_required_workspace_cells is out_required_logits_cells
        or out_required_block_tensor_cells is out_required_logits_cells
    ):
        return MODEL_Q16_ERR_BAD_PARAM

    if (
        staged_preflight_dense[0] < 0
        or staged_preflight_workspace[0] < 0
        or staged_preflight_block[0] < 0
        or staged_preflight_logits[0] < 0
        or staged_commit_dense[0] < 0
        or staged_commit_workspace[0] < 0
        or staged_commit_block[0] < 0
        or staged_commit_logits[0] < 0
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


def test_source_contains_iq_1068_symbol() -> None:
    source = Path("src/model/model.HC").read_text(encoding="utf-8")
    signature = (
        "I32 InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
    )
    assert signature in source
    assert source.count(signature) == 1
    body = source[source.index(signature):]
    assert (
        "status =\n"
        "        InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
        in body
    )
    assert (
        "status =\n"
        "        InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly("
        in body
    )
    assert "if (staged_preflight_required_logits_cells !=" in body


def test_known_vector_success() -> None:
    out_dense = [801]
    out_workspace = [802]
    out_block = [803]
    out_logits = [804]

    status = parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
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

    assert status == MODEL_Q16_OK
    assert out_dense == [24]
    assert out_workspace == [48]
    assert out_block == [96]
    assert out_logits == [24]


def test_no_partial_publish_on_commit_mismatch() -> None:
    out_dense = [901]
    out_workspace = [902]
    out_block = [903]
    out_logits = [904]

    def _bad_commit_only(**kwargs: int) -> int:
        out_dense_cells = kwargs["out_dense_cells"]
        out_required_workspace_cells = kwargs["out_required_workspace_cells"]
        out_required_block_tensor_cells = kwargs["out_required_block_tensor_cells"]
        out_required_logits_cells = kwargs["out_required_logits_cells"]
        out_dense_cells[0] = 24
        out_required_workspace_cells[0] = 48
        out_required_block_tensor_cells[0] = 96
        out_required_logits_cells[0] = 23
        return MODEL_Q16_OK

    status = parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
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
        commit_only_fn=_bad_commit_only,
    )

    assert status == MODEL_Q16_ERR_BAD_PARAM
    assert out_dense == [901]
    assert out_workspace == [902]
    assert out_block == [903]
    assert out_logits == [904]


def test_no_partial_publish_on_preflight_mismatch() -> None:
    out_dense = [1001]
    out_workspace = [1002]
    out_block = [1003]
    out_logits = [1004]

    def _bad_preflight_only(**kwargs: int) -> int:
        out_dense_cells = kwargs["out_dense_cells"]
        out_required_workspace_cells = kwargs["out_required_workspace_cells"]
        out_required_block_tensor_cells = kwargs["out_required_block_tensor_cells"]
        out_required_logits_cells = kwargs["out_required_logits_cells"]
        out_dense_cells[0] = 24
        out_required_workspace_cells[0] = 48
        out_required_block_tensor_cells[0] = 95
        out_required_logits_cells[0] = 24
        return MODEL_Q16_OK

    status = parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
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
        preflight_only_fn=_bad_preflight_only,
    )

    assert status == MODEL_Q16_ERR_BAD_PARAM
    assert out_dense == [1001]
    assert out_workspace == [1002]
    assert out_block == [1003]
    assert out_logits == [1004]


def test_randomized_geometry_capacity_overflow_vectors() -> None:
    rng = random.Random(20260422_1068)

    for _ in range(1400):
        row_count = rng.randint(0, 32)
        lane_count = rng.randint(0, 128)
        block_count = rng.randint(0, 32)

        dense_cells = row_count * lane_count
        block_cells = block_count * row_count * lane_count

        workspace_capacity = dense_cells + rng.randint(0, 16)
        logits_capacity = dense_cells + rng.randint(0, 16)

        out_dense = [0x1111]
        out_workspace = [0x2222]
        out_block = [0x3333]
        out_logits = [0x4444]

        status = parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
            embed_capacity=block_cells,
            block_attn_capacity=block_cells,
            block_ffn_capacity=block_cells,
            block_count=block_count,
            gamma_attn_capacity=lane_count,
            gamma_ffn_capacity=lane_count,
            final_norm_gamma_capacity=lane_count,
            workspace_capacity=workspace_capacity,
            logits_capacity=logits_capacity,
            row_count=row_count,
            lane_count=lane_count,
            out_dense_cells=out_dense,
            out_required_workspace_cells=out_workspace,
            out_required_block_tensor_cells=out_block,
            out_required_logits_cells=out_logits,
        )

        expected_dense = dense_cells
        expected_workspace = max(dense_cells, workspace_capacity)
        expected_block = block_cells
        expected_logits = max(dense_cells, logits_capacity)

        assert status == MODEL_Q16_OK
        assert out_dense == [expected_dense]
        assert out_workspace == [expected_workspace]
        assert out_block == [expected_block]
        assert out_logits == [expected_logits]


def test_null_and_alias_guards() -> None:
    shared = [0]

    status = parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
        embed_capacity=0,
        block_attn_capacity=0,
        block_ffn_capacity=0,
        block_count=0,
        gamma_attn_capacity=0,
        gamma_ffn_capacity=0,
        final_norm_gamma_capacity=0,
        workspace_capacity=0,
        logits_capacity=0,
        row_count=0,
        lane_count=0,
        out_dense_cells=None,
        out_required_workspace_cells=[0],
        out_required_block_tensor_cells=[0],
        out_required_logits_cells=[0],
    )
    assert status == MODEL_Q16_ERR_NULL_PTR

    status = parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
        embed_capacity=0,
        block_attn_capacity=0,
        block_ffn_capacity=0,
        block_count=0,
        gamma_attn_capacity=0,
        gamma_ffn_capacity=0,
        final_norm_gamma_capacity=0,
        workspace_capacity=0,
        logits_capacity=0,
        row_count=0,
        lane_count=0,
        out_dense_cells=shared,
        out_required_workspace_cells=shared,
        out_required_block_tensor_cells=[0],
        out_required_logits_cells=[0],
    )
    assert status == MODEL_Q16_ERR_BAD_PARAM


if __name__ == "__main__":
    test_source_contains_iq_1068_symbol()
    test_known_vector_success()
    test_no_partial_publish_on_commit_mismatch()
    test_no_partial_publish_on_preflight_mismatch()
    test_randomized_geometry_capacity_overflow_vectors()
    test_null_and_alias_guards()
    print("ok")
