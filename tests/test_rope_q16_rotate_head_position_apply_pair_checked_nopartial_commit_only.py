"""Harness for RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnly (IQ-1278)."""

from __future__ import annotations

from pathlib import Path


def _extract_function(source: str, signature: str) -> str:
    start = source.find(signature)
    assert start != -1, f"missing signature: {signature}"
    brace = source.find("{", start)
    assert brace != -1, "missing opening brace"

    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError("unterminated function body")


def _rope_source() -> str:
    return Path("src/model/rope.HC").read_text(encoding="utf-8")


def test_commit_only_signature_exists() -> None:
    source = _rope_source()
    assert (
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnly(" in source
    )


def test_commit_only_calls_nopartial_and_uses_snapshots() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnly(",
    )

    assert "if (!head_cells_q16)" in body
    assert "if (!out_commit_pair_count)" in body
    assert "if (!out_commit_status)" in body

    assert "snapshot_head_cell_capacity = head_cell_capacity;" in body
    assert "snapshot_head_base_index = head_base_index;" in body
    assert "snapshot_head_dim = head_dim;" in body
    assert "snapshot_pair_stride_cells = pair_stride_cells;" in body
    assert "snapshot_freq_base_q16 = freq_base_q16;" in body
    assert "snapshot_position = position;" in body

    assert "status = RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartial(" in body
    assert "if (head_cell_capacity != snapshot_head_cell_capacity)" in body
    assert "if (head_base_index != snapshot_head_base_index)" in body
    assert "if (head_dim != snapshot_head_dim)" in body
    assert "if (pair_stride_cells != snapshot_pair_stride_cells)" in body
    assert "if (freq_base_q16 != snapshot_freq_base_q16)" in body
    assert "if (position != snapshot_position)" in body

    assert "pair_count = head_dim >> 1;" in body
    assert "*out_commit_pair_count = pair_count;" in body
    assert "*out_commit_status = ROPE_Q16_OK;" in body


def test_commit_only_publishes_only_on_success() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnly(",
    )

    failure_marker = "if (status != ROPE_Q16_OK)"
    failure_index = body.find(failure_marker)
    assert failure_index != -1

    publish_pair_index = body.find("*out_commit_pair_count = pair_count;")
    publish_status_index = body.find("*out_commit_status = ROPE_Q16_OK;")
    assert publish_pair_index != -1
    assert publish_status_index != -1

    assert publish_pair_index > failure_index
    assert publish_status_index > failure_index


def test_commit_only_returns_overflow_for_pair_count_shift() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnly(",
    )

    assert "if (head_dim < 0)" in body
    assert "return ROPE_Q16_ERR_BAD_PARAM;" in body
