"""Harness for RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnly (IQ-1279)."""

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


def test_preflight_only_signature_exists() -> None:
    source = _rope_source()
    assert (
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnly("
        in source
    )


def test_preflight_only_calls_nopartial_and_commit_only() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnly(",
    )

    assert "status_no_partial = RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartial(" in body
    assert (
        "status_commit_only = RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnly("
        in body
    )

    assert "if (status_no_partial != ROPE_Q16_OK)" in body
    assert "if (status_commit_only != ROPE_Q16_OK)" in body


def test_preflight_only_uses_snapshot_parity_guards() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnly(",
    )

    assert "snapshot_head_cell_capacity = head_cell_capacity;" in body
    assert "snapshot_head_base_index = head_base_index;" in body
    assert "snapshot_head_dim = head_dim;" in body
    assert "snapshot_pair_stride_cells = pair_stride_cells;" in body
    assert "snapshot_freq_base_q16 = freq_base_q16;" in body
    assert "snapshot_position = position;" in body

    assert body.count("if (head_cell_capacity != snapshot_head_cell_capacity)") >= 2
    assert body.count("if (head_base_index != snapshot_head_base_index)") >= 2
    assert body.count("if (head_dim != snapshot_head_dim)") >= 2
    assert body.count("if (pair_stride_cells != snapshot_pair_stride_cells)") >= 2
    assert body.count("if (freq_base_q16 != snapshot_freq_base_q16)") >= 2
    assert body.count("if (position != snapshot_position)") >= 2


def test_preflight_only_enforces_tuple_parity_before_publish() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnly(",
    )

    assert "pair_count = head_dim >> 1;" in body
    assert "if (commit_status_q16 != ROPE_Q16_OK)" in body
    assert "if (commit_pair_count_q16 != pair_count)" in body

    fail_index = body.find("if (commit_pair_count_q16 != pair_count)")
    publish_a = body.find("*out_pair_count = pair_count;")
    publish_b = body.find("*out_commit_status = commit_status_q16;")
    publish_c = body.find("*out_commit_pair_count = commit_pair_count_q16;")
    assert fail_index != -1
    assert publish_a > fail_index
    assert publish_b > fail_index
    assert publish_c > fail_index


def test_preflight_only_has_output_null_guards_and_head_dim_guard() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnly(",
    )

    assert "if (!head_cells_q16)" in body
    assert "if (!out_pair_count)" in body
    assert "if (!out_commit_status)" in body
    assert "if (!out_commit_pair_count)" in body
    assert "if (head_dim < 0)" in body
