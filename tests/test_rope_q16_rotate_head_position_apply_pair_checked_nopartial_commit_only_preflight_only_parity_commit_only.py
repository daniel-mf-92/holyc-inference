"""Harness for RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly (IQ-1286)."""

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


def test_parity_commit_only_signature_exists() -> None:
    source = _rope_source()
    assert (
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly("
        in source
    )


def test_parity_commit_only_calls_parity_path() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(",
    )

    assert (
        "status = RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnlyParity("
        in body
    )
    assert "if (status != ROPE_Q16_OK)" in body


def test_parity_commit_only_immutable_snapshots() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(",
    )

    assert "snapshot_head_cell_capacity = head_cell_capacity;" in body
    assert "snapshot_head_base_index = head_base_index;" in body
    assert "snapshot_head_dim = head_dim;" in body
    assert "snapshot_pair_stride_cells = pair_stride_cells;" in body
    assert "snapshot_freq_base_q16 = freq_base_q16;" in body
    assert "snapshot_position = position;" in body

    assert "if (head_cell_capacity != snapshot_head_cell_capacity)" in body
    assert "if (head_base_index != snapshot_head_base_index)" in body
    assert "if (head_dim != snapshot_head_dim)" in body
    assert "if (pair_stride_cells != snapshot_pair_stride_cells)" in body
    assert "if (freq_base_q16 != snapshot_freq_base_q16)" in body
    assert "if (position != snapshot_position)" in body


def test_parity_commit_only_enforces_tuple_and_atomic_publish() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(",
    )

    assert "pair_count = head_dim >> 1;" in body
    assert "if (staged_pair_count != pair_count)" in body
    assert "if (staged_commit_status != ROPE_Q16_OK)" in body
    assert "if (staged_commit_pair_count != pair_count)" in body

    fail_index = body.find("if (staged_commit_pair_count != pair_count)")
    publish_pair = body.find("*out_commit_pair_count = staged_commit_pair_count;")
    publish_status = body.find("*out_commit_status = staged_commit_status;")
    assert fail_index != -1
    assert publish_pair > fail_index
    assert publish_status > fail_index


def test_parity_commit_only_null_and_domain_guards() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(",
    )

    assert "if (!head_cells_q16)" in body
    assert "if (!out_commit_pair_count)" in body
    assert "if (!out_commit_status)" in body
    assert "if (head_dim < 0)" in body
