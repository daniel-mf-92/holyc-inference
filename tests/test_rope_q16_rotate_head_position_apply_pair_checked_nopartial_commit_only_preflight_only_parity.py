"""Harness for RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-1280)."""

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


def test_parity_signature_exists() -> None:
    source = _rope_source()
    assert (
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnlyParity("
        in source
    )


def test_parity_calls_preflight_and_commit_only_paths() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnlyParity(",
    )

    assert (
        "status = RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnly("
        in body
    )
    assert "status = RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnly(" in body
    assert "if (status != ROPE_Q16_OK)" in body


def test_parity_uses_immutable_snapshots() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnlyParity(",
    )

    assert "snapshot_head_cell_capacity = head_cell_capacity;" in body
    assert "snapshot_head_base_index = head_base_index;" in body
    assert "snapshot_head_dim = head_dim;" in body
    assert "snapshot_pair_stride_cells = pair_stride_cells;" in body
    assert "snapshot_freq_base_q16 = freq_base_q16;" in body
    assert "snapshot_position = position;" in body
    assert "snapshot_out_pair_count = out_pair_count;" in body
    assert "snapshot_out_commit_status = out_commit_status;" in body
    assert "snapshot_out_commit_pair_count = out_commit_pair_count;" in body

    assert body.count("if (head_cell_capacity != snapshot_head_cell_capacity)") >= 2
    assert body.count("if (head_base_index != snapshot_head_base_index)") >= 2
    assert body.count("if (head_dim != snapshot_head_dim)") >= 2
    assert body.count("if (pair_stride_cells != snapshot_pair_stride_cells)") >= 2
    assert body.count("if (freq_base_q16 != snapshot_freq_base_q16)") >= 2
    assert body.count("if (position != snapshot_position)") >= 2
    assert body.count("if (out_pair_count != snapshot_out_pair_count)") >= 2
    assert body.count("if (out_commit_status != snapshot_out_commit_status)") >= 2
    assert body.count("if (out_commit_pair_count != snapshot_out_commit_pair_count)") >= 2


def test_parity_enforces_tuple_match_before_publish() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnlyParity(",
    )

    assert "pair_count = head_dim >> 1;" in body
    assert "if (preflight_pair_count != pair_count)" in body
    assert "if (preflight_commit_status_q16 != ROPE_Q16_OK)" in body
    assert "if (preflight_commit_pair_count_q16 != pair_count)" in body
    assert "if (commit_only_status_q16 != ROPE_Q16_OK)" in body
    assert "if (commit_only_pair_count_q16 != pair_count)" in body
    assert "if (preflight_pair_count != commit_only_pair_count_q16)" in body
    assert "if (preflight_commit_status_q16 != commit_only_status_q16)" in body
    assert (
        "if (preflight_commit_pair_count_q16 != commit_only_pair_count_q16)" in body
    )

    fail_index = body.find("if (preflight_commit_pair_count_q16 != commit_only_pair_count_q16)")
    publish_a = body.find("*out_pair_count = preflight_pair_count;")
    publish_b = body.find("*out_commit_status = preflight_commit_status_q16;")
    publish_c = body.find("*out_commit_pair_count = preflight_commit_pair_count_q16;")
    assert fail_index != -1
    assert publish_a > fail_index
    assert publish_b > fail_index
    assert publish_c > fail_index


def test_parity_null_and_domain_guards() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartialCommitOnlyPreflightOnlyParity(",
    )

    assert "if (!head_cells_q16)" in body
    assert "if (!out_pair_count)" in body
    assert "if (!out_commit_status)" in body
    assert "if (!out_commit_pair_count)" in body
    assert "if (head_dim < 0)" in body

