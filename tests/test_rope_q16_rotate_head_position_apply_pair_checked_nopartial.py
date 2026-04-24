"""Harness for RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartial (IQ-1203)."""

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


def test_nopartial_signature_exists() -> None:
    source = _rope_source()
    assert "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartial(" in source


def test_nopartial_uses_two_phase_preflight_then_commit() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartial(",
    )

    assert "status = RoPEQ16ValidateHeadSpanForDimChecked(" in body
    assert "for (pair_index = 0; pair_index < pair_count; pair_index++)" in body
    assert "status = RoPEQ16AngleStepChecked(" in body
    assert "status = RoPEQ16AngleForPositionChecked(" in body

    commit_call = "return RoPEQ16RotateHeadByPositionApplyPairChecked("
    assert commit_call in body

    preflight_index = body.find("for (pair_index = 0; pair_index < pair_count; pair_index++)")
    commit_index = body.find(commit_call)
    assert preflight_index != -1
    assert commit_index != -1
    assert preflight_index < commit_index


def test_nopartial_immutable_input_tuple_snapshot_guards() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartial(",
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


def test_nopartial_enforces_basic_pointer_and_bounds_guards() -> None:
    source = _rope_source()
    body = _extract_function(
        source,
        "I32 RoPEQ16RotateHeadByPositionApplyPairCheckedNoPartial(",
    )

    assert "if (!head_cells_q16)" in body
    assert "if (lane_x_index < 0 || lane_y_index < 0)" in body
    assert "if (lane_y_index > last_pair_y_index)" in body

