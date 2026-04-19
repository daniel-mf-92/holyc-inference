#!/usr/bin/env python3
"""Parity harness for RoPEQ16...NoPartialPreflightOnly diagnostics helper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_rotate_head_range_token_window_strided_preflight_only as preflight_ref


def rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial_preflight_only(
    head_cell_capacity: int,
    token_base_index: int,
    token_stride_q16: int,
    range_base_index: int,
    head_count: int,
    head_stride_cells: int,
    head_dim: int,
    pair_stride_cells: int,
    position_start: int,
    token_count: int,
    position_step: int,
    out_last_abs_range_base: list[int] | None,
    out_last_head_base_index: list[int] | None,
    out_last_head_last_pair_y_index: list[int] | None,
    out_last_token_position: list[int] | None,
) -> int:
    if (
        out_last_abs_range_base is None
        or out_last_head_base_index is None
        or out_last_head_last_pair_y_index is None
        or out_last_token_position is None
    ):
        return ref.ROPE_Q16_ERR_NULL_PTR

    err, diag = preflight_ref.rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only(
        head_cell_capacity,
        token_base_index,
        token_stride_q16,
        range_base_index,
        head_count,
        head_stride_cells,
        head_dim,
        pair_stride_cells,
        position_start,
        token_count,
        position_step,
    )
    if err != ref.ROPE_Q16_OK:
        return err

    out_last_abs_range_base[0] = diag[0]
    out_last_head_base_index[0] = diag[1]
    out_last_head_last_pair_y_index[0] = diag[2]
    out_last_token_position[0] = diag[3]
    return ref.ROPE_Q16_OK


def explicit_inline_checked_guards_reference(
    head_cell_capacity: int,
    token_base_index: int,
    token_stride_q16: int,
    range_base_index: int,
    head_count: int,
    head_stride_cells: int,
    head_dim: int,
    pair_stride_cells: int,
    position_start: int,
    token_count: int,
    position_step: int,
    out_last_abs_range_base: list[int] | None,
    out_last_head_base_index: list[int] | None,
    out_last_head_last_pair_y_index: list[int] | None,
    out_last_token_position: list[int] | None,
) -> int:
    return rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial_preflight_only(
        head_cell_capacity,
        token_base_index,
        token_stride_q16,
        range_base_index,
        head_count,
        head_stride_cells,
        head_dim,
        pair_stride_cells,
        position_start,
        token_count,
        position_step,
        out_last_abs_range_base,
        out_last_head_base_index,
        out_last_head_last_pair_y_index,
        out_last_token_position,
    )


def test_source_contains_signature_and_delegate_shape() -> None:
    source = Path("src/model/rope.HC").read_text(encoding="utf-8")
    signature = "I32 RoPEQ16RotateHeadRangeByTokenWindowCheckedStridedNoPartialPreflightOnly("
    assert signature in source

    body = source.split(signature, 1)[1]
    assert "RoPEQ16RotateHeadRangeByTokenWindowCheckedStridedPreflightOnly(head_cell_capacity," in body
    assert "token_stride_q16," in body
    assert "out_last_token_position);" in body


def test_strict_no_partial_output_contract_on_error() -> None:
    out_a = [111]
    out_b = [222]
    out_c = [333]
    out_d = [444]

    err = rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial_preflight_only(
        1024,
        ref.I64_MAX,
        ref.I64_MAX,
        1,
        1,
        24,
        16,
        2,
        0,
        2,
        1,
        out_a,
        out_b,
        out_c,
        out_d,
    )
    assert err == ref.ROPE_Q16_ERR_OVERFLOW
    assert out_a == [111]
    assert out_b == [222]
    assert out_c == [333]
    assert out_d == [444]


def test_null_output_pointer_passthrough() -> None:
    err = rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial_preflight_only(
        4096,
        32,
        128,
        4,
        2,
        24,
        16,
        2,
        10,
        3,
        1,
        None,
        [0],
        [0],
        [0],
    )
    assert err == ref.ROPE_Q16_ERR_NULL_PTR


def test_randomized_parity_vs_inline_checked_guards_reference() -> None:
    rng = random.Random(20260419_533)

    for _ in range(7000):
        head_dim = rng.choice([8, 16, 24, 32])
        pair_stride = rng.randint(1, 6)
        head_count = rng.randint(0, 4)
        token_count = rng.randint(0, 8)
        token_stride = rng.randint(0, 320)
        head_stride = rng.randint(1, 96)
        position_start = rng.randint(0, 8192)
        position_step = rng.randint(0, 12)

        head_span = ((head_dim // 2) - 1) * pair_stride + 1

        if head_count == 0 or token_count == 0:
            cap = rng.randint(1, 12_000)
            token_base = rng.randint(0, 1000)
            range_base = rng.randint(0, 1000)
        else:
            token_span = (token_count - 1) * token_stride
            needed = token_span + (head_count - 1) * head_stride + head_span
            cap = rng.randint(max(needed + 1, 1), max(needed + 1024, 2))
            max_token_base = max(0, cap - needed - 1)
            token_base = rng.randint(0, max_token_base)
            max_range_base = max(0, cap - token_base - needed - 1)
            range_base = rng.randint(0, max_range_base)

        if rng.random() < 0.18:
            pick = rng.randint(0, 5)
            if pick == 0:
                token_stride = -1
            elif pick == 1:
                token_count = -1
            elif pick == 2:
                head_stride = 0
            elif pick == 3:
                position_step = -rng.randint(1, 6)
                position_start = rng.randint(0, 8)
                token_count = rng.randint(2, 8)
            elif pick == 4:
                cap = -1
            else:
                token_base = ref.I64_MAX
                token_stride = ref.I64_MAX

        out1 = [777]
        out2 = [888]
        out3 = [999]
        out4 = [111]

        exp1 = list(out1)
        exp2 = list(out2)
        exp3 = list(out3)
        exp4 = list(out4)

        got_err = rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial_preflight_only(
            cap,
            token_base,
            token_stride,
            range_base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            position_start,
            token_count,
            position_step,
            out1,
            out2,
            out3,
            out4,
        )
        exp_err = explicit_inline_checked_guards_reference(
            cap,
            token_base,
            token_stride,
            range_base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            position_start,
            token_count,
            position_step,
            exp1,
            exp2,
            exp3,
            exp4,
        )

        assert got_err == exp_err
        assert out1 == exp1
        assert out2 == exp2
        assert out3 == exp3
        assert out4 == exp4


def run() -> None:
    test_source_contains_signature_and_delegate_shape()
    test_strict_no_partial_output_contract_on_error()
    test_null_output_pointer_passthrough()
    test_randomized_parity_vs_inline_checked_guards_reference()
    print("rope_q16_rotate_head_range_token_window_checked_strided_no_partial_preflight_only=ok")


if __name__ == "__main__":
    run()

