#!/usr/bin/env python3
"""Reference checks for RoPEQ16ApplyPairChecked semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_rotate_pair as pair_ref

ROPE_Q16_TWO_PI = 411775
ROPE_Q16_LUT_BITS = 10
ROPE_Q16_LUT_SIZE = 1 << ROPE_Q16_LUT_BITS
ROPE_Q16_LUT_MASK = ROPE_Q16_LUT_SIZE - 1


def rope_try_add_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    out = lhs + rhs
    if out < ref.I64_MIN or out > ref.I64_MAX:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    return ref.ROPE_Q16_OK, out


def rope_try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    out = lhs * rhs
    if out < ref.I64_MIN or out > ref.I64_MAX:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    return ref.ROPE_Q16_OK, out


def c_div_trunc(a: int, b: int) -> int:
    assert b != 0
    q = abs(a) // abs(b)
    return -q if (a < 0) ^ (b < 0) else q


def c_mod(a: int, b: int) -> int:
    return a - (c_div_trunc(a, b) * b)


def rope_q16_div_by_positive_int_rounded_checked(num_q16: int, den: int) -> tuple[int, int]:
    if den <= 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0

    is_negative = num_q16 < 0
    abs_num = abs(num_q16)
    q = abs_num // den
    r = abs_num % den

    limit = (1 << 63) if is_negative else ref.I64_MAX
    if q > limit:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0

    if r >= ((den + 1) >> 1):
        if q == limit:
            return ref.ROPE_Q16_ERR_OVERFLOW, 0
        q += 1

    out = -q if is_negative else q
    if out < ref.I64_MIN or out > ref.I64_MAX:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    return ref.ROPE_Q16_OK, out


def build_lut() -> tuple[list[int], list[int]]:
    sin_lut = [0] * ROPE_Q16_LUT_SIZE
    cos_lut = [0] * ROPE_Q16_LUT_SIZE

    for index in range(ROPE_Q16_LUT_SIZE):
        err, scaled_q16 = rope_try_mul_i64_checked(index, ROPE_Q16_TWO_PI)
        assert err == ref.ROPE_Q16_OK

        err, angle_q16 = rope_q16_div_by_positive_int_rounded_checked(scaled_q16, ROPE_Q16_LUT_SIZE)
        assert err == ref.ROPE_Q16_OK

        err, sin_lut[index] = pair_ref.rope_q16_sin_approx_checked(angle_q16)
        assert err == ref.ROPE_Q16_OK

        err, cos_lut[index] = pair_ref.rope_q16_cos_approx_checked(angle_q16)
        assert err == ref.ROPE_Q16_OK

    return sin_lut, cos_lut


SIN_LUT_Q16, COS_LUT_Q16 = build_lut()


def rope_q16_lookup_sincos_checked(angle_q16: int) -> tuple[int, int, int]:
    reduced_q16 = c_mod(angle_q16, ROPE_Q16_TWO_PI)
    if reduced_q16 < 0:
        err, reduced_q16 = rope_try_add_i64_checked(reduced_q16, ROPE_Q16_TWO_PI)
        if err != ref.ROPE_Q16_OK:
            return err, 0, 0

    err, scaled_index_q16 = rope_try_mul_i64_checked(reduced_q16, ROPE_Q16_LUT_SIZE)
    if err != ref.ROPE_Q16_OK:
        return err, 0, 0

    err, lut_index = rope_q16_div_by_positive_int_rounded_checked(scaled_index_q16, ROPE_Q16_TWO_PI)
    if err != ref.ROPE_Q16_OK:
        return err, 0, 0

    if lut_index < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0, 0

    lut_index &= ROPE_Q16_LUT_MASK
    return ref.ROPE_Q16_OK, SIN_LUT_Q16[lut_index], COS_LUT_Q16[lut_index]


def rope_q16_apply_pair_checked(
    head_cells_q16: list[int],
    head_cell_capacity: int,
    head_base_index: int,
    pair_index: int,
    pair_stride_cells: int,
    angle_q16: int,
) -> tuple[int, list[int]]:
    if head_cell_capacity < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if head_base_index < 0 or pair_index < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if pair_stride_cells < 2:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    err, pair_offset = rope_try_mul_i64_checked(pair_index, pair_stride_cells)
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, x_index = rope_try_add_i64_checked(head_base_index, pair_offset)
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, y_index = rope_try_add_i64_checked(x_index, 1)
    if err != ref.ROPE_Q16_OK:
        return err, []

    if x_index < 0 or y_index < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if x_index >= head_cell_capacity or y_index >= head_cell_capacity:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    out = list(head_cells_q16)
    x_q16 = out[x_index]
    y_q16 = out[y_index]

    err, sin_q16, cos_q16 = rope_q16_lookup_sincos_checked(angle_q16)
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, x_cos = ref.rope_q16_mul_checked(x_q16, cos_q16)
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, y_sin = ref.rope_q16_mul_checked(y_q16, sin_q16)
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, x_sin = ref.rope_q16_mul_checked(x_q16, sin_q16)
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, y_cos = ref.rope_q16_mul_checked(y_q16, cos_q16)
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, x_rot_q16 = rope_try_add_i64_checked(x_cos, -y_sin)
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, y_rot_q16 = rope_try_add_i64_checked(x_sin, y_cos)
    if err != ref.ROPE_Q16_OK:
        return err, []

    out[x_index] = x_rot_q16
    out[y_index] = y_rot_q16
    return ref.ROPE_Q16_OK, out


def test_bad_param_contracts() -> None:
    buf = [ref.q16_from_float(0.5)] * 16
    angle_q16 = ref.q16_from_float(1.25)

    assert rope_q16_apply_pair_checked(buf, -1, 0, 0, 2, angle_q16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_apply_pair_checked(buf, 16, -1, 0, 2, angle_q16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_apply_pair_checked(buf, 16, 0, -1, 2, angle_q16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_apply_pair_checked(buf, 16, 0, 0, 1, angle_q16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_bounds_guard_checked_lane_indexing() -> None:
    buf = [ref.q16_from_float(0.1)] * 10
    angle_q16 = ref.q16_from_float(0.75)

    err, _ = rope_q16_apply_pair_checked(
        buf,
        10,
        7,
        1,
        2,
        angle_q16,
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM


def test_pair_rotation_matches_lut_reference() -> None:
    rng = random.Random(202604222201)

    for _ in range(2500):
        capacity = rng.randint(8, 128)
        pair_stride = rng.randint(2, 7)
        pair_index = rng.randint(0, 8)
        head_base_index = rng.randint(0, max(0, capacity - 1))
        angle_q16 = rng.randint(-8 * ROPE_Q16_TWO_PI, 8 * ROPE_Q16_TWO_PI)

        buf = [ref.q16_from_float(rng.uniform(-2.5, 2.5)) for _ in range(capacity)]

        err, got = rope_q16_apply_pair_checked(
            buf,
            capacity,
            head_base_index,
            pair_index,
            pair_stride,
            angle_q16,
        )

        err_mul, pair_offset = rope_try_mul_i64_checked(pair_index, pair_stride)
        if err_mul != ref.ROPE_Q16_OK:
            assert err == ref.ROPE_Q16_ERR_OVERFLOW
            continue

        err_add, x_index = rope_try_add_i64_checked(head_base_index, pair_offset)
        if err_add != ref.ROPE_Q16_OK:
            assert err == ref.ROPE_Q16_ERR_OVERFLOW
            continue

        err_add2, y_index = rope_try_add_i64_checked(x_index, 1)
        if err_add2 != ref.ROPE_Q16_OK:
            assert err == ref.ROPE_Q16_ERR_OVERFLOW
            continue

        if x_index < 0 or y_index < 0 or x_index >= capacity or y_index >= capacity:
            assert err == ref.ROPE_Q16_ERR_BAD_PARAM
            continue

        assert err == ref.ROPE_Q16_OK
        assert len(got) == len(buf)

        err_sincos, sin_q16, cos_q16 = rope_q16_lookup_sincos_checked(angle_q16)
        assert err_sincos == ref.ROPE_Q16_OK

        err_mul, x_cos = ref.rope_q16_mul_checked(buf[x_index], cos_q16)
        assert err_mul == ref.ROPE_Q16_OK
        err_mul, y_sin = ref.rope_q16_mul_checked(buf[y_index], sin_q16)
        assert err_mul == ref.ROPE_Q16_OK
        err_mul, x_sin = ref.rope_q16_mul_checked(buf[x_index], sin_q16)
        assert err_mul == ref.ROPE_Q16_OK
        err_mul, y_cos = ref.rope_q16_mul_checked(buf[y_index], cos_q16)
        assert err_mul == ref.ROPE_Q16_OK

        err_add, want_x = rope_try_add_i64_checked(x_cos, -y_sin)
        assert err_add == ref.ROPE_Q16_OK
        err_add, want_y = rope_try_add_i64_checked(x_sin, y_cos)
        assert err_add == ref.ROPE_Q16_OK

        assert got[x_index] == want_x
        assert got[y_index] == want_y
        for lane in range(capacity):
            if lane != x_index and lane != y_index:
                assert got[lane] == buf[lane]


def test_source_contains_apply_pair_and_lookup_path() -> None:
    source = Path("src/model/rope.HC").read_text(encoding="utf-8")
    assert "I32 RoPEQ16ApplyPairChecked(" in source
    assert "I32 RoPEQ16LookupSinCosChecked(" in source

    start = source.rindex("I32 RoPEQ16ApplyPairChecked(")
    tail = source[start:]
    next_fn = tail.find("\nI32 ", 1)
    body = tail if next_fn == -1 else tail[:next_fn]

    assert "status = RoPEQ16LookupSinCosChecked(angle_q16," in body
    assert "head_cells_q16[x_index] = x_rot_q16;" in body
    assert "head_cells_q16[y_index] = y_rot_q16;" in body


def run() -> None:
    test_bad_param_contracts()
    test_bounds_guard_checked_lane_indexing()
    test_pair_rotation_matches_lut_reference()
    test_source_contains_apply_pair_and_lookup_path()
    print("rope_q16_apply_pair_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
