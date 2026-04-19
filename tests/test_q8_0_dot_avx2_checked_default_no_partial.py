#!/usr/bin/env python3
"""Parity harness for Q8_0DotProductBlocksAVX2CheckedDefaultNoPartial."""

from __future__ import annotations

import pathlib
import random
import struct
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q8_0_dot_avx2 import (
    Q8_0_AVX2_ERR_BAD_LEN,
    Q8_0_AVX2_ERR_NULL_PTR,
    Q8_0_AVX2_OK,
    Q8_0_VALUES_PER_BLOCK,
    _make_random_block,
)
from test_q8_0_dot_avx2_checked_default import q8_0_dot_product_blocks_avx2_checked_default


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def _make_random_blocks(rng: random.Random, count: int):
    return [_make_random_block(rng) for _ in range(count)]


def _make_saturation_blocks(scale: float, count: int, rng: random.Random):
    return [
        {
            "d_fp16": half_bits(scale),
            "qs": [rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)],
        }
        for _ in range(count)
    ]


def q8_0_dot_product_blocks_avx2_checked_default_no_partial(
    lhs_blocks,
    rhs_blocks,
    block_count: int,
    out_holder,
) -> int:
    if out_holder is None:
        return Q8_0_AVX2_ERR_NULL_PTR

    staged_out = {"value": 0}
    err = q8_0_dot_product_blocks_avx2_checked_default(lhs_blocks, rhs_blocks, block_count, staged_out)
    if err != Q8_0_AVX2_OK:
        return err

    out_holder["value"] = staged_out["value"]
    return Q8_0_AVX2_OK


def staged_composition_reference(lhs_blocks, rhs_blocks, block_count: int, out_holder) -> int:
    if out_holder is None:
        return Q8_0_AVX2_ERR_NULL_PTR

    staged_value = 0
    staged_out = {"value": staged_value}
    err = q8_0_dot_product_blocks_avx2_checked_default(lhs_blocks, rhs_blocks, block_count, staged_out)
    if err != Q8_0_AVX2_OK:
        return err

    out_holder["value"] = staged_out["value"]
    return Q8_0_AVX2_OK


def test_success_and_error_parity_vs_staged_composition_randomized() -> None:
    rng = random.Random(202604190474)

    for _ in range(800):
        block_count = rng.randint(0, 36)
        lhs = _make_random_blocks(rng, block_count)
        rhs = _make_random_blocks(rng, block_count)

        out_no_partial = {"value": 10101}
        out_staged_ref = {"value": 20202}

        err_no_partial = q8_0_dot_product_blocks_avx2_checked_default_no_partial(
            lhs, rhs, block_count, out_no_partial
        )
        err_ref = staged_composition_reference(lhs, rhs, block_count, out_staged_ref)

        assert err_no_partial == err_ref
        if err_ref == Q8_0_AVX2_OK:
            assert out_no_partial["value"] == out_staged_ref["value"]
        else:
            assert out_no_partial["value"] == 10101


def test_no_partial_on_malformed_block_counts() -> None:
    good = _make_random_blocks(random.Random(44), 1)
    out = {"value": -555}

    err = q8_0_dot_product_blocks_avx2_checked_default_no_partial(good, good, -1, out)
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out["value"] == -555

    err = q8_0_dot_product_blocks_avx2_checked_default_no_partial(good, good, 2, out)
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out["value"] == -555


def test_saturation_scale_fixtures_and_truncated_payload() -> None:
    rng = random.Random(202604190475)
    scales = [65504.0, -65504.0, 0.0, 0.5, -0.5]

    for scale in scales:
        lhs = _make_saturation_blocks(scale, 3, rng)
        rhs = _make_saturation_blocks(-scale if scale else 1.0, 3, rng)

        out_no_partial = {"value": 333}
        out_staged_ref = {"value": 444}

        err_no_partial = q8_0_dot_product_blocks_avx2_checked_default_no_partial(
            lhs, rhs, 3, out_no_partial
        )
        err_ref = staged_composition_reference(lhs, rhs, 3, out_staged_ref)

        assert err_no_partial == err_ref
        if err_ref == Q8_0_AVX2_OK:
            assert out_no_partial["value"] == out_staged_ref["value"]
        else:
            assert out_no_partial["value"] == 333

    lhs_bad = [{"d_fp16": half_bits(1.0), "qs": [1] * 31}]
    rhs_bad = [{"d_fp16": half_bits(1.0), "qs": [2] * 32}]
    out = {"value": 9999}
    err = q8_0_dot_product_blocks_avx2_checked_default_no_partial(lhs_bad, rhs_bad, 1, out)
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out["value"] == 9999


def test_null_ptr_surface_parity() -> None:
    one = _make_random_blocks(random.Random(77), 1)
    out = {"value": 7}

    err = q8_0_dot_product_blocks_avx2_checked_default_no_partial(None, one, 0, out)
    assert err == Q8_0_AVX2_ERR_NULL_PTR
    assert out["value"] == 7

    err = q8_0_dot_product_blocks_avx2_checked_default_no_partial(one, None, 0, out)
    assert err == Q8_0_AVX2_ERR_NULL_PTR
    assert out["value"] == 7

    err = q8_0_dot_product_blocks_avx2_checked_default_no_partial(one, one, 0, None)
    assert err == Q8_0_AVX2_ERR_NULL_PTR


def test_source_contains_no_partial_wrapper() -> None:
    source = pathlib.Path("src/quant/q8_0_avx2.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0DotProductBlocksAVX2CheckedDefaultNoPartial(" in source
    assert "Q8_0DotProductBlocksAVX2CheckedDefault(lhs," in source


if __name__ == "__main__":
    test_success_and_error_parity_vs_staged_composition_randomized()
    test_no_partial_on_malformed_block_counts()
    test_saturation_scale_fixtures_and_truncated_payload()
    test_null_ptr_surface_parity()
    test_source_contains_no_partial_wrapper()
    print("q8_0_dot_avx2_checked_default_no_partial_parity=ok")
