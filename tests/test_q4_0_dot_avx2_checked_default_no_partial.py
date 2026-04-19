#!/usr/bin/env python3
"""Parity harness for Q4_0DotProductBlocksAVX2CheckedDefaultNoPartial."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_dot_avx2 import (
    Q4_0_DOT_AVX2_ERR_BAD_LEN,
    Q4_0_DOT_AVX2_ERR_NULL_PTR,
    Q4_0_DOT_AVX2_OK,
    _make_random_blocks,
    half_bits,
)
from test_q4_0_dot_avx2_checked_default import q4_0_dot_product_blocks_avx2_checked_default


def q4_0_dot_product_blocks_avx2_checked_default_no_partial(
    lhs_blocks,
    rhs_blocks,
    block_count: int,
    out_holder,
) -> int:
    if out_holder is None:
        return Q4_0_DOT_AVX2_ERR_NULL_PTR

    staged_out = {"value": 0}
    err = q4_0_dot_product_blocks_avx2_checked_default(lhs_blocks, rhs_blocks, block_count, staged_out)
    if err != Q4_0_DOT_AVX2_OK:
        return err

    out_holder["value"] = staged_out["value"]
    return Q4_0_DOT_AVX2_OK


def staged_composition_reference(lhs_blocks, rhs_blocks, block_count: int, out_holder) -> int:
    if out_holder is None:
        return Q4_0_DOT_AVX2_ERR_NULL_PTR

    staged_value = 0
    staged_out = {"value": staged_value}
    err = q4_0_dot_product_blocks_avx2_checked_default(lhs_blocks, rhs_blocks, block_count, staged_out)
    if err != Q4_0_DOT_AVX2_OK:
        return err

    out_holder["value"] = staged_out["value"]
    return Q4_0_DOT_AVX2_OK


def _make_saturation_blocks(scale: float, count: int, rng: random.Random):
    blocks = []
    scale_bits = half_bits(scale)
    for _ in range(count):
        packed = bytes(rng.randrange(0, 256) for _ in range(16))
        blocks.append((scale_bits, packed))
    return blocks


def test_success_and_error_parity_vs_staged_composition_randomized() -> None:
    rng = random.Random(202604190468)

    for _ in range(700):
        block_count = rng.randint(0, 32)
        lhs = _make_random_blocks(rng, block_count)
        rhs = _make_random_blocks(rng, block_count)

        out_no_partial = {"value": 10101}
        out_staged_ref = {"value": 20202}

        err_no_partial = q4_0_dot_product_blocks_avx2_checked_default_no_partial(
            lhs, rhs, block_count, out_no_partial
        )
        err_ref = staged_composition_reference(lhs, rhs, block_count, out_staged_ref)

        assert err_no_partial == err_ref
        if err_ref == Q4_0_DOT_AVX2_OK:
            assert out_no_partial["value"] == out_staged_ref["value"]
        else:
            assert out_no_partial["value"] == 10101


def test_no_partial_on_malformed_block_counts() -> None:
    good = _make_random_blocks(random.Random(44), 1)
    out = {"value": -555}

    err = q4_0_dot_product_blocks_avx2_checked_default_no_partial(good, good, -1, out)
    assert err == Q4_0_DOT_AVX2_ERR_BAD_LEN
    assert out["value"] == -555

    err = q4_0_dot_product_blocks_avx2_checked_default_no_partial(good, good, 2, out)
    assert err == Q4_0_DOT_AVX2_ERR_BAD_LEN
    assert out["value"] == -555


def test_saturation_scale_fixtures_and_truncated_payload() -> None:
    rng = random.Random(202604190469)
    scales = [65504.0, -65504.0, 0.0, 0.5, -0.5]

    for scale in scales:
        lhs = _make_saturation_blocks(scale, 3, rng)
        rhs = _make_saturation_blocks(-scale if scale else 1.0, 3, rng)

        out_no_partial = {"value": 333}
        out_staged_ref = {"value": 444}

        err_no_partial = q4_0_dot_product_blocks_avx2_checked_default_no_partial(
            lhs, rhs, 3, out_no_partial
        )
        err_ref = staged_composition_reference(lhs, rhs, 3, out_staged_ref)

        assert err_no_partial == err_ref
        if err_ref == Q4_0_DOT_AVX2_OK:
            assert out_no_partial["value"] == out_staged_ref["value"]
        else:
            assert out_no_partial["value"] == 333

    lhs_bad = [(half_bits(1.0), b"\x11" * 16)]
    rhs_bad = [(half_bits(1.0), b"\x22" * 15)]
    out = {"value": 9999}
    err = q4_0_dot_product_blocks_avx2_checked_default_no_partial(lhs_bad, rhs_bad, 1, out)
    assert err == Q4_0_DOT_AVX2_ERR_BAD_LEN
    assert out["value"] == 9999


def test_null_ptr_surface_parity() -> None:
    one = _make_random_blocks(random.Random(77), 1)
    out = {"value": 7}

    err = q4_0_dot_product_blocks_avx2_checked_default_no_partial(None, one, 0, out)
    assert err == Q4_0_DOT_AVX2_ERR_NULL_PTR
    assert out["value"] == 7

    err = q4_0_dot_product_blocks_avx2_checked_default_no_partial(one, None, 0, out)
    assert err == Q4_0_DOT_AVX2_ERR_NULL_PTR
    assert out["value"] == 7

    err = q4_0_dot_product_blocks_avx2_checked_default_no_partial(one, one, 0, None)
    assert err == Q4_0_DOT_AVX2_ERR_NULL_PTR


def test_source_contains_no_partial_wrapper() -> None:
    source = pathlib.Path("src/quant/q4_0_dot_avx2.HC").read_text(encoding="utf-8")
    assert "I32 Q4_0DotProductBlocksAVX2CheckedDefaultNoPartial(" in source
    assert "Q4_0DotProductBlocksAVX2CheckedDefault(lhs," in source


if __name__ == "__main__":
    test_success_and_error_parity_vs_staged_composition_randomized()
    test_no_partial_on_malformed_block_counts()
    test_saturation_scale_fixtures_and_truncated_payload()
    test_null_ptr_surface_parity()
    test_source_contains_no_partial_wrapper()
    print("q4_0_dot_avx2_checked_default_no_partial_parity=ok")
