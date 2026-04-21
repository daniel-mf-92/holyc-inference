#!/usr/bin/env python3
"""Parity harness for Q8_0DequantizeBlockQ16CheckedNoPartialArrayDefaultStride (IQ-943)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

import test_q8_0_dequantize_block_q16_checked_no_partial_array as core

Q8_0_OK = 0
Q8_0_ERR_BAD_DST_LEN = 2
Q8_0_VALUES_PER_BLOCK = 32
Q8_0_BLOCK_SIZE = Q8_0_VALUES_PER_BLOCK


def q8_0_dequantize_block_q16_checked_no_partial_array_default_stride(
    src_blocks,
    src_block_capacity: int,
    src_block_stride: int,
    block_count: int,
    dst_q16,
    dst_q16_capacity: int,
) -> int:
    out_stride_values = Q8_0_BLOCK_SIZE
    return core.q8_0_dequantize_block_q16_checked_no_partial_array(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        out_stride_values,
    )


def test_source_contains_iq943_default_stride_wrapper() -> None:
    source = Path("src/quant/q8_0.HC").read_text(encoding="utf-8")

    assert "#define Q8_0_BLOCK_SIZE" in source
    assert "Q8_0_VALUES_PER_BLOCK" in source

    sig = "I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayDefaultStride("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "out_stride_values = Q8_0_BLOCK_SIZE;" in body
    assert "return Q8_0DequantizeBlockQ16CheckedNoPartialArray(" in body


def test_default_stride_matches_explicit_checked_core() -> None:
    rng = random.Random(943)
    block_count = 5
    src_stride = 2
    src_capacity = 1 + (block_count - 1) * src_stride
    src_blocks = [core.make_q8_block(rng) for _ in range(src_capacity)]

    dst_capacity = block_count * Q8_0_VALUES_PER_BLOCK
    dst_default = [0] * dst_capacity
    dst_explicit = [0] * dst_capacity

    err_default = q8_0_dequantize_block_q16_checked_no_partial_array_default_stride(
        src_blocks,
        src_capacity,
        src_stride,
        block_count,
        dst_default,
        dst_capacity,
    )
    err_explicit = core.q8_0_dequantize_block_q16_checked_no_partial_array(
        src_blocks,
        src_capacity,
        src_stride,
        block_count,
        dst_explicit,
        dst_capacity,
        Q8_0_VALUES_PER_BLOCK,
    )

    assert err_default == Q8_0_OK
    assert err_explicit == Q8_0_OK
    assert dst_default == dst_explicit


def test_default_stride_inherits_alias_and_capacity_guards() -> None:
    rng = random.Random(20260421)
    src_blocks = [core.make_q8_block(rng) for _ in range(6)]

    # Alias is forbidden and must be rejected by checked delegate.
    err_alias = q8_0_dequantize_block_q16_checked_no_partial_array_default_stride(
        src_blocks,
        len(src_blocks),
        1,
        2,
        src_blocks,
        len(src_blocks),
    )
    assert err_alias == Q8_0_ERR_BAD_DST_LEN

    # Insufficient destination capacity fails with strict no-partial semantics.
    dst = [5555] * 63
    before = dst[:]
    err_short = q8_0_dequantize_block_q16_checked_no_partial_array_default_stride(
        src_blocks,
        len(src_blocks),
        1,
        2,
        dst,
        len(dst),
    )
    assert err_short == Q8_0_ERR_BAD_DST_LEN
    assert dst == before


def test_default_stride_adversarial_stride_vectors() -> None:
    rng = random.Random(2026943)

    for block_count, src_stride in [
        (0, 1),
        (1, 1),
        (2, 3),
        (4, 5),
    ]:
        src_capacity = 0 if block_count == 0 else 1 + (block_count - 1) * src_stride
        src_blocks = [core.make_q8_block(rng) for _ in range(max(src_capacity, 1))]

        dst_capacity = 0 if block_count == 0 else block_count * Q8_0_VALUES_PER_BLOCK
        dst = [0] * max(dst_capacity, 1)

        err = q8_0_dequantize_block_q16_checked_no_partial_array_default_stride(
            src_blocks,
            src_capacity,
            src_stride,
            block_count,
            dst,
            dst_capacity,
        )

        assert err == Q8_0_OK
