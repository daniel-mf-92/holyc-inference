#!/usr/bin/env python3
"""Parity harness for KVCacheInitQ16CheckedNoAllocCommitOnlyPreflightOnly (IQ-1190)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_kv_cache_init_q16_checked_noalloc import (
    explicit_init_formula,
    kv_cache_q16_init_checked_noalloc,
)
from test_kv_cache_init_q16_checked_noalloc_commit_only import (
    kv_cache_q16_init_checked_noalloc_commit_only,
)
from test_kv_cache_q16_indexing_checked import (
    I64_MAX,
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_ERR_OVERFLOW,
    KV_Q16_OK,
)


# Mirrors HolyC preflight-only parity wrapper in src/model/kv_cache.HC.
def kv_cache_q16_init_checked_noalloc_commit_only_preflight_only(
    k_cache_q16: list[int] | None,
    k_cache_capacity: int,
    v_cache_q16: list[int] | None,
    v_cache_capacity: int,
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    out_used_tokens: list[int] | None,
    out_layer_span_cells: list[int] | None,
    out_total_cells: list[int] | None,
) -> int:
    if (
        k_cache_q16 is None
        or v_cache_q16 is None
        or out_used_tokens is None
        or out_layer_span_cells is None
        or out_total_cells is None
    ):
        return KV_Q16_ERR_NULL_PTR

    if k_cache_q16 is v_cache_q16:
        return KV_Q16_ERR_BAD_PARAM

    if (
        out_used_tokens is out_layer_span_cells
        or out_used_tokens is out_total_cells
        or out_layer_span_cells is out_total_cells
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        out_used_tokens is k_cache_q16
        or out_used_tokens is v_cache_q16
        or out_layer_span_cells is k_cache_q16
        or out_layer_span_cells is v_cache_q16
        or out_total_cells is k_cache_q16
        or out_total_cells is v_cache_q16
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        k_cache_capacity < 0
        or v_cache_capacity < 0
        or layer_count < 0
        or token_capacity < 0
        or kv_heads < 0
        or head_dim < 0
    ):
        return KV_Q16_ERR_BAD_PARAM

    snapshot = (
        k_cache_capacity,
        v_cache_capacity,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
    )

    staged_commit_used_tokens = [0x11]
    staged_commit_layer_span_cells = [0x22]
    staged_commit_total_cells = [0x33]

    err = kv_cache_q16_init_checked_noalloc_commit_only(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        staged_commit_used_tokens,
        staged_commit_layer_span_cells,
        staged_commit_total_cells,
    )
    if err != KV_Q16_OK:
        return err

    staged_canonical_used_tokens = [0x44]
    staged_canonical_layer_span_cells = [0x55]
    staged_canonical_total_cells = [0x66]

    err = kv_cache_q16_init_checked_noalloc(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        staged_canonical_used_tokens,
        staged_canonical_layer_span_cells,
        staged_canonical_total_cells,
    )
    if err != KV_Q16_OK:
        return err

    if snapshot != (
        k_cache_capacity,
        v_cache_capacity,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        staged_commit_used_tokens[0] != staged_canonical_used_tokens[0]
        or staged_commit_layer_span_cells[0] != staged_canonical_layer_span_cells[0]
        or staged_commit_total_cells[0] != staged_canonical_total_cells[0]
    ):
        return KV_Q16_ERR_BAD_PARAM

    if staged_commit_used_tokens[0] != 0:
        return KV_Q16_ERR_BAD_PARAM

    out_used_tokens[0] = staged_commit_used_tokens[0]
    out_layer_span_cells[0] = staged_commit_layer_span_cells[0]
    out_total_cells[0] = staged_commit_total_cells[0]
    return KV_Q16_OK


def test_source_contains_preflight_only_symbol() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheInitQ16CheckedNoAllocCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "KVCacheInitQ16CheckedNoAllocCommitOnly(" in body
    assert "KVCacheInitQ16CheckedNoAlloc(" in body
    assert "snapshot_k_cache_capacity" in body
    assert "staged_commit_used_tokens != staged_canonical_used_tokens" in body
    assert "if (staged_commit_used_tokens != 0)" in body
    assert "*out_used_tokens = staged_commit_used_tokens;" in body
    assert "*out_layer_span_cells = staged_commit_layer_span_cells;" in body
    assert "*out_total_cells = staged_commit_total_cells;" in body


def test_known_vector_success() -> None:
    layer_count = 5
    token_capacity = 7
    kv_heads = 6
    head_dim = 16

    expected_layer_span, expected_total = explicit_init_formula(
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
    )

    k_cache = [0] * expected_total
    v_cache = [0] * expected_total

    out_used_tokens = [101]
    out_layer_span = [202]
    out_total = [303]

    err = kv_cache_q16_init_checked_noalloc_commit_only_preflight_only(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        out_used_tokens,
        out_layer_span,
        out_total,
    )
    assert err == KV_Q16_OK
    assert out_used_tokens == [0]
    assert out_layer_span == [expected_layer_span]
    assert out_total == [expected_total]


def test_no_partial_write_on_failure() -> None:
    k_cache = [0] * 8
    v_cache = [0] * 8

    out_used_tokens = [9001]
    out_layer_span = [9002]
    out_total = [9003]

    err = kv_cache_q16_init_checked_noalloc_commit_only_preflight_only(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        2,
        8,
        8,
        8,
        out_used_tokens,
        out_layer_span,
        out_total,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert out_used_tokens == [9001]
    assert out_layer_span == [9002]
    assert out_total == [9003]


def test_null_alias_and_output_alias_contracts() -> None:
    k_cache = [1] * 64
    v_cache = [2] * 64

    assert (
        kv_cache_q16_init_checked_noalloc_commit_only_preflight_only(
            None,
            64,
            v_cache,
            64,
            1,
            1,
            1,
            1,
            [0],
            [0],
            [0],
        )
        == KV_Q16_ERR_NULL_PTR
    )

    assert (
        kv_cache_q16_init_checked_noalloc_commit_only_preflight_only(
            k_cache,
            64,
            k_cache,
            64,
            1,
            1,
            1,
            1,
            [0],
            [0],
            [0],
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    shared = [123]
    err = kv_cache_q16_init_checked_noalloc_commit_only_preflight_only(
        k_cache,
        64,
        v_cache,
        64,
        1,
        1,
        1,
        1,
        shared,
        shared,
        [0],
    )
    assert err == KV_Q16_ERR_BAD_PARAM


def test_overflow_and_random_vectors() -> None:
    out_used_tokens = [11]
    out_layer_span = [22]
    out_total = [33]

    err = kv_cache_q16_init_checked_noalloc_commit_only_preflight_only(
        [0] * 1,
        I64_MAX,
        [0] * 1,
        I64_MAX,
        I64_MAX,
        I64_MAX,
        2,
        2,
        out_used_tokens,
        out_layer_span,
        out_total,
    )
    assert err == KV_Q16_ERR_OVERFLOW
    assert out_used_tokens == [11]
    assert out_layer_span == [22]
    assert out_total == [33]

    rng = random.Random(0x1190)
    for _ in range(250):
        layer_count = rng.randint(0, 16)
        token_capacity = rng.randint(0, 32)
        kv_heads = rng.randint(0, 16)
        head_dim = rng.randint(0, 32)

        expect = [0]
        expect_layer_span = [0]
        expect_total = [0]

        total = layer_count * token_capacity * kv_heads * head_dim
        cap_pad = rng.randint(0, 8)
        cap = total + cap_pad

        k_cache = [0] * cap
        v_cache = [0] * cap

        err_ref = kv_cache_q16_init_checked_noalloc(
            k_cache,
            len(k_cache),
            v_cache,
            len(v_cache),
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            expect,
            expect_layer_span,
            expect_total,
        )

        got_used = [777]
        got_layer_span = [888]
        got_total = [999]

        err = kv_cache_q16_init_checked_noalloc_commit_only_preflight_only(
            k_cache,
            len(k_cache),
            v_cache,
            len(v_cache),
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            got_used,
            got_layer_span,
            got_total,
        )

        assert err == err_ref
        if err == KV_Q16_OK:
            assert got_used == expect
            assert got_layer_span == expect_layer_span
            assert got_total == expect_total
        else:
            assert got_used == [777]
            assert got_layer_span == [888]
            assert got_total == [999]


def run() -> None:
    test_source_contains_preflight_only_symbol()
    test_known_vector_success()
    test_no_partial_write_on_failure()
    test_null_alias_and_output_alias_contracts()
    test_overflow_and_random_vectors()
    print("kv_cache_init_q16_checked_noalloc_commit_only_preflight_only_reference_checks=ok")


if __name__ == "__main__":
    run()
