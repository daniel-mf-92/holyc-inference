#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartialCommitOnly (IQ-880)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_kv_cache_q16_indexing_checked import (
    I64_MAX,
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_ERR_OVERFLOW,
    KV_Q16_OK,
    try_mul_i64_checked,
)
from test_kv_cache_q16_read_write_token_roundtrip_checked_nopartial import (
    kv_cache_q16_read_write_token_roundtrip_checked_nopartial,
)
from test_kv_cache_q16_write_token_checked_nopartial_commit_only import (
    kv_cache_q16_write_token_checked_nopartial_commit_only,
)


def kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only(
    k_cache_q16: list[int] | None,
    k_cache_capacity: int,
    v_cache_q16: list[int] | None,
    v_cache_capacity: int,
    layer_idx: int,
    token_idx: int,
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    k_token_src_q16: list[int] | None,
    k_token_src_capacity: int,
    v_token_src_q16: list[int] | None,
    v_token_src_capacity: int,
    k_token_out_q16: list[int] | None,
    k_token_out_capacity: int,
    v_token_out_q16: list[int] | None,
    v_token_out_capacity: int,
    out_token_cells: list[int] | None,
    out_token_bytes: list[int] | None,
    out_k_base_index: list[int] | None,
    out_v_base_index: list[int] | None,
) -> int:
    if (
        out_token_cells is None
        or out_token_bytes is None
        or out_k_base_index is None
        or out_v_base_index is None
    ):
        return KV_Q16_ERR_NULL_PTR

    if (
        out_token_cells is out_token_bytes
        or out_token_cells is out_k_base_index
        or out_token_cells is out_v_base_index
        or out_token_bytes is out_k_base_index
        or out_token_bytes is out_v_base_index
        or out_k_base_index is out_v_base_index
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        out_token_cells is k_cache_q16
        or out_token_cells is v_cache_q16
        or out_token_cells is k_token_src_q16
        or out_token_cells is v_token_src_q16
        or out_token_cells is k_token_out_q16
        or out_token_cells is v_token_out_q16
        or out_token_bytes is k_cache_q16
        or out_token_bytes is v_cache_q16
        or out_token_bytes is k_token_src_q16
        or out_token_bytes is v_token_src_q16
        or out_token_bytes is k_token_out_q16
        or out_token_bytes is v_token_out_q16
        or out_k_base_index is k_cache_q16
        or out_k_base_index is v_cache_q16
        or out_k_base_index is k_token_src_q16
        or out_k_base_index is v_token_src_q16
        or out_k_base_index is k_token_out_q16
        or out_k_base_index is v_token_out_q16
        or out_v_base_index is k_cache_q16
        or out_v_base_index is v_cache_q16
        or out_v_base_index is k_token_src_q16
        or out_v_base_index is v_token_src_q16
        or out_v_base_index is k_token_out_q16
        or out_v_base_index is v_token_out_q16
    ):
        return KV_Q16_ERR_BAD_PARAM

    snapshot_layer_idx = layer_idx
    snapshot_token_idx = token_idx
    snapshot_layer_count = layer_count
    snapshot_token_capacity = token_capacity
    snapshot_kv_heads = kv_heads
    snapshot_head_dim = head_dim
    snapshot_k_src_capacity = k_token_src_capacity
    snapshot_v_src_capacity = v_token_src_capacity
    snapshot_k_out_capacity = k_token_out_capacity
    snapshot_v_out_capacity = v_token_out_capacity

    staged_token_cells = [0]
    staged_k_base_index = [0]
    staged_v_base_index = [0]

    err = kv_cache_q16_write_token_checked_nopartial_commit_only(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        staged_token_cells,
        staged_k_base_index,
        staged_v_base_index,
    )
    if err != KV_Q16_OK:
        return err

    err, staged_token_bytes = try_mul_i64_checked(staged_token_cells[0], 8)
    if err != KV_Q16_OK:
        return err

    if (
        snapshot_layer_idx != layer_idx
        or snapshot_token_idx != token_idx
        or snapshot_layer_count != layer_count
        or snapshot_token_capacity != token_capacity
        or snapshot_kv_heads != kv_heads
        or snapshot_head_dim != head_dim
        or snapshot_k_src_capacity != k_token_src_capacity
        or snapshot_v_src_capacity != v_token_src_capacity
        or snapshot_k_out_capacity != k_token_out_capacity
        or snapshot_v_out_capacity != v_token_out_capacity
    ):
        return KV_Q16_ERR_BAD_PARAM

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        k_token_out_q16,
        k_token_out_capacity,
        v_token_out_q16,
        v_token_out_capacity,
    )
    if err != KV_Q16_OK:
        return err

    out_token_cells[0] = staged_token_cells[0]
    out_token_bytes[0] = staged_token_bytes
    out_k_base_index[0] = staged_k_base_index[0]
    out_v_base_index[0] = staged_v_base_index[0]
    return KV_Q16_OK


def explicit_roundtrip_commit_only_composition(*args) -> int:
    return kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only(*args)


def test_source_contains_commit_only_roundtrip_wrapper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartialCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "KVCacheQ16WriteTokenCheckedNoPartialCommitOnly(" in body
    assert "KVTryMulI64Checked(staged_token_cells" in body
    assert "KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartial(" in body
    assert "*out_token_cells = staged_token_cells;" in body
    assert "*out_token_bytes = staged_token_bytes;" in body
    assert "*out_k_base_index = staged_k_base_index;" in body
    assert "*out_v_base_index = staged_v_base_index;" in body


def test_known_vector_success_and_diagnostics() -> None:
    layer_count = 3
    token_capacity = 5
    kv_heads = 2
    head_dim = 4
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [-1] * total_cells
    v_cache = [-2] * total_cells
    k_src = [100 + idx for idx in range(span)]
    v_src = [200 + idx for idx in range(span)]
    k_out = [-3] * span
    v_out = [-4] * span

    out_cells = [0]
    out_bytes = [0]
    out_k_base = [0]
    out_v_base = [0]

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=2,
        token_idx=1,
        layer_count=layer_count,
        token_capacity=token_capacity,
        kv_heads=kv_heads,
        head_dim=head_dim,
        k_token_src_q16=k_src,
        k_token_src_capacity=len(k_src),
        v_token_src_q16=v_src,
        v_token_src_capacity=len(v_src),
        k_token_out_q16=k_out,
        k_token_out_capacity=len(k_out),
        v_token_out_q16=v_out,
        v_token_out_capacity=len(v_out),
        out_token_cells=out_cells,
        out_token_bytes=out_bytes,
        out_k_base_index=out_k_base,
        out_v_base_index=out_v_base,
    )
    assert err == KV_Q16_OK

    expected_base = ((2 * token_capacity) + 1) * span
    assert out_cells == [span]
    assert out_bytes == [span * 8]
    assert out_k_base == [expected_base]
    assert out_v_base == [expected_base]
    assert k_out == k_src
    assert v_out == v_src
    assert k_cache[expected_base : expected_base + span] == k_src
    assert v_cache[expected_base : expected_base + span] == v_src


def test_null_alias_and_no_partial_failure() -> None:
    k_cache = [10] * 64
    v_cache = [20] * 64
    k_src = [1] * 8
    v_src = [2] * 8
    k_out = [3] * 8
    v_out = [4] * 8

    assert (
        kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only(
            k_cache,
            len(k_cache),
            v_cache,
            len(v_cache),
            0,
            0,
            1,
            4,
            2,
            4,
            k_src,
            len(k_src),
            v_src,
            len(v_src),
            k_out,
            len(k_out),
            v_out,
            len(v_out),
            None,
            [0],
            [0],
            [0],
        )
        == KV_Q16_ERR_NULL_PTR
    )

    shared = [0]
    assert (
        kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only(
            k_cache,
            len(k_cache),
            v_cache,
            len(v_cache),
            0,
            0,
            1,
            4,
            2,
            4,
            k_src,
            len(k_src),
            v_src,
            len(v_src),
            k_out,
            len(k_out),
            v_out,
            len(v_out),
            shared,
            shared,
            [0],
            [0],
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    out_cells = [123]
    out_bytes = [456]
    out_k_base = [789]
    out_v_base = [987]
    k_before = k_cache.copy()
    v_before = v_cache.copy()

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=0,
        token_idx=99,
        layer_count=1,
        token_capacity=4,
        kv_heads=2,
        head_dim=4,
        k_token_src_q16=k_src,
        k_token_src_capacity=len(k_src),
        v_token_src_q16=v_src,
        v_token_src_capacity=len(v_src),
        k_token_out_q16=k_out,
        k_token_out_capacity=len(k_out),
        v_token_out_q16=v_out,
        v_token_out_capacity=len(v_out),
        out_token_cells=out_cells,
        out_token_bytes=out_bytes,
        out_k_base_index=out_k_base,
        out_v_base_index=out_v_base,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert k_cache == k_before
    assert v_cache == v_before
    assert out_cells == [123]
    assert out_bytes == [456]
    assert out_k_base == [789]
    assert out_v_base == [987]


def test_adversarial_geometry_capacity_alias_overflow_vectors() -> None:
    vectors = [
        dict(lc=1, tc=1, h=1, d=1, li=0, ti=0),
        dict(lc=2, tc=3, h=2, d=4, li=1, ti=2),
        dict(lc=4, tc=5, h=3, d=2, li=3, ti=4),
        dict(lc=0, tc=0, h=1, d=1, li=0, ti=0),
        dict(lc=1, tc=1, h=I64_MAX, d=2, li=0, ti=0),
    ]

    for idx, cfg in enumerate(vectors):
        span = cfg["h"] * cfg["d"] if cfg["h"] >= 0 and cfg["d"] >= 0 else 0
        if span < 0 or span > 256:
            span = 8

        total_cells = max(0, cfg["lc"] * cfg["tc"] * span)
        total_cells = min(total_cells, 1024)

        k_cache = [1000 + i for i in range(total_cells)]
        v_cache = [2000 + i for i in range(total_cells)]
        k_src = [10 + i for i in range(span)] if span <= 256 else [10] * 8
        v_src = [20 + i for i in range(span)] if span <= 256 else [20] * 8
        k_out = [-1] * len(k_src)
        v_out = [-2] * len(v_src)

        out1 = [[11], [22], [33], [44]]
        out2 = [[11], [22], [33], [44]]

        err_1 = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only(
            k_cache,
            len(k_cache),
            v_cache,
            len(v_cache),
            cfg["li"],
            cfg["ti"],
            cfg["lc"],
            cfg["tc"],
            cfg["h"],
            cfg["d"],
            k_src,
            len(k_src),
            v_src,
            len(v_src),
            k_out,
            len(k_out),
            v_out,
            len(v_out),
            out1[0],
            out1[1],
            out1[2],
            out1[3],
        )

        err_2 = explicit_roundtrip_commit_only_composition(
            k_cache,
            len(k_cache),
            v_cache,
            len(v_cache),
            cfg["li"],
            cfg["ti"],
            cfg["lc"],
            cfg["tc"],
            cfg["h"],
            cfg["d"],
            k_src,
            len(k_src),
            v_src,
            len(v_src),
            k_out,
            len(k_out),
            v_out,
            len(v_out),
            out2[0],
            out2[1],
            out2[2],
            out2[3],
        )

        assert err_1 == err_2, f"vector={idx} cfg={cfg}"
        if err_1 == KV_Q16_OK:
            assert out1 == out2

    # Explicit alias rejection against source/output buffers.
    k_cache = [0] * 16
    v_cache = [0] * 16
    src = [1] * 4
    dst_k = [0] * 4
    dst_v = [0] * 4
    out_token_bytes_alias = src

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        0,
        0,
        1,
        4,
        1,
        4,
        src,
        len(src),
        src.copy(),
        len(src),
        dst_k,
        len(dst_k),
        dst_v,
        len(dst_v),
        [0],
        out_token_bytes_alias,
        [0],
        [0],
    )
    assert err == KV_Q16_ERR_BAD_PARAM


def test_randomized_parity_vectors() -> None:
    rng = random.Random(20260421_880)

    for _ in range(500):
        layer_count = rng.randint(1, 6)
        token_capacity = rng.randint(1, 8)
        kv_heads = rng.randint(1, 4)
        head_dim = rng.randint(1, 8)
        span = kv_heads * head_dim

        total_cells = layer_count * token_capacity * span
        k_cache = [rng.randint(-5000, 5000) for _ in range(total_cells)]
        v_cache = [rng.randint(-5000, 5000) for _ in range(total_cells)]

        layer_idx = rng.randint(0, layer_count - 1)
        token_idx = rng.randint(0, token_capacity - 1)

        k_src = [rng.randint(-2000, 2000) for _ in range(span)]
        v_src = [rng.randint(-2000, 2000) for _ in range(span)]
        k_out_a = [0x1111] * span
        v_out_a = [0x2222] * span
        k_out_b = [0x1111] * span
        v_out_b = [0x2222] * span

        out_a = [[0xA], [0xB], [0xC], [0xD]]
        out_b = [[0xA], [0xB], [0xC], [0xD]]

        err_a = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only(
            k_cache.copy(),
            total_cells,
            v_cache.copy(),
            total_cells,
            layer_idx,
            token_idx,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_src,
            len(k_src),
            v_src,
            len(v_src),
            k_out_a,
            len(k_out_a),
            v_out_a,
            len(v_out_a),
            out_a[0],
            out_a[1],
            out_a[2],
            out_a[3],
        )

        err_b = explicit_roundtrip_commit_only_composition(
            k_cache.copy(),
            total_cells,
            v_cache.copy(),
            total_cells,
            layer_idx,
            token_idx,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_src,
            len(k_src),
            v_src,
            len(v_src),
            k_out_b,
            len(k_out_b),
            v_out_b,
            len(v_out_b),
            out_b[0],
            out_b[1],
            out_b[2],
            out_b[3],
        )

        assert err_a == err_b
        assert out_a == out_b
        assert k_out_a == k_out_b
        assert v_out_a == v_out_b


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__]))
