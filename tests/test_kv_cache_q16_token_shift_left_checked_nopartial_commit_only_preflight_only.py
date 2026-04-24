#!/usr/bin/env python3
"""Parity harness for KVCacheQ16TokenShiftLeftCheckedNoPartialCommitOnlyPreflightOnly (IQ-1283)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_kv_cache_q16_indexing_checked import (
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_OK,
)
from test_kv_cache_q16_token_shift_left_checked_nopartial import _sim_shift
from test_kv_cache_q16_token_shift_left_checked_nopartial_commit_only import (
    kv_cache_q16_token_shift_left_checked_nopartial_commit_only,
)

SRC = Path(__file__).resolve().parents[1] / "src/model/kv_cache.HC"


def _extract_fn(name: str) -> str:
    text = SRC.read_text()
    m = re.search(rf"I32\s+{name}\s*\([^)]*\)\s*\{{", text)
    assert m, f"missing {name}"
    i = m.end() - 1
    depth = 0
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[m.start() : i + 1]
        i += 1
    raise AssertionError("unbalanced braces")


def kv_cache_q16_token_shift_left_checked_nopartial_commit_only_preflight_only(
    k_cache: list[int] | None,
    v_cache: list[int] | None,
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    used_tokens: int,
    shift_tokens: int,
    out_new_used_tokens: list[int] | None,
    out_moved_cells: list[int] | None,
) -> int:
    if k_cache is None or v_cache is None or out_new_used_tokens is None or out_moved_cells is None:
        return KV_Q16_ERR_NULL_PTR
    if k_cache is v_cache:
        return KV_Q16_ERR_BAD_PARAM
    if out_new_used_tokens is out_moved_cells:
        return KV_Q16_ERR_BAD_PARAM
    if (
        out_new_used_tokens is k_cache
        or out_new_used_tokens is v_cache
        or out_moved_cells is k_cache
        or out_moved_cells is v_cache
    ):
        return KV_Q16_ERR_BAD_PARAM
    if (
        layer_count < 0
        or token_capacity < 0
        or kv_heads < 0
        or head_dim < 0
        or used_tokens < 0
        or shift_tokens < 0
    ):
        return KV_Q16_ERR_BAD_PARAM
    if used_tokens > token_capacity:
        return KV_Q16_ERR_BAD_PARAM

    snapshot_used_tokens = used_tokens
    snapshot_shift_tokens = shift_tokens
    snapshot_layer_count = layer_count
    snapshot_token_capacity = token_capacity
    snapshot_kv_heads = kv_heads
    snapshot_head_dim = head_dim
    snapshot_k_cache = k_cache
    snapshot_v_cache = v_cache

    total_cells = layer_count * token_capacity * kv_heads * head_dim
    staged_commit_k = k_cache[:total_cells]
    staged_commit_v = v_cache[:total_cells]
    staged_canonical_k = k_cache[:total_cells]
    staged_canonical_v = v_cache[:total_cells]

    staged_commit_new_used = [0]
    staged_commit_moved = [0]
    status = kv_cache_q16_token_shift_left_checked_nopartial_commit_only(
        staged_commit_k,
        staged_commit_v,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        shift_tokens,
        staged_commit_new_used,
        staged_commit_moved,
    )
    if status != KV_Q16_OK:
        return status

    sim = _sim_shift(
        staged_canonical_k,
        staged_canonical_v,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        shift_tokens,
    )
    if sim == "bad":
        return KV_Q16_ERR_BAD_PARAM
    staged_canonical_k, staged_canonical_v, staged_canonical_new_used, staged_canonical_moved = sim

    if (
        snapshot_used_tokens != used_tokens
        or snapshot_shift_tokens != shift_tokens
        or snapshot_layer_count != layer_count
        or snapshot_token_capacity != token_capacity
        or snapshot_kv_heads != kv_heads
        or snapshot_head_dim != head_dim
        or snapshot_k_cache is not k_cache
        or snapshot_v_cache is not v_cache
    ):
        return KV_Q16_ERR_BAD_PARAM

    if staged_commit_new_used[0] != staged_canonical_new_used:
        return KV_Q16_ERR_BAD_PARAM
    if staged_commit_moved[0] != staged_canonical_moved:
        return KV_Q16_ERR_BAD_PARAM
    if staged_commit_k != staged_canonical_k:
        return KV_Q16_ERR_BAD_PARAM
    if staged_commit_v != staged_canonical_v:
        return KV_Q16_ERR_BAD_PARAM

    out_new_used_tokens[0] = staged_commit_new_used[0]
    out_moved_cells[0] = staged_commit_moved[0]
    return KV_Q16_OK


def explicit_preflight_composition(
    k_cache: list[int] | None,
    v_cache: list[int] | None,
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    used_tokens: int,
    shift_tokens: int,
    out_new_used_tokens: list[int] | None,
    out_moved_cells: list[int] | None,
) -> int:
    if k_cache is None or v_cache is None or out_new_used_tokens is None or out_moved_cells is None:
        return KV_Q16_ERR_NULL_PTR
    if k_cache is v_cache:
        return KV_Q16_ERR_BAD_PARAM
    if out_new_used_tokens is out_moved_cells:
        return KV_Q16_ERR_BAD_PARAM

    staged_commit_k = k_cache[:]
    staged_commit_v = v_cache[:]
    staged_canonical_k = k_cache[:]
    staged_canonical_v = v_cache[:]

    commit_new = [0]
    commit_moved = [0]
    status = kv_cache_q16_token_shift_left_checked_nopartial_commit_only(
        staged_commit_k,
        staged_commit_v,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        shift_tokens,
        commit_new,
        commit_moved,
    )
    if status != KV_Q16_OK:
        return status

    sim = _sim_shift(
        staged_canonical_k,
        staged_canonical_v,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        shift_tokens,
    )
    if sim == "bad":
        return KV_Q16_ERR_BAD_PARAM
    staged_canonical_k, staged_canonical_v, canonical_new, canonical_moved = sim

    if commit_new[0] != canonical_new or commit_moved[0] != canonical_moved:
        return KV_Q16_ERR_BAD_PARAM
    if staged_commit_k != staged_canonical_k or staged_commit_v != staged_canonical_v:
        return KV_Q16_ERR_BAD_PARAM

    out_new_used_tokens[0] = commit_new[0]
    out_moved_cells[0] = commit_moved[0]
    return KV_Q16_OK


def test_function_present_and_key_guards():
    fn = _extract_fn("KVCacheQ16TokenShiftLeftCheckedNoPartialCommitOnlyPreflightOnly")
    for needle in [
        "if (k_cache == v_cache)",
        "if (out_new_used_tokens == out_moved_cells)",
        "status = KVCacheQ16TokenShiftLeftCheckedNoPartialCommitOnly(",
        "status = KVCacheQ16TokenShiftLeftCheckedNoPartial(",
        "if (staged_commit_new_used_tokens != staged_canonical_new_used_tokens",
        "while (idx < total_cells)",
        "*out_new_used_tokens = staged_commit_new_used_tokens",
        "*out_moved_cells = staged_commit_moved_cells",
    ]:
        assert needle in fn


def test_preflight_only_is_zero_write_and_publishes_expected_tuple():
    layer_count, token_capacity, kv_heads, head_dim = 2, 5, 2, 3
    total_cells = layer_count * token_capacity * kv_heads * head_dim
    k = list(range(1, total_cells + 1))
    v = list(range(1001, 1001 + total_cells))
    k_before = k[:]
    v_before = v[:]

    out_new = [77]
    out_moved = [88]
    err = kv_cache_q16_token_shift_left_checked_nopartial_commit_only_preflight_only(
        k, v, layer_count, token_capacity, kv_heads, head_dim, 4, 1, out_new, out_moved
    )

    assert err == KV_Q16_OK
    assert out_new == [3]
    assert out_moved == [3 * layer_count * kv_heads * head_dim]
    assert k == k_before
    assert v == v_before


def test_preflight_only_matches_explicit_composition():
    layer_count, token_capacity, kv_heads, head_dim = 1, 4, 2, 2
    total_cells = layer_count * token_capacity * kv_heads * head_dim
    k0 = list(range(1, total_cells + 1))
    v0 = list(range(500, 500 + total_cells))

    k_a, v_a = k0[:], v0[:]
    k_b, v_b = k0[:], v0[:]
    out_new_a = [0]
    out_moved_a = [0]
    out_new_b = [0]
    out_moved_b = [0]

    err_a = kv_cache_q16_token_shift_left_checked_nopartial_commit_only_preflight_only(
        k_a, v_a, layer_count, token_capacity, kv_heads, head_dim, 3, 2, out_new_a, out_moved_a
    )
    err_b = explicit_preflight_composition(
        k_b, v_b, layer_count, token_capacity, kv_heads, head_dim, 3, 2, out_new_b, out_moved_b
    )

    assert err_a == KV_Q16_OK
    assert err_b == KV_Q16_OK
    assert out_new_a == out_new_b
    assert out_moved_a == out_moved_b
    assert k_a == k0
    assert v_a == v0
    assert k_b == k0
    assert v_b == v0


def test_zero_total_cells_path_still_reports_parity_tuple():
    k = [9, 8, 7]
    v = [6, 5, 4]
    out_new = [123]
    out_moved = [456]

    err = kv_cache_q16_token_shift_left_checked_nopartial_commit_only_preflight_only(
        k,
        v,
        layer_count=0,
        token_capacity=4,
        kv_heads=2,
        head_dim=3,
        used_tokens=3,
        shift_tokens=1,
        out_new_used_tokens=out_new,
        out_moved_cells=out_moved,
    )

    assert err == KV_Q16_OK
    assert out_new == [2]
    assert out_moved == [0]
    assert k == [9, 8, 7]
    assert v == [6, 5, 4]


def test_alias_and_error_paths_do_not_publish_outputs():
    out_a = [11]
    out_b = [22]

    assert (
        kv_cache_q16_token_shift_left_checked_nopartial_commit_only_preflight_only(None, [], 1, 1, 1, 1, 1, 0, [0], [0])
        == KV_Q16_ERR_NULL_PTR
    )

    both = [1, 2, 3, 4]
    assert (
        kv_cache_q16_token_shift_left_checked_nopartial_commit_only_preflight_only(both, both, 1, 2, 1, 2, 1, 0, [0], [0])
        == KV_Q16_ERR_BAD_PARAM
    )

    assert (
        kv_cache_q16_token_shift_left_checked_nopartial_commit_only_preflight_only([1, 2], [3, 4], 1, 1, 1, 2, 1, 0, out_a, out_a)
        == KV_Q16_ERR_BAD_PARAM
    )

    err = kv_cache_q16_token_shift_left_checked_nopartial_commit_only_preflight_only(
        [1, 2, 3],
        [4, 5, 6],
        1,
        2,
        1,
        1,
        3,
        0,
        out_a,
        out_b,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert out_a == [11]
    assert out_b == [22]
