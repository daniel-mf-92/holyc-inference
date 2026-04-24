#!/usr/bin/env python3
"""Parity harness for KVCacheQ16TokenShiftLeftCheckedNoPartialCommitOnly (IQ-1282)."""

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

SRC = Path(__file__).resolve().parents[1] / "src/model/kv_cache.HC"


def _extract_fn(name: str) -> str:
    text = SRC.read_text()
    m = re.search(rf"I32\\s+{name}\\s*\\([^)]*\\)\\s*\\{{", text)
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


def kv_cache_q16_token_shift_left_checked_nopartial_commit_only(
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
    snapshot_layer_count = layer_count
    snapshot_token_capacity = token_capacity
    snapshot_kv_heads = kv_heads
    snapshot_head_dim = head_dim

    if layer_count * token_capacity * kv_heads * head_dim:
        total_cells = layer_count * token_capacity * kv_heads * head_dim
        if len(k_cache) < total_cells or len(v_cache) < total_cells:
            return KV_Q16_ERR_BAD_PARAM

    sim = _sim_shift(k_cache, v_cache, layer_count, token_capacity, kv_heads, head_dim, used_tokens, shift_tokens)
    if sim == "bad":
        return KV_Q16_ERR_BAD_PARAM

    outk, outv, staged_new_used_tokens, staged_moved_cells = sim

    if (
        snapshot_used_tokens != used_tokens
        or snapshot_layer_count != layer_count
        or snapshot_token_capacity != token_capacity
        or snapshot_kv_heads != kv_heads
        or snapshot_head_dim != head_dim
    ):
        return KV_Q16_ERR_BAD_PARAM

    if staged_new_used_tokens < 0 or staged_new_used_tokens > used_tokens:
        return KV_Q16_ERR_BAD_PARAM
    if staged_moved_cells < 0:
        return KV_Q16_ERR_BAD_PARAM

    k_cache[:] = outk
    v_cache[:] = outv
    out_new_used_tokens[0] = staged_new_used_tokens
    out_moved_cells[0] = staged_moved_cells
    return KV_Q16_OK


def explicit_commit_only_composition(
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
    snapshot_layer_count = layer_count
    snapshot_token_capacity = token_capacity
    snapshot_kv_heads = kv_heads
    snapshot_head_dim = head_dim

    sim = _sim_shift(k_cache, v_cache, layer_count, token_capacity, kv_heads, head_dim, used_tokens, shift_tokens)
    if sim == "bad":
        return KV_Q16_ERR_BAD_PARAM
    outk, outv, staged_new_used_tokens, staged_moved_cells = sim

    if (
        snapshot_used_tokens != used_tokens
        or snapshot_layer_count != layer_count
        or snapshot_token_capacity != token_capacity
        or snapshot_kv_heads != kv_heads
        or snapshot_head_dim != head_dim
    ):
        return KV_Q16_ERR_BAD_PARAM

    if staged_new_used_tokens < 0 or staged_new_used_tokens > used_tokens:
        return KV_Q16_ERR_BAD_PARAM
    if staged_moved_cells < 0:
        return KV_Q16_ERR_BAD_PARAM

    k_cache[:] = outk
    v_cache[:] = outv
    out_new_used_tokens[0] = staged_new_used_tokens
    out_moved_cells[0] = staged_moved_cells
    return KV_Q16_OK


def test_function_present_and_key_guards():
    fn = _extract_fn("KVCacheQ16TokenShiftLeftCheckedNoPartialCommitOnly")
    for needle in [
        "if (k_cache == v_cache)",
        "if (out_new_used_tokens == out_moved_cells)",
        "snapshot_used_tokens = used_tokens",
        "status = KVCacheQ16TokenShiftLeftCheckedNoPartial(",
        "if (snapshot_used_tokens != used_tokens",
        "*out_new_used_tokens = staged_new_used_tokens",
        "*out_moved_cells = staged_moved_cells",
    ]:
        assert needle in fn


def test_alias_and_null_guards():
    out_a = [0]
    assert (
        kv_cache_q16_token_shift_left_checked_nopartial_commit_only(None, [], 1, 1, 1, 1, 1, 1, [0], [0])
        == KV_Q16_ERR_NULL_PTR
    )
    both = [1, 2]
    assert (
        kv_cache_q16_token_shift_left_checked_nopartial_commit_only(both, both, 1, 1, 1, 1, 1, 1, [0], [0])
        == KV_Q16_ERR_BAD_PARAM
    )
    assert (
        kv_cache_q16_token_shift_left_checked_nopartial_commit_only([1, 2], [3, 4], 1, 1, 1, 1, 1, 1, out_a, out_a)
        == KV_Q16_ERR_BAD_PARAM
    )


def test_commit_only_matches_explicit_composition_and_outputs():
    layer_count, token_capacity, kv_heads, head_dim = 2, 5, 2, 3
    total_cells = layer_count * token_capacity * kv_heads * head_dim
    k0 = list(range(1, total_cells + 1))
    v0 = list(range(1000, 1000 + total_cells))

    k_a, v_a = k0[:], v0[:]
    k_b, v_b = k0[:], v0[:]

    out_new_a = [123]
    out_moved_a = [456]
    out_new_b = [0]
    out_moved_b = [0]

    err_a = kv_cache_q16_token_shift_left_checked_nopartial_commit_only(
        k_a, v_a, layer_count, token_capacity, kv_heads, head_dim, 4, 1, out_new_a, out_moved_a
    )
    err_b = explicit_commit_only_composition(
        k_b, v_b, layer_count, token_capacity, kv_heads, head_dim, 4, 1, out_new_b, out_moved_b
    )

    assert err_a == KV_Q16_OK
    assert err_b == KV_Q16_OK
    assert out_new_a == out_new_b == [3]
    assert out_moved_a == out_moved_b == [3 * layer_count * kv_heads * head_dim]
    assert k_a == k_b
    assert v_a == v_b


def test_full_shift_scrubs_to_zero_and_reports_zero_move():
    layer_count, token_capacity, kv_heads, head_dim = 1, 4, 2, 2
    total_cells = layer_count * token_capacity * kv_heads * head_dim
    k = [7] * total_cells
    v = [9] * total_cells
    out_new = [99]
    out_moved = [99]

    err = kv_cache_q16_token_shift_left_checked_nopartial_commit_only(
        k, v, layer_count, token_capacity, kv_heads, head_dim, 3, 3, out_new, out_moved
    )

    assert err == KV_Q16_OK
    assert out_new == [0]
    assert out_moved == [0]
    assert k == [0] * total_cells
    assert v == [0] * total_cells
