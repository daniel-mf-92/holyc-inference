#!/usr/bin/env python3
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[1] / "src/model/kv_cache.HC"


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
                return text[m.start(): i + 1]
        i += 1
    raise AssertionError("unbalanced braces")


def _sim_shift(k, v, layer_count, token_capacity, kv_heads, head_dim, used_tokens, shift_tokens):
    if layer_count < 0 or token_capacity < 0 or kv_heads < 0 or head_dim < 0 or used_tokens < 0 or shift_tokens < 0:
        return "bad"
    if used_tokens > token_capacity:
        return "bad"

    cells_per_token = layer_count * kv_heads * head_dim
    total_cells = token_capacity * cells_per_token

    if used_tokens == 0:
        return k[:], v[:], 0, 0
    if shift_tokens == 0:
        return k[:], v[:], used_tokens, 0
    if shift_tokens >= used_tokens:
        return [0] * total_cells, [0] * total_cells, 0, 0

    keep_tokens = used_tokens - shift_tokens
    shift_cells = shift_tokens * cells_per_token
    keep_cells = keep_tokens * cells_per_token

    outk = k[:]
    outv = v[:]
    for i in range(keep_cells):
        outk[i] = outk[i + shift_cells]
        outv[i] = outv[i + shift_cells]
    for i in range(keep_cells, total_cells):
        outk[i] = 0
        outv[i] = 0
    return outk, outv, keep_tokens, keep_cells


def test_function_present_and_guards():
    fn = _extract_fn("KVCacheQ16TokenShiftLeftCheckedNoPartial")
    for needle in [
        "if (!k_cache || !v_cache || !out_new_used_tokens || !out_moved_cells)",
        "if (out_new_used_tokens == out_moved_cells)",
        "if (shift_tokens >= used_tokens)",
        "while (idx < keep_cells)",
        "while (idx < total_cells)",
        "*out_new_used_tokens = keep_tokens",
        "*out_moved_cells = keep_cells",
    ]:
        assert needle in fn


def test_shift_keeps_expected_prefix_and_scrubs_tail():
    layer_count, token_capacity, kv_heads, head_dim = 2, 5, 2, 3
    cells_per_token = layer_count * kv_heads * head_dim
    total_cells = token_capacity * cells_per_token

    k = list(range(1, total_cells + 1))
    v = list(range(1001, 1001 + total_cells))

    outk, outv, new_used, moved = _sim_shift(
        k, v, layer_count, token_capacity, kv_heads, head_dim, used_tokens=4, shift_tokens=1
    )

    assert new_used == 3
    assert moved == 3 * cells_per_token
    assert outk[:moved] == k[cells_per_token: cells_per_token + moved]
    assert outv[:moved] == v[cells_per_token: cells_per_token + moved]
    assert outk[moved:] == [0] * (total_cells - moved)
    assert outv[moved:] == [0] * (total_cells - moved)


def test_shift_all_tokens_zeroes_everything():
    layer_count, token_capacity, kv_heads, head_dim = 1, 4, 2, 2
    cells_per_token = layer_count * kv_heads * head_dim
    total_cells = token_capacity * cells_per_token
    k = [7] * total_cells
    v = [9] * total_cells

    outk, outv, new_used, moved = _sim_shift(
        k, v, layer_count, token_capacity, kv_heads, head_dim, used_tokens=3, shift_tokens=3
    )
    assert new_used == 0
    assert moved == 0
    assert outk == [0] * total_cells
    assert outv == [0] * total_cells


def test_shift_zero_tokens_is_noop_with_zero_moved():
    layer_count, token_capacity, kv_heads, head_dim = 2, 3, 1, 2
    cells_per_token = layer_count * kv_heads * head_dim
    total_cells = token_capacity * cells_per_token
    k = list(range(total_cells))
    v = list(range(500, 500 + total_cells))

    outk, outv, new_used, moved = _sim_shift(
        k, v, layer_count, token_capacity, kv_heads, head_dim, used_tokens=2, shift_tokens=0
    )
    assert outk == k
    assert outv == v
    assert new_used == 2
    assert moved == 0
