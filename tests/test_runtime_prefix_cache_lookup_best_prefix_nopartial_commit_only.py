#!/usr/bin/env python3
from pathlib import Path

SRC = Path('src/runtime/prefix_cache.HC')


def extract_function(name: str) -> str:
    text = SRC.read_text()
    start = text.find(f"I32 {name}(")
    assert start >= 0, f"missing {name}"
    brace = text.find('{', start)
    depth = 1
    i = brace + 1
    while i < len(text) and depth > 0:
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        i += 1
    return text[start:i]


def test_commit_only_exists_and_uses_snapshot_and_atomic_publish():
    fn = extract_function('PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnly')
    assert 'snapshot_entry_count = cache->count;' in fn
    assert 'snapshot_max_prompt_tokens = max_prompt_tokens;' in fn
    assert 'snapshot_query_hash = query_hash;' in fn
    assert 'PrefixCacheLookupBestPrefixCheckedNoPartial(' in fn
    assert 'if (cache->count != snapshot_entry_count ||' in fn
    assert 'if (max_prompt_tokens != snapshot_max_prompt_tokens ||' in fn
    assert 'if (query_hash != snapshot_query_hash)' in fn
    assert '*out_best_index = best_index;' in fn
    assert '*out_best_tokens = best_tokens;' in fn


def test_commit_only_enforces_output_alias_and_null_rejection():
    fn = extract_function('PrefixCacheLookupBestPrefixCheckedNoPartialCommitOnly')
    assert 'if (!cache || !cache->entries || !out_best_index || !out_best_tokens)' in fn
    assert 'if (out_best_index == out_best_tokens)' in fn
    assert 'return PREFIX_CACHE_ERR_NULL_PTR;' in fn


def test_iq_1296_marked_done():
    master = Path('MASTER_TASKS.md').read_text()
    assert '- [x] IQ-1296' in master
