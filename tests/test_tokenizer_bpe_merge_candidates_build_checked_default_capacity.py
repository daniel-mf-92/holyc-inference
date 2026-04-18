#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergeCandidatesBuildCheckedDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_merge_candidates_build_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
    tokenizer_bpe_merge_candidates_build_checked,
)


def tokenizer_bpe_merge_candidates_build_checked_default_capacity(
    token_ids: list[int] | None,
    token_count: int,
    token_capacity: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_table_count: int,
    rank_table_capacity: int,
    out_left_tokens: list[int] | None,
    out_right_tokens: list[int] | None,
    out_left_indices: list[int] | None,
    out_ranks: list[int] | None,
    out_candidate_count: list[int] | None,
) -> int:
    if (
        token_ids is None
        or out_left_tokens is None
        or out_right_tokens is None
        or out_left_indices is None
        or out_ranks is None
        or out_candidate_count is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_capacity > I64_MAX or rank_table_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    # Canonical adjacent-pair bound: N tokens can produce at most N-1 pairs.
    if token_count < 2:
        derived_capacity = 0
    else:
        derived_capacity = token_count - 1

    return tokenizer_bpe_merge_candidates_build_checked(
        token_ids,
        token_count,
        token_capacity,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        rank_table_capacity,
        out_left_tokens,
        out_right_tokens,
        out_left_indices,
        out_ranks,
        derived_capacity,
        out_candidate_count,
    )


def run_case(
    token_ids: list[int] | None,
    token_count: int,
    token_capacity: int,
    rank_left: list[int] | None,
    rank_right: list[int] | None,
    rank_values: list[int] | None,
    rank_count: int,
    rank_capacity: int,
) -> None:
    out_left_core = [0x11111111] * 128
    out_right_core = [0x22222222] * 128
    out_idx_core = [0x33333333] * 128
    out_rank_core = [0x44444444] * 128
    out_count_core = [0x55555555]

    out_left_def = out_left_core.copy()
    out_right_def = out_right_core.copy()
    out_idx_def = out_idx_core.copy()
    out_rank_def = out_rank_core.copy()
    out_count_def = out_count_core.copy()

    derived_capacity = 0 if token_count < 2 else token_count - 1

    err_core = tokenizer_bpe_merge_candidates_build_checked(
        token_ids,
        token_count,
        token_capacity,
        rank_left,
        rank_right,
        rank_values,
        rank_count,
        rank_capacity,
        out_left_core,
        out_right_core,
        out_idx_core,
        out_rank_core,
        derived_capacity,
        out_count_core,
    )
    err_def = tokenizer_bpe_merge_candidates_build_checked_default_capacity(
        token_ids,
        token_count,
        token_capacity,
        rank_left,
        rank_right,
        rank_values,
        rank_count,
        rank_capacity,
        out_left_def,
        out_right_def,
        out_idx_def,
        out_rank_def,
        out_count_def,
    )

    assert err_def == err_core
    assert out_count_def[0] == out_count_core[0]
    assert out_left_def == out_left_core
    assert out_right_def == out_right_core
    assert out_idx_def == out_idx_core
    assert out_rank_def == out_rank_core


def test_ranked_fixture_parity() -> None:
    run_case(
        [10, 20, 30, 20],
        4,
        4,
        [10, 20, 30],
        [20, 30, 20],
        [7, 4, 9],
        3,
        3,
    )
    run_case(
        [7, 8, 7, 8, 9],
        5,
        5,
        [7, 7, 8, 8],
        [8, 8, 7, 9],
        [12, 3, 6, 2],
        4,
        4,
    )


def test_underflow_clamp_for_short_spans() -> None:
    run_case([42], 1, 1, [1], [2], [3], 1, 1)
    run_case([], 0, 0, [], [], [], 0, 0)


def test_malformed_boundaries_and_no_partial_on_error() -> None:
    out_left = [91, 92, 93]
    out_right = [81, 82, 83]
    out_idx = [71, 72, 73]
    out_rank = [61, 62, 63]
    out_count = [51]

    err = tokenizer_bpe_merge_candidates_build_checked_default_capacity(
        [10, 20, 30, 40],
        5,
        4,
        [10, 20, 30],
        [20, 30, 40],
        [1, 2, 3],
        3,
        3,
        out_left,
        out_right,
        out_idx,
        out_rank,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out_left == [91, 92, 93]
    assert out_right == [81, 82, 83]
    assert out_idx == [71, 72, 73]
    assert out_rank == [61, 62, 63]
    assert out_count[0] == 51

    err2 = tokenizer_bpe_merge_candidates_build_checked_default_capacity(
        [1, 2],
        2,
        I64_MAX + 1,
        [],
        [],
        [],
        0,
        0,
        out_left,
        out_right,
        out_idx,
        out_rank,
        out_count,
    )
    assert err2 == TOKENIZER_BPE_ERR_OVERFLOW
    assert out_count[0] == 51


def test_null_ptr_contract() -> None:
    err = tokenizer_bpe_merge_candidates_build_checked_default_capacity(
        None,
        0,
        0,
        [],
        [],
        [],
        0,
        0,
        [],
        [],
        [],
        [],
        [1],
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR


def test_randomized_parity() -> None:
    rng = random.Random(20260418_379)

    for _ in range(4000):
        token_count = rng.randint(0, 80)
        token_capacity = token_count + rng.randint(0, 8)
        rank_count = rng.randint(0, 200)
        rank_capacity = rank_count + rng.randint(0, 8)

        tokens = [rng.randint(-200, 200) for _ in range(token_capacity)]
        rank_left = [rng.randint(-200, 200) for _ in range(rank_capacity)]
        rank_right = [rng.randint(-200, 200) for _ in range(rank_capacity)]
        ranks = [rng.randint(0, 5000) for _ in range(rank_capacity)]

        run_case(
            tokens,
            token_count,
            token_capacity,
            rank_left,
            rank_right,
            ranks,
            rank_count,
            rank_capacity,
        )


def main() -> None:
    test_ranked_fixture_parity()
    test_underflow_clamp_for_short_spans()
    test_malformed_boundaries_and_no_partial_on_error()
    test_null_ptr_contract()
    test_randomized_parity()
    print("ok")


if __name__ == "__main__":
    main()
