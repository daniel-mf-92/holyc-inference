#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergeCandidatesBuildCheckedDefault."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_merge_candidates_build_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    tokenizer_bpe_merge_candidates_build_checked,
)


def tokenizer_bpe_merge_candidates_build_checked_default(
    token_ids: list[int] | None,
    token_count: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_table_count: int,
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

    if token_count > I64_MAX or rank_table_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_token_capacity = token_count
    derived_rank_table_capacity = rank_table_count
    derived_candidate_capacity = 0 if token_count < 2 else token_count - 1

    return tokenizer_bpe_merge_candidates_build_checked(
        token_ids,
        token_count,
        derived_token_capacity,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        derived_rank_table_capacity,
        out_left_tokens,
        out_right_tokens,
        out_left_indices,
        out_ranks,
        derived_candidate_capacity,
        out_candidate_count,
    )


def run_case(
    token_ids: list[int] | None,
    token_count: int,
    rank_left: list[int] | None,
    rank_right: list[int] | None,
    rank_values: list[int] | None,
    rank_count: int,
) -> None:
    cap = max(1, token_count + 6)

    out_left_core = [100000 + i for i in range(cap)]
    out_right_core = [200000 + i for i in range(cap)]
    out_idx_core = [300000 + i for i in range(cap)]
    out_rank_core = [400000 + i for i in range(cap)]
    out_count_core = [500000]

    out_left_def = out_left_core.copy()
    out_right_def = out_right_core.copy()
    out_idx_def = out_idx_core.copy()
    out_rank_def = out_rank_core.copy()
    out_count_def = out_count_core.copy()

    derived_capacity = 0 if token_count < 2 else token_count - 1

    err_core = tokenizer_bpe_merge_candidates_build_checked(
        token_ids,
        token_count,
        token_count,
        rank_left,
        rank_right,
        rank_values,
        rank_count,
        rank_count,
        out_left_core,
        out_right_core,
        out_idx_core,
        out_rank_core,
        derived_capacity,
        out_count_core,
    )
    err_def = tokenizer_bpe_merge_candidates_build_checked_default(
        token_ids,
        token_count,
        rank_left,
        rank_right,
        rank_values,
        rank_count,
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


def test_known_vectors_parity() -> None:
    run_case([10, 20, 30, 20], 4, [10, 20, 30], [20, 30, 20], [7, 4, 9], 3)
    run_case([7, 8, 7, 8, 9], 5, [7, 7, 8, 8], [8, 8, 7, 9], [12, 3, 6, 2], 4)
    run_case([42], 1, [1], [2], [3], 1)
    run_case([], 0, [], [], [], 0)


def test_malformed_span_and_rank_inputs() -> None:
    run_case([11, 12, 13], 3, None, None, None, 0)
    run_case([9, 9, 9], 3, [9, 9], [9, 9], [8, 7], 2)


def test_overflow_and_null_ptr_contract() -> None:
    baseline_left = [91, 92, 93]
    baseline_right = [81, 82, 83]
    baseline_idx = [71, 72, 73]
    baseline_rank = [61, 62, 63]
    baseline_count = [51]

    err = tokenizer_bpe_merge_candidates_build_checked_default(
        [1, 2],
        I64_MAX + 1,
        [],
        [],
        [],
        0,
        baseline_left,
        baseline_right,
        baseline_idx,
        baseline_rank,
        baseline_count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert baseline_left == [91, 92, 93]
    assert baseline_right == [81, 82, 83]
    assert baseline_idx == [71, 72, 73]
    assert baseline_rank == [61, 62, 63]
    assert baseline_count[0] == 51

    err2 = tokenizer_bpe_merge_candidates_build_checked_default(
        [1, 2],
        2,
        [],
        [],
        [],
        I64_MAX + 1,
        baseline_left,
        baseline_right,
        baseline_idx,
        baseline_rank,
        baseline_count,
    )
    assert err2 == TOKENIZER_BPE_ERR_OVERFLOW

    err3 = tokenizer_bpe_merge_candidates_build_checked_default(
        None,
        0,
        [],
        [],
        [],
        0,
        [],
        [],
        [],
        [],
        [1],
    )
    assert err3 == TOKENIZER_BPE_ERR_NULL_PTR


def test_randomized_parity() -> None:
    rng = random.Random(20260418_394)

    for _ in range(5000):
        token_count = rng.randint(0, 96)
        rank_count = rng.randint(0, 240)

        token_ids = [rng.randint(0, 220) for _ in range(token_count)]

        rows = [
            (rng.randint(0, 220), rng.randint(0, 220), rng.randint(0, 12000))
            for _ in range(rank_count)
        ]
        rows.sort(key=lambda row: (row[0], row[1]))
        rank_left = [row[0] for row in rows]
        rank_right = [row[1] for row in rows]
        rank_values = [row[2] for row in rows]

        run_case(
            token_ids,
            token_count,
            rank_left,
            rank_right,
            rank_values,
            rank_count,
        )


if __name__ == "__main__":
    test_known_vectors_parity()
    test_malformed_span_and_rank_inputs()
    test_overflow_and_null_ptr_contract()
    test_randomized_parity()
    print("tokenizer_bpe_merge_candidates_build_checked_default_reference_checks=ok")
