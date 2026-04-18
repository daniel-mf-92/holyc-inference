#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergeCandidatesBuildCheckedDefaultNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_merge_candidates_build_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
    tokenizer_bpe_merge_candidates_build_checked,
)


def tokenizer_bpe_merge_candidates_build_checked_default_no_partial(
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

    # Stage all caller-visible outputs so failures do not partially mutate state.
    staged_left = out_left_tokens.copy()
    staged_right = out_right_tokens.copy()
    staged_indices = out_left_indices.copy()
    staged_ranks = out_ranks.copy()
    staged_count = [out_candidate_count[0]]

    err = tokenizer_bpe_merge_candidates_build_checked(
        token_ids,
        token_count,
        derived_token_capacity,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        derived_rank_table_capacity,
        staged_left,
        staged_right,
        staged_indices,
        staged_ranks,
        derived_candidate_capacity,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    out_left_tokens[:] = staged_left
    out_right_tokens[:] = staged_right
    out_left_indices[:] = staged_indices
    out_ranks[:] = staged_ranks
    out_candidate_count[0] = staged_count[0]
    return TOKENIZER_BPE_OK


def run_case(
    token_ids: list[int] | None,
    token_count: int,
    rank_left: list[int] | None,
    rank_right: list[int] | None,
    rank_values: list[int] | None,
    rank_count: int,
    seed: int,
) -> None:
    cap = max(1, token_count + 6)

    expected_left = [100000 + seed + i for i in range(cap)]
    expected_right = [200000 + seed + i for i in range(cap)]
    expected_indices = [300000 + seed + i for i in range(cap)]
    expected_ranks = [400000 + seed + i for i in range(cap)]
    expected_count = [500000 + seed]

    actual_left = expected_left.copy()
    actual_right = expected_right.copy()
    actual_indices = expected_indices.copy()
    actual_ranks = expected_ranks.copy()
    actual_count = expected_count.copy()

    # Explicit staged baseline for no-partial default composition.
    derived_candidate_capacity = 0 if token_count < 2 else token_count - 1
    staged_left = expected_left.copy()
    staged_right = expected_right.copy()
    staged_indices = expected_indices.copy()
    staged_ranks = expected_ranks.copy()
    staged_count = [expected_count[0]]

    err_expected = tokenizer_bpe_merge_candidates_build_checked(
        token_ids,
        token_count,
        token_count,
        rank_left,
        rank_right,
        rank_values,
        rank_count,
        rank_count,
        staged_left,
        staged_right,
        staged_indices,
        staged_ranks,
        derived_candidate_capacity,
        staged_count,
    )
    if err_expected == TOKENIZER_BPE_OK:
        expected_left = staged_left
        expected_right = staged_right
        expected_indices = staged_indices
        expected_ranks = staged_ranks
        expected_count[0] = staged_count[0]

    err_actual = tokenizer_bpe_merge_candidates_build_checked_default_no_partial(
        token_ids,
        token_count,
        rank_left,
        rank_right,
        rank_values,
        rank_count,
        actual_left,
        actual_right,
        actual_indices,
        actual_ranks,
        actual_count,
    )

    assert err_actual == err_expected
    assert actual_left == expected_left
    assert actual_right == expected_right
    assert actual_indices == expected_indices
    assert actual_ranks == expected_ranks
    assert actual_count[0] == expected_count[0]


def test_known_vectors_parity_vs_explicit_staged_composition() -> None:
    run_case([10, 20, 30, 20], 4, [10, 20, 30], [20, 30, 20], [7, 4, 9], 3, 1)
    run_case([7, 8, 7, 8, 9], 5, [7, 7, 8, 8], [8, 8, 7, 9], [12, 3, 6, 2], 4, 2)
    run_case([42], 1, [1], [2], [3], 1, 3)
    run_case([], 0, [], [], [], 0, 4)


def test_error_paths_preserve_exact_output_state() -> None:
    left = [91, 92, 93]
    right = [81, 82, 83]
    indices = [71, 72, 73]
    ranks = [61, 62, 63]
    count = [51]

    err = tokenizer_bpe_merge_candidates_build_checked_default_no_partial(
        [10, 20],
        2,
        None,
        None,
        None,
        1,
        left,
        right,
        indices,
        ranks,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert left == [91, 92, 93]
    assert right == [81, 82, 83]
    assert indices == [71, 72, 73]
    assert ranks == [61, 62, 63]
    assert count[0] == 51

    err2 = tokenizer_bpe_merge_candidates_build_checked_default_no_partial(
        [1, 2],
        I64_MAX + 1,
        [],
        [],
        [],
        0,
        left,
        right,
        indices,
        ranks,
        count,
    )
    assert err2 == TOKENIZER_BPE_ERR_OVERFLOW
    assert left == [91, 92, 93]
    assert right == [81, 82, 83]
    assert indices == [71, 72, 73]
    assert ranks == [61, 62, 63]
    assert count[0] == 51


def test_null_ptr_contract() -> None:
    err = tokenizer_bpe_merge_candidates_build_checked_default_no_partial(
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
    assert err == TOKENIZER_BPE_ERR_NULL_PTR


def test_randomized_parity() -> None:
    rng = random.Random(20260418_407)

    for i in range(5000):
        token_count = rng.randint(0, 80)
        rank_count = rng.randint(0, 220)

        tokens = [rng.randint(0, 150) for _ in range(token_count)]

        raw = [
            (rng.randint(0, 150), rng.randint(0, 150), rng.randint(0, 10000))
            for _ in range(rank_count)
        ]
        raw.sort(key=lambda row: (row[0], row[1]))
        left = [r[0] for r in raw]
        right = [r[1] for r in raw]
        values = [r[2] for r in raw]

        run_case(
            tokens,
            len(tokens),
            left,
            right,
            values,
            len(values),
            i + 20,
        )


if __name__ == "__main__":
    test_known_vectors_parity_vs_explicit_staged_composition()
    test_error_paths_preserve_exact_output_state()
    test_null_ptr_contract()
    test_randomized_parity()
    print("tokenizer_bpe_merge_candidates_build_checked_default_no_partial_reference_checks=ok")
