#!/usr/bin/env python3
"""Harness for IQ-1113 dispatch commit-only parity wrapper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_inference_forward_dispatch_model_arch_q16_checked_nopartial_commit_only_preflight_only import (
    DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
    DISPATCH_ARCH_Q16_ID_LLAMA,
    DISPATCH_ARCH_Q16_ID_MISTRAL,
    DISPATCH_ARCH_Q16_ID_QWEN2,
    DISPATCH_ARCH_Q16_ID_PHI3,
    SAMPLING_Q16_OK,
    SAMPLING_Q16_ERR_BAD_PARAM,
    SAMPLING_Q16_ERR_NULL_PTR,
    SAMPLING_Q16_ERR_OVERFLOW,
    dispatch_model_arch_commit_only_preflight_only_model,
)
from test_inference_forward_dispatch_model_arch_q16_checked_nopartial_commit_only_preflight_only_parity import (
    dispatch_model_arch_commit_only_preflight_only_parity_reference,
)


def dispatch_model_arch_commit_only_preflight_only_parity_commit_only_model(
    *,
    meta_key: bytes | None,
    arch_value: bytes | None,
    row_count: int,
    lane_count: int,
    out_rows_processed: list[int] | None,
    out_logits_written: list[int] | None,
) -> int:
    if (
        meta_key is None
        or arch_value is None
        or out_rows_processed is None
        or out_logits_written is None
    ):
        return SAMPLING_Q16_ERR_NULL_PTR

    if out_rows_processed is out_logits_written:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if row_count < 0 or lane_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM

    snapshot_meta_key = meta_key
    snapshot_arch_value = arch_value
    snapshot_row_count = row_count
    snapshot_lane_count = lane_count
    snapshot_out_rows_processed = out_rows_processed
    snapshot_out_logits_written = out_logits_written

    staged_rows = [0]
    staged_logits = [0]
    status, _ = dispatch_model_arch_commit_only_preflight_only_parity_reference(
        meta_key=meta_key,
        arch_value=arch_value,
        row_count=row_count,
        lane_count=lane_count,
        out_rows_processed=staged_rows,
        out_logits_written=staged_logits,
    )
    if status != SAMPLING_Q16_OK:
        return status

    canonical_rows = [0]
    canonical_logits = [0]
    status, _ = dispatch_model_arch_commit_only_preflight_only_model(
        meta_key=meta_key,
        arch_value=arch_value,
        row_count=row_count,
        lane_count=lane_count,
        out_rows_processed=canonical_rows,
        out_logits_written=canonical_logits,
    )
    if status != SAMPLING_Q16_OK:
        return status

    if (
        staged_rows[0] < 0
        or staged_logits[0] < 0
        or canonical_rows[0] < 0
        or canonical_logits[0] < 0
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if (
        snapshot_meta_key is not meta_key
        or snapshot_arch_value is not arch_value
        or snapshot_row_count != row_count
        or snapshot_lane_count != lane_count
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if (
        snapshot_out_rows_processed is not out_rows_processed
        or snapshot_out_logits_written is not out_logits_written
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if (
        staged_rows[0] != canonical_rows[0]
        or staged_logits[0] != canonical_logits[0]
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    out_rows_processed[0] = staged_rows[0]
    out_logits_written[0] = staged_logits[0]
    return SAMPLING_Q16_OK


def explicit_checked_composition(**kwargs) -> int:
    return dispatch_model_arch_commit_only_preflight_only_parity_commit_only_model(**kwargs)


def test_source_contains_iq1113_signature_and_contract() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    sig = "I32 InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 InferenceGenerateTokensCheckedTopKTopPCommitOnlyPreflightOnlyParity(", 1)[0]

    assert "IQ-1113 commit-only diagnostics wrapper for architecture dispatch parity." in source
    assert "InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "if (snapshot_meta_key_bytes != meta_key_bytes ||" in body
    assert "if (snapshot_out_rows_processed != out_rows_processed ||" in body
    assert "if (staged_rows_processed != canonical_rows_processed ||" in body
    assert "*out_rows_processed = staged_rows_processed;" in body


def test_supported_architectures_publish_expected_tuple() -> None:
    vectors = [
        (b"llama", DISPATCH_ARCH_Q16_ID_LLAMA),
        (b"mistral", DISPATCH_ARCH_Q16_ID_MISTRAL),
        (b"qwen2", DISPATCH_ARCH_Q16_ID_QWEN2),
        (b"qwen", DISPATCH_ARCH_Q16_ID_QWEN2),
        (b"phi3", DISPATCH_ARCH_Q16_ID_PHI3),
    ]

    for arch_tag, _expected_arch_id in vectors:
        rows_out = [701]
        logits_out = [702]
        status = dispatch_model_arch_commit_only_preflight_only_parity_commit_only_model(
            meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
            arch_value=arch_tag,
            row_count=13,
            lane_count=17,
            out_rows_processed=rows_out,
            out_logits_written=logits_out,
        )
        assert status == SAMPLING_Q16_OK
        assert rows_out[0] == 13
        assert logits_out[0] == 221


def test_bad_param_vectors_and_no_partial_publish() -> None:
    rows_out = [411]
    logits_out = [412]

    status = dispatch_model_arch_commit_only_preflight_only_parity_commit_only_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"gemma",
        row_count=3,
        lane_count=5,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows_out == [411]
    assert logits_out == [412]

    status = dispatch_model_arch_commit_only_preflight_only_parity_commit_only_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"llama",
        row_count=-1,
        lane_count=5,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert rows_out == [411]
    assert logits_out == [412]

    aliased = [901]
    status = dispatch_model_arch_commit_only_preflight_only_parity_commit_only_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"llama",
        row_count=3,
        lane_count=5,
        out_rows_processed=aliased,
        out_logits_written=aliased,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM

    status = dispatch_model_arch_commit_only_preflight_only_parity_commit_only_model(
        meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
        arch_value=b"phi3",
        row_count=1 << 62,
        lane_count=8,
        out_rows_processed=rows_out,
        out_logits_written=logits_out,
    )
    assert status == SAMPLING_Q16_ERR_OVERFLOW
    assert rows_out == [411]
    assert logits_out == [412]


def test_randomized_parity_vectors() -> None:
    rng = random.Random(20260422_1113)
    tags = [b"llama", b"mistral", b"qwen2", b"qwen", b"phi3", b"gemma"]

    for _ in range(600):
        rows = rng.randint(0, 10_000)
        lanes = rng.randint(0, 10_000)
        tag = rng.choice(tags)

        out_rows = [1001]
        out_logits = [1002]
        err_new = dispatch_model_arch_commit_only_preflight_only_parity_commit_only_model(
            meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
            arch_value=tag,
            row_count=rows,
            lane_count=lanes,
            out_rows_processed=out_rows,
            out_logits_written=out_logits,
        )

        ref_rows = [2001]
        ref_logits = [2002]
        err_ref = explicit_checked_composition(
            meta_key=DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE,
            arch_value=tag,
            row_count=rows,
            lane_count=lanes,
            out_rows_processed=ref_rows,
            out_logits_written=ref_logits,
        )

        assert err_new == err_ref
        if err_new == SAMPLING_Q16_OK:
            assert out_rows == ref_rows
            assert out_logits == ref_logits
        else:
            assert out_rows == [1001]
            assert out_logits == [1002]


if __name__ == "__main__":
    test_source_contains_iq1113_signature_and_contract()
    test_supported_architectures_publish_expected_tuple()
    test_bad_param_vectors_and_no_partial_publish()
    test_randomized_parity_vectors()
    print(
        "inference_forward_dispatch_model_arch_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only=ok"
    )
